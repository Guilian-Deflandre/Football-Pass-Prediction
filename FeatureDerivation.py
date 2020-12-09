# ! /usr/bin/env python
# -*- coding: utf-8 -*-
import time
import datetime
from contextlib import contextmanager

import pandas as pd
import numpy as np

import math


@contextmanager
def measure_time(label):
    """
    Context manager to measure time of computation.
    >>> with measure_time('Heavy computation'):
    >>>     do_heavy_computation()
    'Duration of [Heavy computation]: 0:04:07.765971'

    Parameters
    ----------
    label: str
        The label by which the computation will be referred
    """
    start = time.time()
    yield
    end = time.time()
    print('Duration of [{}]: {}'.format(label,
                                        datetime.timedelta(seconds=end-start)))


def load_from_csv(path, delimiter=','):
    """
    Load csv file and return a NumPy array of its data

    Parameters
    ----------
    path: str
        The path to the csv file to load
    delimiter: str (default: ',')
        The csv field delimiter

    Return
    ------
    D: array
        The NumPy array of the data contained in the file
    """
    return pd.read_csv(path, delimiter=delimiter)


def same_team_(sender, player_j):
    if sender <= 11:
        return int(player_j <= 11)
    else:
        return int(player_j > 11)


def build_distance_matrix(pass_):
    '''
    param : pass vector
    return : 22x22 matrix of elements e_ij = distance(player_i, player_j)
    '''
    positions = pass_.drop(["sender", "time_start", "Id"])
    pos_size = np.size(positions)
    s = (int(pos_size/2), int(pos_size/2))
    distance_matrix = np.zeros(s)

    n = 2
    positions = [(positions[k:k+n]) for k in range(0, len(positions), n)]
    for i in range(int(pos_size/2)):
        for j in range(int(pos_size/2)):
            distance_matrix[i, j] = np.sqrt(
                (positions[i][0]-positions[j][0])**2 +
                (positions[i][1]-positions[j][1])**2)

    return distance_matrix


def distance_to_opp(sender, player, dist_mat):
    '''
    param : sender id; player id; distance matrix
    return : distances between player and the two closest opponents of the
    sender
    '''
    if(same_team_(sender, player) == 0):
        distance = (0, 0)
    else:
        row = player-1
        team = int(sender/22 <= 0.5)
        start = team*11
        opp_dist = dist_mat[row, start: start+11]
        distance = (min(opp_dist), min(np.delete(opp_dist,
                                                 np.argmin(opp_dist))))

    return distance


def number_of_opp(sender, player, dist_mat):
    '''
    param : sender id; player id; distance matrix
    return : number of opponents in a radius of 15 meters around the potential
    receiver
    '''
    numb_opp = 0
    radius = 1580  # Mean length of a pass

    row = player-1
    team = int(sender/22 <= 0.5)
    start = team*11
    opp_dist = dist_mat[row, start: start+11]

    for i in range(0, opp_dist.shape[0]):

        if opp_dist[i] <= radius:
            numb_opp = numb_opp + 1

    return numb_opp


def heron(sender, player, distance, dist_mat):
    '''
    param: sender id; player id; disatnce between sender and player;
           distance matrix
    return: smallest distance between the pass line and an opponent of the
            sender, using Heron's formula
    '''
    if(same_team_(sender, player) == 0):
        h = 0
    else:
        row_p = player-1
        row_s = sender-1
        team = int(sender/22 <= 0.5)
        start = team*11
        # Distances between player and opponents
        opp_dist_p = dist_mat[row_p, start: start+11]
        opp_dist_s = dist_mat[row_s, start: start+11]

        s = 0.5*(distance + opp_dist_p + opp_dist_s)

        tol = 10e-9
        sd = s-distance
        ss = s-opp_dist_s
        sp = s-opp_dist_p
        (sd)[abs(sd) < tol] = 0.0
        (ss)[abs(ss) < tol] = 0.0
        (sp)[abs(sp) < tol] = 0.0

        if distance == 0 or math.isnan(distance):
            h = min(opp_dist_s)
            return h

        area = np.sqrt(s*(sd)*(ss)*(sp))
        h = 2*area/distance
        h = min(h)

    return h


def define_zone(x, y):
    zone = 0
    if y >= -1750 and y <= 1750:
        if x >= -3200 and x <= 3200:
            zone = 1  # Central zone
        if x >= -5250 and x < -3200:
            zone = 2  # Zone but 1
        else:
            zone = 3  # Zone but 2
    else:
        if y > 1750:
            zone = 4  # Couloir haut
        else:
            zone = 5  # Couloir bas
    return zone


def make_pair_of_players(X_, y_=None):
    n_ = X_.shape[0]
    pair_feature_col = ["sender", "x_sender", "y_sender",
                        "player_j", "x_j", "y_j", "same_team", "distance",
                        "distance_opp_1", "distance_opp_2", "distance_line",
                        "nb_opp", "zone_send", "zone_rec", "x_ball_gain"]

    X_pairs = pd.DataFrame(data=np.zeros((n_*22, len(pair_feature_col))),
                           columns=pair_feature_col)
    y_pairs = pd.DataFrame(data=np.zeros((n_*22, 1)), columns=["pass"])

    # From pass to pair of players
    idx = 0
    for i in range(n_):
        p_i_ = X_.iloc[i]
        distance_matrix = build_distance_matrix(p_i_)
        x_ball_gain = compute_x_ball_gain(p_i_)
        sender = X_.iloc[i].sender
        players = np.arange(1, 23)
        other_players = np.delete(players, sender-1)
        ss_numb_opp = number_of_opp(sender, sender, distance_matrix)
        X_pairs.iloc[idx] = [sender,  p_i_["x_{:0.0f}".format(sender)],
                             p_i_["y_{:0.0f}".format(sender)], sender,
                             p_i_["x_{:0.0f}".format(sender)],
                             p_i_["y_{:0.0f}".format(sender)],
                             same_team_(sender, sender), 0, 0, 0, 0,
                             ss_numb_opp, 0, 0, 0]
        zone_sender = define_zone(p_i_["x_{:0.0f}".format(sender)],
                                  p_i_["y_{:0.0f}".format(sender)])

        idx += 1
        for player_j in other_players:
            distance = distance_matrix[sender-1, player_j-1]
            distance_opp = distance_to_opp(sender, player_j, distance_matrix)
            distance_line = heron(sender, player_j, distance, distance_matrix)
            numb_opp = number_of_opp(sender, player_j, distance_matrix)
            X_pairs.iloc[idx] = [sender,  p_i_["x_{:0.0f}".format(sender)],
                                 p_i_["y_{:0.0f}".format(sender)],
                                 player_j,
                                 p_i_["x_{:0.0f}".format(player_j)],
                                 p_i_["y_{:0.0f}".format(player_j)],
                                 same_team_(sender, player_j), distance,
                                 distance_opp[0], distance_opp[1],
                                 distance_line, numb_opp, zone_sender,
                                 define_zone(p_i_["x_{:0.0f}"
                                                  .format(player_j)],
                                             p_i_["y_{:0.0f}"
                                                  .format(player_j)]),
                                 x_ball_gain["x_{:0.0f}".format(player_j)]]

            if y_ is not None:
                y_pairs.iloc[idx]["pass"] = int(player_j == y_.iloc[i])
            idx += 1

    return X_pairs, y_pairs


def compute_x_ball_gain(pass_):
    sender = pass_["sender"]
    x_positions = pass_.drop(["sender", "time_start", "Id"])
    for i in range(1, 23):
        x_positions = x_positions.drop("y_{:0.0f}".format(i))

    leftmost_player = find_team_left_side(x_positions)
    x_gains = {}

    for i in range(len(x_positions)):
        x_gains["x_{:0.0f}".format(i+1)] = (x_positions["x_{:0.0f}"
                                                        .format(i+1)] -
                                            x_positions["x_{:0.0f}"
                                                        .format(sender)])

        if(same_team_(sender, leftmost_player) == 0):
            x_gains["x_{:0.0f}".format(i+1)] = -x_gains["x_{:0.0f}"
                                                        .format(i+1)]

    return x_gains


def find_team_left_side(x_positions):
    leftmost_key = x_positions.keys()[np.argmin(x_positions)]
    return int(leftmost_key.replace('x_', ''))


def compute_distance_(X_):
    d = np.zeros((X_.shape[0],))

    d = np.sqrt((X_["x_sender"]-X_["x_j"])**2 + (X_["y_sender"]-X_["y_j"])**2)
    return d


def write_submission(predictions=None, probas=None, estimated_score=0,
                     file_name="submission", date=True, indexes=None):
    """
    Write a submission file for the Kaggle platform

    Parameters
    ----------
    predictions: array [n_predictions, 1]
        `predictions[i]` is the prediction for player
        receiving pass `i` (or indexes[i] if given).
    probas: array [n_predictions, 22]
        `probas[i,j]` is the probability that player `j` receives
        the ball with pass `i`.
    estimated_score: float [1]
        The estimated accuracy of predictions.
    file_name: str or None (default: 'submission')
        The path to the submission file to create (or override). If none is
        provided, a default one will be used. Also note that the file extension
        (.txt) will be appended to the file.
    date: boolean (default: True)
        Whether to append the date in the file name

    Return
    ------
    file_name: path
        The final path to the submission file
    """

    if date:
        file_name = '{}_{}'.format(file_name, time.strftime('%d-%m-%Y_%Hh%M'))

    file_name = '{}.txt'.format(file_name)

    if predictions is None and probas is None:
        raise ValueError('Predictions and/or probas should be provided.')

    n_samples = 3000
    if indexes is None:
        indexes = np.arange(n_samples)

    if probas is None:
        print('Deriving probabilities from predictions.')
        probas = np.zeros((n_samples, 22))
        for i in range(n_samples):
            probas[i, predictions[i]-1] = 1

    if predictions is None:
        print('Deriving predictions from probabilities')
        predictions = np.zeros((n_samples, ))
        for i in range(n_samples):
            mask = probas[i] == np.max(probas[i])
            selected_players = np.arange(1, 23)[mask]
            predictions[i] = int(selected_players[0])

    # Writing into the file
    with open(file_name, 'w') as handle:
        # Creating header
        header = '"Id","Predicted",'
        for j in range(1, 23):
            header = header + '"P_{:0.0f}",'.format(j)
        handle.write(header[:-1]+"\n")

        # Adding your estimated score
        first_line = '"Estimation",{},'.format(estimated_score)
        for j in range(1, 23):
            first_line = first_line + '0,'
        handle.write(first_line[:-1]+"\n")

        # Adding your predictions
        for i in range(n_samples):
            line = "{},{:0.0f},".format(indexes[i], predictions[i])
            pj = probas[i, :]
            for j in range(22):
                line = line + '{},'.format(pj[j])
            handle.write(line[:-1]+"\n")

    return file_name
