# ! /usr/bin/env python
# -*- coding: utf-8 -*-
import time
import datetime
from contextlib import contextmanager

import pandas as pd
import numpy as np

import Vectors as vec

import math


@contextmanager
def measure_time(label):
    """
    Context manager to measure time of computation.
    >>> with measure_time(aHeavy computation'):
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


def angle_alkashi(b, c, a):
    """From Al-Kashi cos(A) = (b**2 + c**2 - a**2)/(2*b*c)"""
    print("la")
    return math.acos((b**2 + c**2 - a**2)/(2*b*c))


def min_angle(sender, player, dist_mat):
    """
    Parameters
    ----------
    sender : sender_id
    player : receiver_id
    dist_mat : distance matrix of a pass

    Returns
    -------
    out : minimum angle between the sender receiver line and an opponent of the
        sender.

    The function is based on Al'Kashi theorem
        """
    angle = []
    dist_sender_rec = dist_mat[sender-1][player-1]
    out = 0

    row = sender-1
    team = int(sender/22 <= 0.5)
    start = team*11
    opp_dist = dist_mat[row, start: start+11]
    opp_rec_dist = dist_mat[player-1, start: start + 11]


    for i in range(11):
        if dist_sender_rec != 0 and opp_dist[i] != 0 and dist_sender_rec != opp_dist[i]:
            a = round(opp_rec_dist[i],2)
            b = round(opp_dist[i], 2)
            c = round(dist_sender_rec,2)

            num = b**2 + c**2 - a**2
            denom = (2*b*c)
            frac = num/denom
            frac = round(frac,3)
            if frac == -1 or frac == 1:
                angle.append(0)
            else:
                angle.append(math.acos(num/denom))
    if len(angle) != 0:
        out = np.amin(angle)*(180/3.1415)

    return out


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
    row = player-1
    team = int(sender/22 <= 0.5)
    start = team*11
    opp_dist = dist_mat[row, start: start+11]
    if(same_team_(sender, player) == 1):
        distance = (min(opp_dist), min(np.delete(opp_dist,
                                                 np.argmin(opp_dist))))
    else:
        opp_dist = np.delete(opp_dist, np.argmin(opp_dist))
        distance = (min(opp_dist), min(np.delete(opp_dist,
                                                 np.argmin(opp_dist))))
    return distance


def distance_to_opp_rec(player, dist_mat):
    '''
    param : sender id; player id; distance matrix
    return : distances between player and the two closest opponents of player
    '''
    row = player-1
    team = int(player/22 <= 0.5)
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
    radius = 300  # Mean length of a pass

    row = player-1
    team = int(sender/22 <= 0.5)
    start = team*11
    opp_dist = dist_mat[row, start: start+11]

    for i in range(0, opp_dist.shape[0]):

        if opp_dist[i] <= radius:
            numb_opp = numb_opp + 1

    return numb_opp


def smallest_distance_pt_seg(opponent, sender, receiver):
    pass_vector = vec.vector(sender, receiver)
    snd2opp_vector = vec.vector(sender, opponent)
    pass_len = vec.norm(pass_vector)
    pass_vector_unit = vec.unit(pass_vector)
    snd2opp_vector_scaled = vec.scale(snd2opp_vector, 1.0/pass_len)
    t = vec.dot(pass_vector_unit, snd2opp_vector_scaled)
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0

    closest_match = vec.scale(pass_vector, t)
    smallest_dist = vec.distance(closest_match, snd2opp_vector)

    return smallest_dist


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


def is_in_attack(pass_):
    ''' Based on the x_position of a player, determines if it is in an attack
        postion (on the in the opposing camp) or not.
        
        Param: pass_, 
        Return: 1 if the sender is in an attack position, 0 otherwise
    '''
    sender = pass_["sender"]
    x_positions = pass_.drop(["sender", "time_start", "Id"])
    for i in range(1, 23):
        x_positions = x_positions.drop("y_{:0.0f}".format(i))
    
    leftmost_player = find_team_left_side(x_positions)
    sum_position_team_sender = 0
    
    for i in range(len(x_positions)):
        if(same_team_(sender, i) == 1):
            sum_position_team_sender = (sum_position_team_sender + 
                                        x_positions["x_{:0.0f}".format(i+1)])
    
    sum_position_team_sender = sum_position_team_sender/11
    
    
    if(same_team_(sender, leftmost_player) == 1):
        if(sum_position_team_sender <= 0):
            return 0
        else:
            return 1
    else:
        if(sum_position_team_sender <= 0):
            return 1
        else:
            return 0
    return 0

def find_team_left_side(x_positions):
    leftmost_key = x_positions.keys()[np.argmin(x_positions.idxmin())]
    return int(leftmost_key.replace('x_', ''))


def compute_distance_(X_):
    d = np.zeros((X_.shape[0],))

    d = np.sqrt((X_["x_sender"]-X_["x_j"])**2 + (X_["y_sender"]-X_["y_j"])**2)
    return d


def define_zone(x, y):
    zone = 0
    if y >= -1750 and y <= 1750:
        if x >= -3200 and x <= 3200:
            zone = [1, 0, 0, 0, 0]  # Central zone => 1
        if x >= -5250 and x < -3200:
            zone = [0, 1, 0, 0, 0]  # Zone but 1 => 2
        else:
            zone = [0, 0, 1, 0, 0]  # Zone but 2 => 3
    else:
        if y > 1750:
            zone = [0, 0, 0, 1, 0]  # Couloir haut => 4
        else:
            zone = [0, 0, 0, 0, 1]  # Couloir bas => 5
    return zone


def heron(sender, player, pass_, dist_mat):

    positions = pass_.drop(["sender", "time_start", "Id"])

    n = 2
    positions = [(positions[k:k+n]) for k in range(0, len(positions), n)]
    sender_coordinates = positions[sender-1]
    player_coordinates = positions[player-1]

    team = int(sender/22 <= 0.5)
    start = team*11

    row = player-1
    opp_dist = dist_mat[row, start: start+11]

    if(same_team_(sender, player) == 0):
        h = 0
    elif((sender_coordinates[0] == player_coordinates[0]) and
         (sender_coordinates[1] == player_coordinates[1])):
        h = min(opp_dist)
    else:
        distances = []
        for i in range(start, start+11):
            opponent_coordinates = positions[i]
            distances.append(smallest_distance_pt_seg(opponent_coordinates,
                                                      sender_coordinates,
                                                      player_coordinates))

        h = min(distances)

    return h


def make_pair_of_players(X_, y_=None):
    n_ = X_.shape[0]
    pair_feature_col = ["time", "sender", "x_sender", "y_sender",
                        "player_j", "x_j", "y_j", "same_team", "distance",
                        "distance_opp_1", "distance_opp_2", "distance_opp_rec_1",
                        "distance_opp_rec_2", "distance_line",
                        "nb_opp", "zone_1_send", "zone_2_send",
                        "zone_3_send", "zone_4_send", "zone_5_send",
                        "zone_1_rec", "zone_2_rec", "zone_3_rec",
                        "zone_4_rec", "zone_5_rec", "x_ball_gain",
                        "dist_y_abs", "rec_to_cntr_dist", "snd_to_cntr_dist",
                        "smallest_angle", "is_in_attack"]

    X_pairs = pd.DataFrame(data=np.zeros((n_*22, len(pair_feature_col))),
                           columns=pair_feature_col)
    y_pairs = pd.DataFrame(data=np.zeros((n_*22, 1)), columns=["pass"])

    # From pass to pair of players
    idx = 0
    for i in range(n_):
        p_i_ = X_.iloc[i]
        time = p_i_.time_start
        distance_matrix = build_distance_matrix(p_i_)
        x_ball_gain = compute_x_ball_gain(p_i_)
        sender = p_i_.sender
        players = np.arange(1, 23)
        other_players = np.delete(players, sender-1)
        ss_numb_opp = number_of_opp(sender, sender, distance_matrix)
        zone_sender = define_zone(p_i_["x_{:0.0f}".format(sender)],
                                  p_i_["y_{:0.0f}".format(sender)])
        snd_to_cntr_dist = np.sqrt((p_i_["x_{:0.0f}".format(sender)])**2 +
                           (p_i_["y_{:0.0f}".format(sender)])**2)
        attack = is_in_attack(p_i_)

        X_pairs.iloc[idx] = [time, sender,  p_i_["x_{:0.0f}".format(sender)],
                             p_i_["y_{:0.0f}".format(sender)],
                             sender,
                             p_i_["x_{:0.0f}".format(sender)],
                             p_i_["y_{:0.0f}".format(sender)],
                             same_team_(sender, sender),
                             0,
                             distance_to_opp(sender, sender, distance_matrix)[0],
                             distance_to_opp(sender, sender, distance_matrix)[1],
                             distance_to_opp_rec(sender, distance_matrix)[0],
                             distance_to_opp_rec(sender, distance_matrix)[1],
                             distance_to_opp(sender, sender, distance_matrix)[0],
                             ss_numb_opp, zone_sender[0], zone_sender[1],
                             zone_sender[2], zone_sender[3], zone_sender[4],
                             zone_sender[0], zone_sender[1], zone_sender[2],
                             zone_sender[3], zone_sender[4], 0, 0,
                             snd_to_cntr_dist, snd_to_cntr_dist, 0, attack]


        idx += 1
        for player_j in other_players:
            distance = distance_matrix[sender-1, player_j-1]
            distance_opp = distance_to_opp(sender, player_j, distance_matrix)
            distance_opp_rec = distance_to_opp_rec(player_j, distance_matrix)
            distance_line = heron(sender, player_j, p_i_, distance_matrix)
            numb_opp = number_of_opp(sender, player_j, distance_matrix)
            zone_receiver = define_zone(p_i_["x_{:0.0f}".format(player_j)],
                                        p_i_["y_{:0.0f}".format(player_j)])
            dist_y_abs =  abs(p_i_["y_{:0.0f}".format(sender)] -
                              p_i_["y_{:0.0f}".format(player_j)])
            rec_to_cntr_dist = np.sqrt((p_i_["x_{:0.0f}".format(player_j)])**2
                               + (p_i_["y_{:0.0f}".format(player_j)])**2)
            angle = min_angle(sender, player_j, distance_matrix)
            attack = is_in_attack(p_i_)

            X_pairs.iloc[idx] = [time, sender, p_i_["x_{:0.0f}".format(sender)],
                                 p_i_["y_{:0.0f}".format(sender)],
                                 player_j,
                                 p_i_["x_{:0.0f}".format(player_j)],
                                 p_i_["y_{:0.0f}".format(player_j)],
                                 same_team_(sender, player_j), distance,
                                 distance_opp[0], distance_opp[1],
                                 distance_opp_rec[0], distance_opp_rec[1],
                                 distance_line, numb_opp, zone_sender[0],
                                 zone_sender[1], zone_sender[2],
                                 zone_sender[3], zone_sender[4],
                                 zone_receiver[0], zone_receiver[1],
                                 zone_receiver[2], zone_receiver[3],
                                 zone_receiver[4],
                                 x_ball_gain["x_{:0.0f}".format(player_j)],
                                 dist_y_abs, rec_to_cntr_dist,
                                 snd_to_cntr_dist, angle, attack]

            if y_ is not None:
                y_pairs.iloc[idx]["pass"] = int(player_j == y_.iloc[i])
            idx += 1

    return X_pairs, y_pairs


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

    file_name = '{}.csv'.format(file_name)

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
        header = 'Id,Predicted,'
        for j in range(1, 23):
            header = header + 'P_{:0.0f},'.format(j)
        handle.write(header[:-1]+"\n")

        # Adding your estimated score
        first_line = 'Estimation,{},'.format(estimated_score)
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
