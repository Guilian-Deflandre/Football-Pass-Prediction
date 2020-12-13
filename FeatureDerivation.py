# ! /usr/bin/env python
# -*- coding: utf-8 -*-
import time
import datetime
from contextlib import contextmanager

import pandas as pd
import numpy as np

import Vectors as vec


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


def build_distance_matrix(pass_):
    """
    Compute a matrix containing the distances between each players.

    PARAMETERS
    pass_: array[1, 47]
        The information of a pass (x and y coordinate of each players, id of
        the sender, time in ms since the half start and the if of the pass).

    RETURN
    distance_matrix: array[22, 22]
        The matrix of distances between each players.
    """
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
    """
    Compute the distance between player and the two closest opponent of the
    sender

    PARAMETERS
    sender: int, 1 <= sender <= 22
        The id of the sender of the ball.
    player: int, 1 <= sender <= 22
        The id of the potential receiver of the ball.
    dist_mat: array[22, 22]
        The distance matrix of a pass_.

    RETURN
    distance: pair, (float, float)
        The 2 minimal distance between the receiver and sender's opponent.
    """
    row = sender-1
    team = int(sender/22 <= 0.5)
    start = team*11
    opp_dist = dist_mat[row, start: start+11]
    d1 = np.min(opp_dist)
    opp_dist = np.delete(opp_dist, np.argmin(opp_dist))
    d2 = np.min(opp_dist)
    distance = (d1, d2)
    return distance

"""
Feature used before error detection (see report)
def distance_to_opp(sender, player, dist_mat):
    row = player-1
    team = int(sender/22 <= 0.5)
    start = team*11
    opp_dist = dist_mat[row, start: start+11]
    if(same_team_(sender, player)==1):
        distance = (min(opp_dist), min(np.delete(opp_dist,
                                    np.argmin(opp_dist))))
    else:
        opp_dist = np.delete(opp_dist, np.argmin(opp_dist))
        distance = (min(opp_dist), min(np.delete(opp_dist,
                                    np.argmin(opp_dist))))
    return distance
"""

def sender_distance_to_closest_teammate(sender, dist_mat):
    """
    Compute the distance of the two closest team mates of the sender.

    PARAMETERS
    sender: int, 1 <= sender <= 22
        The id of the sender of the ball.
    dist_mat: array[22, 22]
        The distance matrix of a pass_.

    RETURN
    distance: pair (float, float)
        The minimal distance between the sender and a team mate.
    """
    row = sender-1
    team = int(sender/22 >= 0.5)
    start = team * 11
    team_dist = dist_mat[row, start: start+11]
    d1 = np.min(team_dist)
    if d1 == 0:
        team_dist = np.delete(team_dist, np.argmin(team_dist))
        d1 = np.min(team_dist)
    team_dist = np.delete(team_dist, np.argmin(team_dist))
    d2 = np.min(team_dist)
    distance = (d1, d2)

    return distance


def distance_to_opp_rec(player, dist_mat):
    """
    Compute the distances between player and its two closest opponents.

    PARAMETERS
    player: int, 1 <= sender <= 22
        The id of the potential receiver of the ball.
    dist_mat: array[22, 22]
        The distance matrix of a pass_.

    RETURN
    distance: pair (float, float)
        The minimal distance between the receiver and an opponent.
    """
    row = player-1
    team = int(player/22 <= 0.5)
    start = team*11
    opp_dist = dist_mat[row, start: start+11]
    distance = (min(opp_dist), min(np.delete(opp_dist,
                                             np.argmin(opp_dist))))

    return distance


def receiver_closest(sender, player, distance_matrix):
    """
    Compute the distance btween the potential recever and the two closest
    teammate to the sender.

    PARAMETERS
    sender: int, 1 <= sender <= 22
        The id of the sender of the ball.
    player: int, 1 <= sender <= 22
        The id of the potential receiver of the ball.
    dist_mat: array[22, 22]
        The distance matrix of a pass_.

    RETURN
    distance: pair, (float, float)
        The 2 minimal distances between the receiver and the sender's teammate.
    """
    row = player-1
    team = int(sender/22 >= 0.5)
    start = team*11
    team_dist = distance_matrix[row, start: start+11]
    if min(team_dist) == 0:
        team_dist = np.delete(team_dist, np.argmin(team_dist))
    distance = (min(team_dist), min(np.delete(team_dist,
                                    np.argmin(team_dist))))

    return distance


def number_of_opp(sender, player, dist_mat):
    """
    Compute the number of opponents in a radius of 15 meters around the
    potential receiver.

    PARAMETERS
    sender: int, 1 <= sender <= 22
        The id of the sender of the ball.
    player: int, 1 <= sender <= 22
        The id of the potential receiver of the ball.
    dist_mat: array[22, 22]
        The distance matrix of a pass_.

    RETURN
    distance: int
        The number of opponent around the sender in a perimeter of 15m.
    """
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


def compute_x_ball_gain(pass_):
    """
    Compute the gain in x of the ball for a pass.

    PARAMETERS
    pass_: array[1, 47]
        The information of a pass (x and y coordinate of each players, id of
        the sender, time in ms since the half start and the if of the pass).

    RETURN
    x_gains: float
        The abscissa displacement of the ball.
    """
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
    """
    Based on the x positions of a team's players, determines if the phase is
    offensive or defensive (using the mean x postion of the team).

    PARAMETERS
    pass_: array[1, 47]
        The information of a pass (x and y coordinate of each players, id of
        the sender, time in ms since the half start and the if of the pass).

    RETURN
    _ : int
        1 if the sender is in an attack position, 0 otherwise.
    """
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
    """
    Given the x coordinates of all players on the field, determine the leftmost
    player.

    PARAMETERS
    x_positions: dictionnary, {"x_i": int}, 1 <= i <= 22 the id of player
        The x coordinates of all players on the football field.

    RETURN
    _ : int, 1 <= _ <= 22]
        The id of the leftmost player on the field.
    """
    leftmost_key = x_positions.keys()[np.argmin(x_positions.idxmin())]
    return int(leftmost_key.replace('x_', ''))


def compute_distance_(X_):
    """
    Given the coordinates of sender and a player j on the field, compute their
    Euclidean distance.

    PARAMETERS
    X_: array[2, 2]
        An array containing 2 coordinates pair.

    RETURN
    d : float
        The Euclidean distance between the sender and a player j.
    """
    leftmost_key = x_positions.keys()[np.argmin(x_positions.idxmin())]
    return int(leftmost_key.replace('x_', ''))
    d = np.zeros((X_.shape[0],))

    d = np.sqrt((X_["x_sender"]-X_["x_j"])**2 + (X_["y_sender"]-X_["y_j"])**2)
    return d


def define_zone(x, y):
    """
    Given the coordinates of player, compute the zone on the field in which it
    is. Each of the five zone is represented by a boolean value in an array.

    PARAMETERS
    x: int
        The x coordinate of the player.
    y: int
        The y coordinate of the player.

    RETURN
    zone : array[1, 5]
        An array of boolean representing each zone.
    """
    zone = 0
    if y >= -1750 and y <= 1750:
        if x >= -3200 and x <= 3200:
            zone = [1, 0, 0, 0, 0]
        if x >= -5250 and x < -3200:
            zone = [0, 1, 0, 0, 0]
        else:
            zone = [0, 0, 1, 0, 0]
    else:
        if y > 1750:
            zone = [0, 0, 0, 1, 0]
        else:
            zone = [0, 0, 0, 0, 1]
    return zone


def smallest_distance_pt_seg(opponent, sender, receiver):
    """
    Compute the smallest distance between the pass line segment and a given
    opponent.

    PARAMETERS
    opponent: pair, (int, int)
        The coordinates of the considered opponent of the sender.
    sender: pair, (int, int)
        The coordinates of the sender of the ball.
    receiver: pair, (int, int)
        The coordinates of the potential receiver of the ball.

    RETURN
    smallest_dist: float
        The smallest distance of oppenant to the line pass segment.
    """
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


def heron(sender, player, pass_, dist_mat):
    """
    Compute the smallest distance of an opponent and the pass line.

    PARAMETERS
    sender: int, 1 <= sender <= 22
        The id of the sender of the ball.
    player: int, 1 <= sender <= 22
        The id of the potential receiver of the ball.
    pass_: array[1, 47]
        The information of a pass (x and y coordinate of each players, id of
        the sender, time in ms since the half start and the if of the pass).
    dist_mat: array[22, 22]
        The distance matrix of a pass_.

    RETURN
    h: float
        The minimum distance between an opponent and the line pass.
    """
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
    pair_feature_col = ["sender", "x_sender", "y_sender",
                        "player_j", "x_j", "y_j", "same_team", "distance",
                        "distance_opp_1", "distance_opp_2",
                        "distance_opp_rec_1", "distance_opp_rec_2",
                        "receiver_closest_t1", "receiver_closest_t2",
                        "nb_opp", "x_ball_gain", "zone_1_send", "zone_2_send",
                        "zone_3_send", "zone_4_send", "zone_5_send",
                        "zone_1_rec", "zone_2_rec", "zone_3_rec",
                        "zone_4_rec", "zone_5_rec", "distance_line",
                        "dist_y_abs", "is_in_attack", "sc_dist", "rc_dist"]

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
        zone_sender = define_zone(p_i_["x_{:0.0f}".format(sender)],
                                  p_i_["y_{:0.0f}".format(sender)])
        attack = is_in_attack(p_i_)
        snd_to_cntr_dist = np.sqrt(
            (p_i_["x_{:0.0f}".format(sender)])**2 +
            (p_i_["y_{:0.0f}".format(sender)])**2)

        for player_j in players:
            distance = distance_matrix[sender-1, player_j-1]
            distance_opp = distance_to_opp(sender, player_j, distance_matrix)
            distance_opp_rec = distance_to_opp_rec(player_j, distance_matrix)
            receiver_clos = receiver_closest(sender, player_j, distance_matrix)
            numb_opp = number_of_opp(sender, player_j, distance_matrix)
            zone_receiver = define_zone(p_i_["x_{:0.0f}".format(player_j)],
                                        p_i_["y_{:0.0f}".format(player_j)])
            distance_line = heron(sender, player_j, p_i_, distance_matrix)
            dist_y_abs = abs(p_i_["y_{:0.0f}".format(sender)] -
                             p_i_["y_{:0.0f}".format(player_j)])
            rec_to_cntr_dist = np.sqrt(
                (p_i_["x_{:0.0f}".format(player_j)])**2 +
                (p_i_["y_{:0.0f}".format(player_j)])**2)

            X_pairs.iloc[idx] = [sender,  p_i_["x_{:0.0f}".format(sender)],
                                 p_i_["y_{:0.0f}".format(sender)],
                                 player_j,
                                 p_i_["x_{:0.0f}".format(player_j)],
                                 p_i_["y_{:0.0f}".format(player_j)],
                                 same_team_(sender, player_j), distance,
                                 distance_opp[0], distance_opp[1],
                                 distance_opp_rec[0], distance_opp_rec[1],
                                 receiver_clos[0], receiver_clos[1], numb_opp,
                                 x_ball_gain["x_{:0.0f}".format(player_j)],
                                 zone_sender[0], zone_sender[1],
                                 zone_sender[2], zone_sender[3],
                                 zone_sender[4], zone_receiver[0],
                                 zone_receiver[1], zone_receiver[2],
                                 zone_receiver[3], zone_receiver[4],
                                 distance_line, dist_y_abs, attack,
                                 snd_to_cntr_dist, rec_to_cntr_dist]

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
