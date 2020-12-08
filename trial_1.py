# ! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import datetime
from contextlib import contextmanager

import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import check_random_state

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

def same_team_(sender,player_j):
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
    positions = [(positions[k:k+n]) for k in range(0, len(positions), n)]       #positions are grouped in coordinates (x,y)
    for i in range(int(pos_size/2)):                                                 #rows
        for j in range(int(pos_size/2)):                                             #columns
            distance_matrix[i,j] = np.sqrt((positions[i][0]-positions[j][0])**2 + (positions[i][1]-positions[j][1])**2)

    return distance_matrix

def distance_to_opp(sender, player, dist_mat):
    '''
    param : sender id; player id; distance matrix
    return : distances between player and the two closest opponents of the sender
    '''
    if(same_team_(sender, player)==0):
        distance = (0,0)
    else:
        row = player-1
        team = int(sender/22<=0.5)
        start = team*11
        opp_dist = dist_mat[row, start : start+11]
        distance = (min(opp_dist), min(np.delete(opp_dist, np.argmin(opp_dist))))

    return distance

def heron(sender, player, distance, dist_mat):
    '''
    param : sender id; player id; disatnce between sender and player; distance matrix
    return : smallest distance between the pass line and an opponent of the sender, using Heron's formula
    '''
    if(same_team_(sender, player)==0):
        h = 0
    else:
        row_p = player-1
        row_s = sender-1
        team = int(sender/22<=0.5)
        start = team*11
        opp_dist_p = dist_mat[row_p, start : start+11] #distances between player and opponents
        opp_dist_s = dist_mat[row_s, start : start+11] #distances between sender and opponents

        s = 0.5*(distance + opp_dist_p + opp_dist_s)

        tol = 10e-9
        sd = s-distance
        ss = s-opp_dist_s
        sp = s-opp_dist_p
        (sd)[abs(sd) < tol] = 0.0
        (ss)[abs(ss) < tol] = 0.0
        (sp)[abs(sp) < tol] = 0.0

        if distance == 0 or math.isnan(distance):                               #if player and sender @ the same place
            h = min(opp_dist_s)
            return h

        area = np.sqrt(s*(sd)*(ss)*(sp))
        h =2*area/distance
        h = min(h)

    return h

def make_pair_of_players(X_, y_=None):
    n_ = X_.shape[0]
    pair_feature_col = ["sender", "x_sender", "y_sender", "player_j", "x_j", "y_j", "same_team", "distance",
                        "distance_opp_1", "distance_opp_2", "distance_line"]

    X_pairs = pd.DataFrame(data=np.zeros((n_*22,len(pair_feature_col))), columns=pair_feature_col)
    y_pairs = pd.DataFrame(data=np.zeros((n_*22, 1)), columns=["pass"])

    # From pass to pair of players
    idx = 0
    for i in range(n_):
        print("iteration nb {}".format(i))
        p_i_ = X_.iloc[i]
        distance_matrix = build_distance_matrix(p_i_)                           #build 22x22 distance matrix
        sender = X_.iloc[i].sender
        players = np.arange(1, 23)
        other_players = np.delete(players, sender-1)
        X_pairs.iloc[idx] = [sender,  p_i_["x_{:0.0f}".format(sender)], p_i_["y_{:0.0f}".format(sender)],
                             sender, p_i_["x_{:0.0f}".format(sender)], p_i_["y_{:0.0f}".format(sender)],
                             same_team_(sender, sender), 0, 0, 0, 0]
        idx += 1
        for player_j in other_players:
            distance = distance_matrix[sender-1, player_j-1]
            distance_opp = distance_to_opp(sender, player_j, distance_matrix)
            distance_line = heron(sender, player_j, distance, distance_matrix)
            X_pairs.iloc[idx] = [sender,  p_i_["x_{:0.0f}".format(sender)], p_i_["y_{:0.0f}".format(sender)],
                                 player_j, p_i_["x_{:0.0f}".format(player_j)], p_i_["y_{:0.0f}".format(player_j)],
                                 same_team_(sender, player_j), distance, distance_opp[0], distance_opp[1], distance_line]

            if not y_ is None:
                y_pairs.iloc[idx]["pass"] = int(player_j == y_.iloc[i])
            idx += 1

    return X_pairs, y_pairs

def compute_distance_(X_):
    d = np.zeros((X_.shape[0],))

    d = np.sqrt((X_["x_sender"]-X_["x_j"])**2 + (X_["y_sender"]-X_["y_j"])**2)
    return d

def write_submission(predictions=None, probas=None, estimated_score=0, file_name="submission", date=True, indexes=None):
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
        probas = np.zeros((n_samples,22))
        for i in range(n_samples):
            probas[i, predictions[i]-1] = 1

    if predictions is None:
        print('Deriving predictions from probabilities')
        predictions = np.zeros((n_samples, ))
        for i in range(n_samples):
            mask = probas[i] == np.max(probas[i])
            selected_players = np.arange(1,23)[mask]
            predictions[i] = int(selected_players[0])


    # Writing into the file
    with open(file_name, 'w') as handle:
        # Creating header
        header = '"Id","Predicted",'
        for j in range(1,23):
            header = header + '"P_{:0.0f}",'.format(j)
        handle.write(header[:-1]+"\n")

        # Adding your estimated score
        first_line = '"Estimation",{},'.format(estimated_score)
        for j in range(1,23):
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

if __name__ == '__main__':
    prefix = 'Data/'

    # ------------------------------- Learning ------------------------------- #
    # Load training data
    X_LS = load_from_csv(prefix+'input_training_set.csv')
    y_LS = load_from_csv(prefix+'output_training_set.csv')

    'Pre-process the data to remove what has to be removed?'

    X_LS_pairs, y_LS_pairs = make_pair_of_players(X_LS, y_LS)

    X_features = X_LS_pairs[["distance", "distance_opp_1", "distance_opp_2", "distance_line","same_team"]]

    # Build the model
    model = DecisionTreeClassifier()

    with measure_time('Training'):
        print('Training...')
        model.fit(X_features, y_LS_pairs)

    # --------------------------- Cross validation --------------------------- #


    # ------------------------------ Prediction ------------------------------ #
    # Load test data
    X_TS = load_from_csv(prefix+'input_test_set.csv')
    print(X_TS.shape)

    # Same transformation as LS
    X_TS_pairs, _ = make_pair_of_players(X_TS)

    X_TS_features = X_TS_pairs[["distance", "distance_opp_1", "distance_opp_2", "distance_line", "same_team"]]
    print(X_TS_features)
    # Predict
    y_pred = model.predict_proba(X_TS_features)[:,1]

    # Deriving probas
    probas = y_pred.reshape(X_TS.shape[0], 22)

    # Estimated score of the model
    predicted_score = 0.01 # it is quite logical...

    # Making the submission file
    fname = write_submission(probas=probas, estimated_score=predicted_score, file_name=prefix+"trial_1_ligne_de_passe_probas")
    print('Submission file "{}" successfully written'.format(fname))

    # -------------------------- Random Prediction -------------------------- #

    '''random_state = 0
    random_state = check_random_state(random_state)
    predictions = random_state.choice(np.arange(1,23), size=X_TS.shape[0], replace=True)
    fname = write_submission(predictions=predictions, estimated_score=predicted_score, file_name="trial_1_predictions")
    print('Submission file "{}" successfully written'.format(fname))'''
