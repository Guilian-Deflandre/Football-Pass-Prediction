# ! /usr/bin/env python
# -*- coding: utf-8 -*-

import os                                                  # idées : distance par rapport à x joueurs adverses; distance du joeur à la ligne de touche/fond; méthode d'ensemble
import time
import datetime
from contextlib import contextmanager

import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import check_random_state

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

def make_pair_of_players(X_, y_=None):
    n_ = X_.shape[0]
    pair_feature_col = ["sender", "x_sender", "y_sender", "player_j", "x_j", "y_j", "same_team"]
    X_pairs = pd.DataFrame(data=np.zeros((n_*22,len(pair_feature_col))), columns=pair_feature_col)
    y_pairs = pd.DataFrame(data=np.zeros((n_*22, 1)), columns=["pass"])

    # From pass to pair of players
    idx = 0
    for i in range(n_):
        sender = X_.iloc[i].sender
        players = np.arange(1, 23)
        #other_players = np.delete(players, sender-1)
        p_i_ = X_.iloc[i]
        for player_j in players:

            X_pairs.iloc[idx] = [sender,  p_i_["x_{:0.0f}".format(sender)], p_i_["y_{:0.0f}".format(sender)],
                                 player_j, p_i_["x_{:0.0f}".format(player_j)], p_i_["y_{:0.0f}".format(player_j)], same_team_(sender, player_j)]

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
    prefix = 'Data/Toy'

    # ------------------------------- Learning ------------------------------- #
    # Load training data
    X_LS = load_from_csv(prefix+'input_training_set.csv')
    y_LS = load_from_csv(prefix+'output_training_set.csv')

    # Transform data as pair of players
    # !! This step is only one way of addressing the problem.
    # We strongly recommend to also consider other approaches that the one provided here.

    X_LS_pairs, y_LS_pairs = make_pair_of_players(X_LS, y_LS)
    X_LS_pairs["distance"] = compute_distance_(X_LS_pairs) # ajout d'une colonne "distance" dans le DataFrame

    X_features = X_LS_pairs[["distance", "same_team"]]

    # Build the model
    model = DecisionTreeClassifier()

    with measure_time('Training'):
        print('Training...')
        model.fit(X_features, y_LS_pairs)

    # ------------------------------ Prediction ------------------------------ #
    # Load test data
    X_TS = load_from_csv(prefix+'input_test_set.csv')
    print(X_TS.shape)

    # Same transformation as LS
    X_TS_pairs, _ = make_pair_of_players(X_TS)
    X_TS_pairs["distance"] = compute_distance_(X_TS_pairs)

    X_TS_features = X_TS_pairs[["distance", "same_team"]]

    # Predict
    y_pred = model.predict_proba(X_TS_features)[:,1]

    # Deriving probas
    probas = y_pred.reshape(X_TS.shape[0], 22)

    # Estimated score of the model
    predicted_score = 0.01 # it is quite logical...

    # Making the submission file
    fname = write_submission(probas=probas, estimated_score=predicted_score, file_name=prefix+"toy_example_probas")
    print('Submission file "{}" successfully written'.format(fname))

    # -------------------------- Random Prediction -------------------------- #

    random_state = 0
    random_state = check_random_state(random_state)
    predictions = random_state.choice(np.arange(1,23), size=X_TS.shape[0], replace=True)

    fname = write_submission(predictions=predictions, estimated_score=predicted_score, file_name=prefix+"toy_example_predictions")
    print('Submission file "{}" successfully written'.format(fname))
