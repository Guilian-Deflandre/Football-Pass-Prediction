# ! /usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


import FeatureDerivation

def output_reconstruction(y):
    """Reconstruct output vector as in original data"""
    size = len(y)
    y = np.asarray(y)
    output = np.zeros((int(size/22),))
    y = np.reshape(y, (int(size/22), 22))
    for i in range(int(size/22)):
        output[i] = np.argmax(y[i])+1

    return output

if __name__ == '__main__':
    prefix = 'Data/'

    # -------------------------- Data retrievement -------------------------- #
    # Load training data
    X_LS_tot = FeatureDerivation.load_from_csv(prefix+'input_training_set.csv')
    y_LS_tot = FeatureDerivation.load_from_csv(prefix+'output_training_set.csv')

    # --------------------------- Test set method --------------------------- #
    size = round(0.2*(X_LS_tot.shape[0]))
    print('size = {}'.format(size))
    X_LS_VS, X_TS, y_LS_VS, y_TS = train_test_split(X_LS_tot, y_LS_tot, test_size=size, random_state=1)
    X_LS, X_VS, y_LS, y_VS = train_test_split(X_LS_VS, y_LS_VS, test_size=size, random_state=1)
    print("| SHAPES |\nX_train : {}\nX_valid : {}\nX_test : {}".format(X_LS.shape[0], X_VS.shape[0], X_TS.shape[0]))

    print('Learning set features derivation...')
    X_LS_pairs, y_LS_pairs = FeatureDerivation.make_pair_of_players(X_LS, y_LS)
    X_LS_features = X_LS_pairs[["distance", "distance_opp_1", "distance_opp_2",
                             "distance_line", "same_team", "nb_opp",
                             "zone_send", "zone_rec", "x_ball_gain"]]

    # Build models, train them on LS, and evaluate them on VS
    print('Validation set features derivation...')
    X_VS_pairs, y_VS_pairs = FeatureDerivation.make_pair_of_players(X_VS, y_VS)
    X_VS_features = X_VS_pairs[["distance", "distance_opp_1", "distance_opp_2",
                             "distance_line", "same_team", "nb_opp",
                             "zone_send", "zone_rec", "x_ball_gain"]]

    """depth = np.arange(1, 50)
    scores = []
    for i in range(depth.size):
        print('\nTraining for max_depth = {}...'.format(depth[i]))
        model = RandomForestClassifier(
            max_depth=depth[i]).fit(X_LS_features, np.ravel(y_LS_pairs))
        y_hat = model.predict_proba(X_VS_features)[:, 1]
        y_hat = output_reconstruction(y_hat)
        scores.append(accuracy_score(y_VS, y_hat))"""

    nb_trees = np.arange(50, 150, 5)
    scores = []
    for i in range(nb_trees.size):
        print('\nTraining for nb_trees = {}...'.format(nb_trees[i]))
        model = RandomForestClassifier(
            nb_estimators=nb_trees[i], max_depth=14).fit(X_LS_features, np.ravel(y_LS_pairs))
        y_hat = model.predict_proba(X_VS_features)[:, 1]
        y_hat = output_reconstruction(y_hat)
        scores.append(accuracy_score(y_VS, y_hat))

    # Select the best model based on its performance on the VS
    scores = np.asarray(scores)
    print('Scores: {}'.format(scores))
    best = np.argmax(scores)
    best_model = RandomForestClassifier(nb_estimators=nb_trees[best], max_depth=14)
    print('\nBest model: nb_trees = {}'.format(nb_trees[best]))

    fig = plt.figure()
    plt.plot(nb_trees, scores)
    plt.xlabel('Number of trees')
    plt.ylabel('Accuracy score')
    plt.title('max_depth = 14')
    plt.show()
    fig.savefig('RF_test_set_nb_trees')
    # Retrain this model on LS+VS
    X_LS_VS_features = pd.concat([X_LS_features, X_VS_features])
    print('X_LS_VS is of shape {}'.format(X_LS_VS_features.shape))
    y_LS_VS_pairs = pd.concat([y_LS_pairs, y_VS_pairs])
    print('\nTraining on LS+VS...')
    best_model = best_model.fit(X_LS_VS_features, np.ravel(y_LS_VS_pairs))

    # Test this model on the TS
    print('Test set features derivation...')
    X_TS_pairs, y_TS_pairs = FeatureDerivation.make_pair_of_players(X_TS, y_TS)
    X_TS_features = X_TS_pairs[["distance", "distance_opp_1", "distance_opp_2",
                             "distance_line", "same_team", "nb_opp",
                             "zone_send", "zone_rec", "x_ball_gain"]]
    y_hat = best_model.predict_proba(X_TS_features)[:,1]
    y_hat = output_reconstruction(y_hat)
    perf_estim = accuracy_score(y_TS, y_hat)
    print('\nPerformance estimate: {}'.format(perf_estim))

    # Retrain this model on LS+VS+TS
    X_LS_VS_TS_features = pd.concat([X_LS_VS_features, X_TS_features])
    print('X_LS_VS_TS is of shape {}'.format(X_LS_VS_TS_features.shape))
    y_LS_VS_TS_pairs = pd.concat([y_LS_VS_pairs, y_TS_pairs])
    print('\nTraining on LS+VS+TS...')
    final_model = RandomForestClassifier(
        nb_estimators=nb_trees[best], max_depth=14).fit(X_LS_VS_TS_features, np.ravel(y_LS_VS_TS_pairs))

    """
    'Pre-process the data to remove what has to be removed?'
    print('Features derivation...')
    X_LS_pairs, y_LS_pairs = FeatureDerivation.make_pair_of_players(X_LS, y_LS)

    X_features = X_LS_pairs[["distance", "distance_opp_1", "distance_opp_2",
                             "distance_line", "same_team", "nb_opp",
                             "zone_send", "zone_rec", "x_ball_gain"]]

    # -------------------------- Test set method ---------------------------- #
    print('Test set method...')
    # Split data into 3 parts (60-20-20) [%]
    X_LS_VS, X_test, y_LS_VS, y_test = train_test_split(X_features,
                                                        y_LS_pairs,
                                                        test_size=0.2,
                                                        random_state=1)
    X_train, X_val, y_train, y_val = train_test_split(X_LS_VS, y_LS_VS,
                                                      test_size=0.25,
                                                      random_state=1)
    print("| SHAPES |\nX_train : {}\nX_valid : {}\nX_test : {}"
          .format(X_train.shape, X_val.shape, X_test.shape))

    # Build models, train them on LS, and evaluate them on VS
    depth = np.array([2, 4, 6, 8, 10])
    scores = []
    for i in range(depth.size):
        print('\nTraining for max_depth = {}...'.format(depth[i]))
        model = RandomForestClassifier(
            max_depth=depth[i]).fit(X_train, np.ravel(y_train))
        scores.append(model.score(X_val, y_val))

    # Select the best model based on its performance on the VS
    scores = np.asarray(scores)
    print('Scores: {}'.format(scores))
    best = np.argmax(scores)
    best_model = RandomForestClassifier(max_depth=depth[best])
    print('\nBest model: max depth = {}'.format(depth[best]))

    # Retrain this model on LS+VS
    print('\nTraining on LS+VS...')
    best_model = best_model.fit(X_LS_VS, np.ravel(y_LS_VS))

    # Test this model on the TS
    perf_estim = best_model.score(X_test, y_test)
    print('\nPerformance estimate: {}'.format(perf_estim))

    # Retrain this model on LS+VS+TS
    print('\nTraining on LS+VS+TS...')
    final_model = RandomForestClassifier(
        max_depth=depth[best]).fit(X_features, np.ravel(y_LS_pairs))

    # ------------------------------ Prediction ----------------------------- #
    print('\nPredicting...')
    # Load test data
    X_TS = FeatureDerivation.load_from_csv(prefix+'input_test_set.csv')
    print(X_TS.shape)

    # Same transformation as LS
    X_TS_pairs, _ = FeatureDerivation.make_pair_of_players(X_TS)

    X_TS_features = X_TS_pairs[["distance", "distance_opp_1", "distance_opp_2",
                                "distance_line", "same_team", "nb_opp",
                                "zone_send", "zone_rec", "x_ball_gain"]]

    # Predict
    y_pred = final_model.predict_proba(X_TS_features)[:, 1]

    # Deriving probas
    probas = y_pred.reshape(X_TS.shape[0], 22)

    # Estimated score of the model
    predicted_score = perf_estim

    # Making the submission file
    fname = FeatureDerivation.write_submission(probas=probas,
                                               estimated_score=predicted_score,
                                               file_name=prefix +
                                               "FINAL_random_forest" +
                                               "_test_set_method")
    print('\nSubmission file "{}" successfully written'.format(fname))"""
