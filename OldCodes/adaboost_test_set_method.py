# ! /usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
import pandas as pd
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

    # ------------------------- Data retrievement ------------------------- #
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

    n = np.arange(100, 800, 50)
    scores = []
    for i in range(n.size):
        print('\nTraining for n_estimators = {}...'.format(n[i]))
        model = AdaBoostClassifier(
            n_estimators=n[i]).fit(X_LS_features, np.ravel(y_LS_pairs))
        y_hat = model.predict_proba(X_VS_features)[:, 1]
        y_hat = output_reconstruction(y_hat)
        scores.append(accuracy_score(y_VS, y_hat))

    # Select the best model based on its performance on the VS
    scores = np.asarray(scores)
    print('Scores: {}'.format(scores))
    best = np.argmax(scores)
    best_model = AdaBoostClassifier(n_estimators=n[best])
    print('\nBest model: n_estimators = {}'.format(n[best]))

    fig = plt.figure()
    plt.plot(n, scores)
    plt.show()
    fig.savefig('ADA_test_set')

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
    final_model = AdaBoostClassifier(
        n_estimators=n[best]).fit(X_LS_VS_TS_features, np.ravel(y_LS_VS_TS_pairs))

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
    X_LS_VS, X_test, y_LS_VS, y_test = train_test_split(X_features, y_LS_pairs,
                                                        test_size=0.2,
                                                        random_state=1)
    X_train, X_val, y_train, y_val = train_test_split(X_LS_VS, y_LS_VS,
                                                      test_size=0.25,
                                                      random_state=1)
    print("| SHAPES |\nX_train : {}\nX_valid : {}\nX_test : {}"
          .format(X_train.shape, X_val.shape, X_test.shape))

    # Build models, train them on LS, and evaluate them on VS
    n_estimators = np.array([50, 100, 200, 300, 500])
    scores = []
    for i in range(n_estimators.size):
        print('\nTraining for n_estimators = {}...'.format(n_estimators[i]))
        model = AdaBoostClassifier(
            n_estimators=n_estimators[i]).fit(X_train, np.ravel(y_train))
        scores.append(model.score(X_val, y_val))

    # Select the best model based on its performance on the VS
    scores = np.asarray(scores)
    print('Scores: {}'.format(scores))
    best = np.argmax(scores)
    best_model = AdaBoostClassifier(n_estimators=n_estimators[best])
    print('\nBest model: n_estimators = {}'.format(n_estimators[best]))

    # Retrain this model on LS+VS
    print('\nTraining on LS+VS...')
    best_model = best_model.fit(X_LS_VS, np.ravel(y_LS_VS))

    # Test this model on the TS
    perf_estim = best_model.score(X_test, y_test)
    print('\nPerformance estimate: {}'.format(perf_estim))

    # Retrain this model on LS+VS+TS
    print('\nTraining on LS+VS+TS...')
    final_model = AdaBoostClassifier(
        n_estimators=n_estimators[best]).fit(X_features, np.ravel(y_LS_pairs))

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
                                               "FINALadaboost_test_set_method")
    print('\nSubmission file "{}" successfully written'.format(fname))"""
