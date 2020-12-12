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
    output = output.astype(int)

    return output

if __name__ == '__main__':
    prefix = 'Data/'
    features = ["same_team", "distance", "distance_opp_1", "distance_opp_2",
                "distance_opp_rec_1", "distance_opp_rec_2", "distance_line",
                "nb_opp", "zone_1_send", "zone_2_send", "zone_3_send", 
                "zone_4_send", "zone_5_send", "zone_1_rec", "zone_2_rec", 
                "zone_3_rec", "zone_4_rec", "zone_5_rec", "x_ball_gain"]

    # -------------------------- Data retrievement -------------------------- #
    # Load training data
    X_LS_tot = FeatureDerivation.load_from_csv(prefix+'input_training_set.csv')
    y_LS_tot = FeatureDerivation.load_from_csv(prefix+'output_training_set.csv')

    # --------------------------- Test set method --------------------------- #
    size = round(0.2*(X_LS_tot.shape[0]))
    X_LS_VS, X_TS, y_LS_VS, y_TS = train_test_split(X_LS_tot, y_LS_tot,
                                                    test_size=size,
                                                    random_state=1)
    X_LS, X_VS, y_LS, y_VS = train_test_split(X_LS_VS, y_LS_VS,
                                              test_size=size, random_state=1)

    print('Learning set features derivation...')
    X_LS_pairs, y_LS_pairs = FeatureDerivation.make_pair_of_players(X_LS, y_LS)
    X_LS_features = X_LS_pairs[features]

    # Build models, train them on LS, and evaluate them on VS
    print('Validation set features derivation...')
    X_VS_pairs, y_VS_pairs = FeatureDerivation.make_pair_of_players(X_VS, y_VS)
    X_VS_features = X_VS_pairs[features]

    #nb_trees = np.arange(50, 150, 5)
    depth = np.arange(1, 50)
    scores = []
    for i in range(depth.size):
        print('\nTraining for depth = {}...'.format(depth[i]))
        model = RandomForestClassifier(max_depth=depth[i],
                                       random_state=1).fit(
                                       X_LS_features,
                                       np.ravel(y_LS_pairs))
        y_hat = model.predict_proba(X_VS_features)[:, 1]
        y_hat = output_reconstruction(y_hat)
        scores.append(accuracy_score(y_VS, y_hat))

    # Select the best model based on its performance on the VS
    scores = np.asarray(scores)
    print('Scores: {}'.format(scores))
    best = np.argmax(scores)
    best_model = RandomForestClassifier(max_depth=depth[best], random_state=1)
    print('\nBest model: max_depth = {}'.format(depth[best]))

    fig = plt.figure()
    plt.plot(depth, scores)
    plt.xlabel('max_depth')
    plt.ylabel('Accuracy score')
    plt.show()
    fig.savefig('RF_test_set_distance_opp_rec')
    # Retrain this model on LS+VS
    X_LS_VS_features = pd.concat([X_LS_features, X_VS_features])
    print('\nX_LS_VS is of shape {}'.format(X_LS_VS_features.shape))
    y_LS_VS_pairs = pd.concat([y_LS_pairs, y_VS_pairs])
    print('\nTraining on LS+VS...')
    best_model = best_model.fit(X_LS_VS_features, np.ravel(y_LS_VS_pairs))

    # Test this model on the TS
    print('Test set features derivation...')
    X_TS_pairs, y_TS_pairs = FeatureDerivation.make_pair_of_players(X_TS, y_TS)
    X_TS_features = X_TS_pairs[features]
    y_hat = best_model.predict_proba(X_TS_features)[:,1]
    y_hat = output_reconstruction(y_hat)
    perf_estim = accuracy_score(y_TS, y_hat)
    print('\nPerformance estimate: {}'.format(perf_estim))

    # Retrain this model on LS+VS+TS
    X_LS_VS_TS_features = pd.concat([X_LS_VS_features, X_TS_features])
    print('X_LS_VS_TS is of shape {}'.format(X_LS_VS_TS_features.shape))
    y_LS_VS_TS_pairs = pd.concat([y_LS_VS_pairs, y_TS_pairs])
    print('\nTraining on LS+VS+TS...')
    final_model = RandomForestClassifier(max_depth=depth[best],
                                         random_state=1).fit(
                                         X_LS_VS_TS_features,
                                         np.ravel(y_LS_VS_TS_pairs))

    # ------------------------------ Prediction ----------------------------- #
    print('\nPredicting...')
    # Load test data
    X_TS = FeatureDerivation.load_from_csv(prefix+'input_test_set.csv')
    print(X_TS.shape)

    # Same transformation as LS
    X_TS_pairs, _ = FeatureDerivation.make_pair_of_players(X_TS)

    X_TS_features = X_TS_pairs[features]

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
                                               "random_forest" +
                                               "_test_set_method" +
                                               "_distance_to_opp_rec")

    print('\nSubmission file "{}" successfully written'.format(fname))
