# ! /usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier

import FeatureDerivation

if __name__ == '__main__':
    prefix = 'Data/'

    # ------------------------- Features derivation ------------------------- #
    # Load training data
    X_LS = FeatureDerivation.load_from_csv(prefix+'input_training_set.csv')
    y_LS = FeatureDerivation.load_from_csv(prefix+'output_training_set.csv')

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
    print('\nSubmission file "{}" successfully written'.format(fname))
