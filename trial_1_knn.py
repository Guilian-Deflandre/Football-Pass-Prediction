# ! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from sklearn import svm

import FeatureDerivation

if __name__ == '__main__':
    prefix = 'Data/'

    # ------------------------------- Learning ------------------------------- #
    # Load training data
    X_LS = FeatureDerivation.load_from_csv(prefix+'input_training_set.csv')
    y_LS = FeatureDerivation.load_from_csv(prefix+'output_training_set.csv')

    'Pre-process the data to remove what has to be removed?'

    X_LS_pairs, y_LS_pairs = FeatureDerivation.make_pair_of_players(X_LS, y_LS)

    X_features = X_LS_pairs[["distance", "distance_opp_1", "distance_opp_2", "distance_line","same_team"]]

    # Build the model
    k = 100
    model = svm.SVC(probability=True, degree=5)

    #scores = cross_val_score(model, X_features, np.ravel(y_LS_pairs), cv = 5)
    #print('k nearest neighbors for k = {} ----- Mean score: {}'.format(k, scores.mean()))

    with FeatureDerivation.measure_time('Training'):
        print('Training...')
        model.fit(X_features, np.ravel(y_LS_pairs))

    # --------------------------- Cross validation --------------------------- #


    # ------------------------------ Prediction ------------------------------ #
    # Load test data
    X_TS = FeatureDerivation.load_from_csv(prefix+'input_test_set.csv')
    print(X_TS.shape)

    # Same transformation as LS
    X_TS_pairs, _ = FeatureDerivation.make_pair_of_players(X_TS)

    X_TS_features = X_TS_pairs[["distance", "distance_opp_1", "distance_opp_2", "distance_line", "same_team"]]
    print(X_TS_features)
    # Predict
    y_pred = model.predict_proba(X_TS_features)[:,1]

    # Deriving probas
    probas = y_pred.reshape(X_TS.shape[0], 22)

    # Estimated score of the model
    predicted_score = 0.01 # it is quite logical...

    # Making the submission file
    fname = FeatureDerivation.write_submission(probas=probas, estimated_score=predicted_score, file_name="svm_degree_5_all_features")
    print('Submission file "{}" successfully written'.format(fname))

    # -------------------------- Random Prediction -------------------------- #

    '''random_state = 0
    random_state = check_random_state(random_state)
    predictions = random_state.choice(np.arange(1,23), size=X_TS.shape[0], replace=True)

    fname = write_submission(predictions=predictions, estimated_score=predicted_score, file_name="trial_1_predictions")
    print('Submission file "{}" successfully written'.format(fname))'''
