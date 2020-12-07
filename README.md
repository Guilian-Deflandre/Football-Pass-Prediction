# Introduction
This competition is organized in the context of the "Introduction to Machine Learning" course (ELEN0062-1) at the University of Liège, Belgium. The goal is to let you apply the methods and principles exposed in the theoretical course in order to address a real problem: **pass prediction** during football matches.

# Task
The mission of your group, should you decide to accept it, is to use (supervised) learning techniques to design a model able to predict the next player who will receive the ball via a pass based on the position of all players and the ball at a given time.

To achieve this, we provide you with a set of `8682` samples (passes) with the player positions, who are the sending and receiving players. A test set of `3000` unlabeled data is provided for which we ask you to provide the next player who will receive the ball via a pass.

A more precise description of the data is available in the Data section.

Note that you can not use any information outside of those provided (see more in the Rules section).

# Data
The provided data set has been (randomly) divided into a training set and a test set.

Each sample (corresponding to a pass) is a snapshot of the situation when the ball is passed, and contains the positions of the 22 players on the field (11 per team), the ID of the player (sender) with the ball, and the time (in ms, since the start of the half). The corresponding output is the ID of the player (receiver) that receives the passed ball.

Each player is represented by an ID from 1 to 22.

The pitch is 105 meters long (x-axis) and 68 meters wide (y-axis), and coordinates are given in cm. The coordinates of the center of the pitch are (0,0).

## Data description
    * `input_train_set.csv` - the input data of the training set
    * `output_train_set.csv` - the output data of the training set
    * `input_test_set.csv` - the input data of the test set

## Data fields
`input_train_set.csv`:

    * `Id` - pass Id;
    * `time_start` - number of milliseconds since the beginning of the concerned half-time period;
    * `sender` - the ID of the player who has the ball - `{1..22}`;
    * `x_<ID>` - the x position of the player `<ID>` - `[-5250 (cm); +5250 (cm)]`;
    * `y_<ID>` - the y position of the player `<ID>` - `[-3400 (cm); +3400 (cm)]`;

`output_train_set.csv`:

    * `receiver` - the `ID` of the player who receive the ball via a pass [data to predict] - `{1..22}`

`input_test_set.csv` : see `input_train_set.csv`.

## Other files provided
`toy_example.py` : a "naive" script that helps you start and mainly consists in:

    * Loading the training set (both input and output files);
    * Deriving features for each pair of (sender, potential receiver) (This step is only one way of addressing the problem. We strongly recommend to also consider other approaches that the one provided here);
    * Making random/naive predictions (see below for more details);
    * Creating a submission file following the guidelines provided in the evaluation section.

In particular, please note that:
    * It creates a new sample set where each sample is a pair of players: the sender and player `j` with `j=1,..,22` (including the sender).
    * It computes two features: the distance between the two players and if they belong to the same team.
    * The `write_submission` function can make a submission with only the predicted player `ID` (`predictions` of size `(n_test_samples, )`) and/or predicted probabilities (`probas` of size `(n_test_samples,22)`). If only the predictions are provided, then the probabilities are derived (i.e., `1.0` for the predicted player in predictions and `0.0` for all others). If only the probabilities are given, then the predictions are derived (i.e., the predicted player is the one with the largest probability). Note that if several players have the same probability, the one with the smaller Id is selected.
    * `predicted_score` is hard-coded. Do not forget to modify this value.
The `toy_example` script generates two submission files: one based on (randomly) predicted player ids, and one based on player probabilities predicted by a decision tree.

# Organization
The competition follows the same guidelines as the previous projects, ***i.e.***, a written report and codes must be submitted before the deadline on the Montefiore Submission Platform. The competition will end a couple of days before the deadline in order to let you add your latest results in the report. You can find more information here.


# Acknowledgment
This challenge is inspired from the Football Pass Prediction Challenge of the 5th Workshop on Machine Learning and Data Mining for Sports Analytics.
