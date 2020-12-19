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
- `input_train_set.csv` - the input data of the training set
- `output_train_set.csv` - the output data of the training set
- `input_test_set.csv` - the input data of the test set

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


# Computed features
## Quantitative Features
It has been decided to enrich our data set with some numerical features, mainly being distancesover the football field. Let’s review which features have been selected to solve the prediction problem.
* Distance between the sender and the receiver: This feature was already implement by the `toy_example.py` provided file and has been kept. It simply consists of the Euclidean distance between the ball sender and the potential receiver (drawnamong all players on the field including the sender).
* Distances between the sender and his two closest team mates: these features are represented correspond to the distance between the sender of the pass and his two closest teammates.
* Distances between the receiver and his two closest opponents: These features has also been added and are the Euclidean distances between the receiver of the ball during the pass and his two closest opponents around him.
* Distances between the sender and his two closest opponents: These features has also been added and are the Euclidean distances between the sender of the ballduring the pass and his two closest opponents around him.
* Distance between the pass line and its closest sender’s opponent: An important feature for the team members was the smallest distance between a member of the sender’s opposed team and the pass line. Indeed, if some opponent are between the sender and the receiver, it seems logical that the pass is more risky and therefore that the sender will tend to choose another receiver.
* Distance between the sender/receiver and the field center: These features correspond to the Euclidean distances between the sender and the receiver of the ball during the potential pass and the center of the football field located in (0,0).
* Gain on the abscissa of the ball: This feature represents the horizontal movement of the ball towards the goal of the team opposed to the one of the sender. To compute this attribute, the side of the field occupied by each team has to be known. To determine this, the algorithm begins by determining the leftmost player on the field, considering that it is the goalkeeper of one of the 2 teams. If this player is on the same team than the ball sender, the goal in which he will have to score is on the right of the field, otherwise it is on the left. Note that here, an abstraction is made: *a player will never be located behind the goalkeeper of the team on the left side of the field*. It has been assumed that these situations are rare in football and thus that this is a right way to guess teams’ sides.
* Ordinate displacement of the ball: This new data has been added to the model by simply computing the absolute value of the difference of ordinates coordinates between the sender and the potential receiver of a pass.
## Qualitative Features
* Predicate indicating if the sender is on the same team as the receiver of not: This feature was already implement by the `toy_example.py` provided file and has been kept. It simply consists of a boolean value being `1` (i.e. `True`) if the potential receiver of a pass is a team member of the sender, and `0` (i.e. `False`) otherwise.
* Predicate indicating if the sender’s team is performing an attack or not: In a football game, the pass performed can depend of whether the team possessing the ball performs an attack or not. It thus has been decided to implement a function able to determine if thesender is in an offensive or defensive phase of game.  To do this, first the sender’s team field side is determined using the same method as described above. After that, the mean value of the `x` location of each player on the sender’s team is computed. Using these values, we can determine where most of the players in a team are located. If they are mostly in their camp, the action is considered as a defensive one, if they are not it’s an offensive one.
* Number of sender’s opponents around the receiver: This characteristic simply gives the number of opponents in a radius of 15,8 meters from the ball receiver. This distance has not been chosen by chance. Indeed it represents the mean length of a pass for the whole data set.
* Location zone of the sender and the receiver on the field: it has been decided to split the field into several zones. The following zones definition was decided
    -Zone 1 (center): `x &isin; [−3200cm; 3200cm]`, `y &isin; [−1750cm; 1750cm]`;
    -Zone 2 (left goal): `x &isin; [−5250cm;−3200cm]`, `y &isin; [−1750cm; 1750cm]`;
    -Zone 3 (right goal): `x &isin; [3200cm; 5250cm]`, `y &isin; [−1750cm; 1750cm]`;
    -Zone 4 (high flank): `x &isin; [−5250cm; 5250cm]`, `y &isin; [1750cm; 3400cm]`;
    -Zone 5 (low flank): `x &isin; [−5250cm; 5250cm]`, `y &isin; [−3400cm;−1750cm]`.

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
