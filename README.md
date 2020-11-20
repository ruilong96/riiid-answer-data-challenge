# Riiid AIEd Challenge 2020
#### Track knowledge states of 1M+ students in the wild - Use LSBM model and feature engineering and achieve 75.8% accuracy in prediction

## Problem Overview

Riiid Labs, an AI solutions provider delivering creative disruption to the education market, empowers global education players to rethink traditional ways of learning leveraging AI. With a strong belief in equal opportunity in education, Riiid launched an AI tutor based on deep-learning algorithms in 2017 that attracted more than one million South Korean students. This year, the company released EdNet, the world’s largest open database for AI education containing more than 100 million student interactions.

The challenge is to create algorithms for "Knowledge Tracing," the modeling of student knowledge over time. The **goal** is to accurately predict how students will perform on future interactions. You will pair your machine learning skills using Riiid’s EdNet data.

## Exploratory Data Analysis

For this analysis, Kaggle provides us with a massive dataset containing five anonymized  files, in which of three contains the details about the students' activities online, the details about the questions, and the details about the lectures. The dataset is a time-series data, and the students' data is recorded chronologically relative to their entry to the system. 

To understand the dataset better, I conducted an exploratory analysis which yielded the following observation:

* `answer_correctly` is not balanced in a way that the answer correctly cases is more than the answer incorrectly cases.
* From `user_id` distribution plot, most of the users does not have a lot of activities.
* We observe that the same long left tail distribution for three categorical variables(`user_id`, `task_id`, `task_container_id`), indicating a lot specific instances or users does not have a lot of occurrences.

* As we can see that some distribution plot has a long left tail which is an indicator of outliers because it suggests that some specific users may have very different patterns than the rest of the user.

- `timestamp` is correlated with `task_container_id` and `prior_question_had_explanation`
- `content_id` is correlated with `content_type_id` showing that they are somehow related.
- `prior_question_had_explanation` has positive correlation `answered_correctly`, implying that the explanation may help the pupils learn better. Together with the correlation with `timestamp`, it may indicate that the longer usage of the platform results in better performance.

More details are presented in this [notebook](https://github.com/ruilong96/riiid-answer-data-challenge/blob/master/eda-riiid-answer.ipynb)

## Data Prepossessing 

Since the training dataset is massive and the computing resource is limited on the Kaggle platform, we need to cross sampling data without losing generality, and the data preparing and cleaning consisted of the following steps:

1. The given timestamp is the time that has elapsed since the user's first event, not the actual time, so I set a random first access time for each user within a certain interval.
2. Splits the data into training and validation data based on the timestamp. To simulate the learning process, the later part of the data is used for validation data.

3. Isolate the input data and compute the summary transformation (e.g. sum, count) and then Iteratively select the important ones.
4. To mimic the learning process and handling new users, progressively update the user statistics, such as the correct answer rate, number of question answered. 
5. For NA values, we filled in the average values accordingly based on our summary transformation before, or filled in False for Boolean values. 
6. Lastly, convert all variables into numerical variables for LSBM model.

## Modeling

### LSBM

The first method used for knowledge tracing model was the LSBM model due to its interpretability and the computational speed. The hyperparameters are tuned within the model, and cross-validated on the prepared validation data to avoid overfitting. 

The final training's auc is 0.754181, and valid_1's auc is 0.758319.

![feature_im](https://github.com/ruilong96/riiid-answer-data-challenge/blob/master/f_imp_lsbm.png)

To see more detail, please check out [here(ipynb)](https://github.com/ruilong96/riiid-answer-data-challenge/blob/master/lgbm-v2-feature-engineering-riiid-answer.ipynb)

### DNN (in progress)

Since the dataset is a time-series dataset tracing students' knowledge, the deep neural network with its feed forward feature can better approximate students' performance and make prediction. I am planning to use PyTorch to implement this neural net with convolution and pooling layers.

