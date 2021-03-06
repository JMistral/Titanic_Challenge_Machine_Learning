#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from pandas import Series,DataFrame

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


train_df = pd.read_csv("../data/train.csv")
test_df    = pd.read_csv("../data/test.csv")



train_df.head()


train_df.info()
print("----------------------------")
test_df.info()


""" start data cleaning for training dataset
    we have 11 raw features and missing value for Age, Cabin, Embarked.
"""
#%%
# drop Cabin, too many missed values
train_df.drop("Cabin",axis=1,inplace=True)
test_df.drop("Cabin",axis=1,inplace=True)
# drop danduncy features
train_df.drop("Ticket",axis=1,inplace=True)
test_df.drop("Ticket",axis=1,inplace=True)

#train_df.drop("PassengerId",axis=1,inplace=True)
#test_df.drop("PassengerId",axis=1,inplace=True)
#%%


train_df['Embarked'].value_counts()

train_df["Embarked"] = train_df["Embarked"].fillna("S")

fig = plt.figure()
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

sns.countplot(x='Survived', hue="Embarked", data=train_df, order=[1,0],ax = ax1)
sns.countplot(x='Embarked', hue="Survived", data=train_df,ax=ax2)

# As we see in the second plot, "Embarked" feature is a deterministic factor for "Survived"
# For example, the passenger embarked through "C" tend to survive rather than those who embarked
# through "S"

"""
Now we're interested in the influence of factor, "Pclass" and "Sex"
"""
fig = plt.figure()
sns.countplot(x='Pclass', hue="Survived", data=train_df)
# seems like the first class passengers have a higher chance to survive

fig = plt.figure()
sns.countplot(x='Sex', hue="Survived", data=train_df)
# females are more likely to survive than male

"""
Dealing with the missed values within feature "Age"
"""
age_non_survived = train_df["Age"][train_df["Survived"] == 0]
age_survived = train_df["Age"][train_df["Survived"] == 1]

#use FacetGrid to plot multiple kdeplots on one plot
fig = sns.FacetGrid(train_df,hue='Survived')
#call FacetGrid.map() to use sns.kdeplot() to show age distribution
fig.map(sns.kdeplot,'Age',shade=True)

#set the x max limit by the oldest passenger
oldest = train_df['Age'].max()
fig.set(xlim=(0,oldest))
fig.add_legend()

# fill the NaN value in column "Age"
# get average, std, and number of NaN values in train_df
average_age_train   = train_df["Age"].mean()
std_age_train       = train_df["Age"].std()
count_nan_age_train = train_df["Age"].isnull().sum()

# get average, std, and number of NaN values in test_df
average_age_test   = test_df["Age"].mean()
std_age_test       = test_df["Age"].std()
count_nan_age_test = test_df["Age"].isnull().sum()

# generate normally distributed random numbers  (mean - std) & (mean + std)
rand_1 = np.random.randint(average_age_train - std_age_train, average_age_train + std_age_train\
                           , size = count_nan_age_train)
rand_2 = np.random.randint(average_age_test - std_age_test, average_age_test + std_age_test \
                           , size = count_nan_age_test)

# fill NaN values in Age column with random values generated
train_df["Age"][np.isnan(train_df["Age"])] = rand_1
test_df["Age"][np.isnan(test_df["Age"])] = rand_2

# convert from float to int
train_df['Age'] = train_df['Age'].astype(int)
test_df['Age']    = test_df['Age'].astype(int)
