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
# drop Cabin, too many missed values