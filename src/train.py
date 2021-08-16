import glob
import os
import tensorflow as tf
import pandas as pd
import numpy as np

path_fischer = '/Data/CSV_FISCHER'
path_morphy = '/Data/CSV_MORPHY'
path_capablanca = '/Data/CSV_CAPABLANCA'

files_fischer = glob.glob(path_fischer + "/*.csv")
files_morphy = glob.glob(path_morphy + "/*.csv")
files_capablanca = glob.glob(path_capablanca + "/*.csv")

li = []

for filename in files_fischer:
    df = pd.read_csv(filename, index_col=None, header=0)
    li.append(df)

for filename in files_morphy:
    df = pd.read_csv(filename, index_col=None, header=0)
    li.append(df)

for filename in files_capablanca:
    df = pd.read_csv(filename, index_col=None, header=0)
    li.append(df)

train = pd.concat(li, axis=0, ignore_index=True)

train.shape