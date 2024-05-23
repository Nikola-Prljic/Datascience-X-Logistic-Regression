from src.utils.load_csv import load
import tensorflow as tf
import pandas as pd
import numpy as np
import keras

from model.normalize import Normalize

def load_data(path_train, path_test):
    train_df = load(path_train)
    test_df = load(path_test)
    return train_df, test_df

def get_input_layers(features):
    # Features used to train the model on.
    inputs = {name: keras.Input(shape=(1,)) for name in features}
    return inputs

def split_df(df: pd.DataFrame, split=0.8):
    df = df.dropna()
    split_index = int(len(df) * split)
    return df[:split_index], df[split_index:]

def normalize(df: pd.DataFrame):
    df_mean = df.mean()
    df_std = df.std()
    df = (df - df_mean) / df_std
    return df

def str_to_int(train_df: pd.DataFrame, test_df: pd.DataFrame):
    house_dict = {'Gryffindor': 0, 'Slytherin': 1, 'Ravenclaw': 2, 'Hufflepuff': 3}
    train_label = np.array(train_df['Hogwarts House'].replace(house_dict), dtype=np.int8)
    test_label = np.array(test_df['Hogwarts House'].replace(house_dict), dtype=np.int8)
    return train_label, test_label

def prepare_data(path_train, path_test, split=0.8):
    df, predict_df = load_data(path_train, path_test)
    train_df, test_df = split_df(df, split)
    train_label, test_label = str_to_int(train_df, test_df)

    train_df = train_df.drop(['Hogwarts House'], axis=1).select_dtypes(include=['number'])
    test_df = test_df.drop(['Hogwarts House'], axis=1).select_dtypes(include=['number'])

    train_df = normalize(train_df)
    test_df = normalize(test_df)

    return train_df, test_df, train_label, test_label
