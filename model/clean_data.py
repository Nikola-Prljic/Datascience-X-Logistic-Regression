from src.utils.load_csv import load
import tensorflow as tf
import pandas as pd
import keras
from model.normalize import Normalize

def load_data(path_train, path_test):
    train_df = load(path_train)
    test_df = load(path_test)
    return train_df, test_df

def get_input_layers(feature1, feature2):
    # Features used to train the model on.
    inputs = {
        feature1: keras.Input(shape=(1,)),
        feature2: keras.Input(shape=(1,))
    }
    return inputs

def normalize_columns(df: pd.DataFrame, feature1, feature2, split=0.8):
    df = df.dropna()
    features = Normalize(pd.concat([df[feature1], df[feature2]]))
    df[feature1] = features.norm(df[feature1])
    df[feature2] = features.norm(df[feature2])
    split_index = int(len(df) * split)
    return df[:split_index], df[split_index:]

def prepare_output_layer(train_df: pd.DataFrame):
    #replace the house name with a number
    train_df['Hogwarts House'].replace({'Gryffindor': 0, 'Slytherin': 1, 'Ravenclaw': 2, 'Hufflepuff': 3}, inplace=True)
    return train_df

def prepare_data(path_train, path_test):
    train_df, test_df = load_data(path_train, path_test)
    train_df = prepare_output_layer(train_df)
    train_df = train_df.select_dtypes(include=['number'])
    return train_df, test_df
