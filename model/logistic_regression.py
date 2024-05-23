from model.clean_data import prepare_data, get_input_layers
from model.clean_data import normalize_columns
import tensorflow as tf
import numpy as np
import keras
from keras import layers
import pandas as pd

def create_model(my_inputs, my_learning_rate):

    classification_threshold = 0
    METRICS = [
           keras.metrics.BinaryAccuracy(name='accuracy',
                                           threshold=classification_threshold),
          ]

    concatenated_inputs = layers.Concatenate()(my_inputs.values())
    dense = layers.Dense(units=1, name='dense_layer', activation=tf.sigmoid)
    dense_output = dense(concatenated_inputs)
    my_outputs = {
        'dense': dense_output,
    }
    model = keras.Model(inputs=my_inputs, outputs=my_outputs)
    model.compile(optimizer=keras.optimizers.experimental.RMSprop(learning_rate=my_learning_rate),
                loss=keras.losses.BinaryCrossentropy(),
                metrics=METRICS)
    return model

def train_model(model: keras.Model, dataset, epochs, label_name, batch_size=None, shuffle=True):

    features = {name:np.array(value) for name, value in dataset.items()}
    label = np.array(features.pop(label_name))
    history = model.fit(x=features, y=label, batch_size=batch_size,
                      epochs=epochs, shuffle=shuffle)
    epochs = history.epoch
    hist = pd.DataFrame(history.history)
    return epochs, hist

def logisticRegression(feature1, feature2, train_path, test_path, learning_rate, epochs, batch_size=10):
    train_df, test_df = prepare_data(train_path, test_path)
    train_df_norm = normalize_columns(train_df, feature1, feature2)
    inputs = get_input_layers(feature1, feature2)

    model = create_model(inputs, learning_rate)

    train_df_norm = train_df_norm.dropna()
    print(train_df_norm)
    train_model(model, train_df_norm, epochs, "Hogwarts House", batch_size)
    return model
