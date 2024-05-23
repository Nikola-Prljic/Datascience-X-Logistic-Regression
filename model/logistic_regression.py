from model.clean_data import prepare_data, get_input_layers
from model.clean_data import normalize_columns
import tensorflow as tf
import numpy as np
import keras
from keras import layers
import pandas as pd

""" def create_model(my_inputs, my_learning_rate):

    model = keras.models.Sequential()
    model.add(layers.Flatten(input_shape=(1, )))
    model.add(layers.Dense(units=32, activation='relu'))
    model.add(layers.Dropout(rate=0.2))
    model.add(layers.Dense(units=4, activation='softmax'))     
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=my_learning_rate),
                    loss="sparse_categorical_crossentropy",
                    metrics=['accuracy'])
    return model """

def create_model(my_inputs, my_learning_rate):
    #model.add(layers.Flatten(input_shape=(1, )))
    #model.add(layers.Dense(units=32, activation='relu'))
    #model.add(layers.Dropout(rate=0.2))
    #model.add(layers.Dense(units=4, activation='softmax'))
    concatenated_inputs = layers.Concatenate()(my_inputs.values())
    x = layers.Dense(64, activation='relu', name='32')(concatenated_inputs)
    x = layers.Dense(32, activation='elu', name='16')(x)
    x = layers.Dropout(rate=0.4, name='dropout')(x)
    #dense = layers.Dense(units=1, name='dense_layer', activation=tf.sigmoid)
    output = layers.Dense(4, activation='softmax')(x)
    """ dense_output = dense(concatenated_inputs)
    output = {
        'dense': dense_output,
    }   """ 
    model = keras.Model(inputs=my_inputs, outputs=output)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=my_learning_rate),
                    loss="sparse_categorical_crossentropy",
                    metrics=['accuracy'])
    return model

def train_model(model: keras.Model, dataset: pd.DataFrame, epochs, label_name, batch_size=None, shuffle=True):
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

    epochs, hist = train_model(model, train_df_norm, epochs, "Hogwarts House", batch_size)
    


    ################ Predict

    features = {name:np.array(value) for name, value in train_df_norm.items()}
    predict = model.predict(features)
    predict = np.argmax(predict[:20], axis=1)
    
    houses_dict = {0: 'Gryffindor', 1: 'Slytherin', 2: 'Ravenclaw', 3: 'Hufflepuff'}
    predict = [houses_dict[house] for house in list(predict)]
    print(predict)

    return model, epochs, hist
