from model.clean_data import prepare_data, get_input_layers
import tensorflow as tf
import numpy as np
import keras
from keras import layers
import pandas as pd
import datetime

def create_model(my_inputs: dict, my_learning_rate: int) -> keras.Model:
    
    # Combine layers
    concatenated_inputs = layers.Concatenate()(my_inputs.values())
    
    # Create hidden neural network layers
    x = layers.Dense(64, activation='relu', name='32')(concatenated_inputs)
    x = layers.Dense(32, activation='elu', name='16')(x)
    x = layers.Dropout(rate=0.4, name='dropout')(x)

    # Create output layer we need for because we have 4 possibilities
    output = layers.Dense(4, activation='softmax')(x)

    # Create Model and use a more accurate loss
    model = keras.Model(inputs=my_inputs, outputs=output)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=my_learning_rate),
                  loss="sparse_categorical_crossentropy",
                  metrics=['accuracy'])
    
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    return model, tensorboard_callback

def train_model(model: keras.Model, tensorboard_callback, dataset: pd.DataFrame, label, epochs, batch_size=10, shuffle=True):
    
    # Create features dict
    features = {name:np.array(value, dtype=np.float16) for name, value in dataset.items()}
    
    # Fit everything into the model and set
    history = model.fit(x=features, y=label, batch_size=batch_size,
                        epochs=epochs, shuffle=shuffle,
                        callbacks=[tensorboard_callback])
    
    # Save the loss and accuracy
    epochs = history.epoch
    hist = pd.DataFrame(history.history)
    return epochs, hist

def evaluate_model(model: keras.Model, train_df, test_df, test_label):

    # Evaluate create features dict
    test_features = {name:np.array(value) for name, value in test_df.items()}
    print('------------Evaluated Model------------')
    model.evaluate(test_features, test_label, batch_size=10)

    # Predict and small test
    features = {name:np.array(value) for name, value in train_df.items()}
    predict = model.predict(features)
    predict = np.argmax(predict[:20], axis=1)
    houses_dict = {0: 'Gryffindor', 1: 'Slytherin', 2: 'Ravenclaw', 3: 'Hufflepuff'}
    predict = [houses_dict[house] for house in list(predict)]
    print(predict)

def logisticRegression(features_names, train_path, test_path, learning_rate, epochs, batch_size=10, split=0.8):
    
    # Prepare data, split Normalize and create labels
    train_df, test_df, train_label, test_label = prepare_data(train_path, test_path, split)

    # Create a dict with the input labels
    inputs = get_input_layers(features_names)

    model, tensorboard_callback = create_model(inputs, learning_rate)

    epochs, hist = train_model(model, tensorboard_callback,
                               train_df, train_label, epochs, batch_size)

    # Eval :O
    evaluate_model(model, train_df, test_df, test_label)
    return model, epochs, hist
