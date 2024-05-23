from model.logistic_regression import logisticRegression

learning_rate=0.001
epochs = 35
batch_size = 30

model = logisticRegression('Muggle Studies', 'Charms',
                           'src/datasets/dataset_train.csv', 'src/datasets/dataset_test.csv',
                           learning_rate, epochs, batch_size)

