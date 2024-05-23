from model.logistic_regression import logisticRegression
from model.plotLoss import plot_loss

learning_rate=0.0005
epochs = 35
batch_size = 40
train_test_split = 0.8

features_names = ['Muggle Studies', 'Charms', 'History of Magic', 'Transfiguration']

model, epochs, hist = logisticRegression(features_names,
                           'src/datasets/dataset_train.csv', 'src/datasets/dataset_test.csv',
                           learning_rate, epochs, batch_size, train_test_split)

plot_loss(epochs, hist)
