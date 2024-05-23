from model.logistic_regression import logisticRegression
from model.plotLoss import plot_loss

learning_rate=0.004
epochs = 100
batch_size = 80

model, epochs, hist = logisticRegression('Muggle Studies', 'Charms',
                           'src/datasets/dataset_train.csv', 'src/datasets/dataset_test.csv',
                           learning_rate, epochs, batch_size)

plot_loss(epochs, hist)
