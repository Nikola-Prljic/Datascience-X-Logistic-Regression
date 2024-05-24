from model.logistic_regression import logisticRegression
from model.plotLoss import plot_loss

learning_rate=0.0003
epochs = 35
batch_size = 40
train_test_split = 0.8

# [ 'Arithmancy', 'Astronomy', 'Herbology', 
#   'Defense Against the Dark Arts', 'Divination', 
#   'Muggle Studies', 'Ancient Runes', 'History of Magic', 
#   'Transfiguration', 'Potions', 'Care of Magical Creatures', 
#   'Charms', 'Flying']

# You can add here Features with that the Model can train.
# Just look pick 2 or more.

features_names = ['Muggle Studies', 'Charms', 'Defense Against the Dark Arts', 'Transfiguration', 'Arithmancy']

model, epochs, hist = logisticRegression(features_names,
                           'src/datasets/dataset_train.csv', 'src/datasets/dataset_test.csv',
                           learning_rate, epochs, batch_size, train_test_split)

plot_loss(epochs, hist)
