from matplotlib import pyplot as plt

def plot_loss(epochs, hist):
    plt.plot(epochs, hist['accuracy'], label='accuracy')
    plt.plot(epochs, hist['loss'], label='loss')
    plt.legend()
    plt.ylabel('epochs')
    plt.ylabel('accuracy / loss')
    plt.show()