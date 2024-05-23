import numpy as np

class Normalize():
  def __init__(self, x):
    # Initialize the mean and standard deviation for normalization
    self.mean = np.mean(x)
    self.std = np.std(x)

  def norm(self, x):
    # Normalize the input
    return (x - self.mean)/self.std

  def unnorm(self, x):
    # Unnormalize the input
    return (x * self.std) + self.mean