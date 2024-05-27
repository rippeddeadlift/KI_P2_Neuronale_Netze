import matplotlib.pyplot as plt
import numpy as np
import pickle
import os


# Get the training and testing dataset
with open(os.path.join("dataset", "train.p"), mode='rb') as training_data:
    train = pickle.load(training_data)
with open(os.path.join("dataset", "valid.p"), mode='rb') as validation_data:
    valid = pickle.load(validation_data)

# Get the features and labels of the datasets
# The features are the images of the signs
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']

print("Number of training examples: ", X_train.shape[0])
print("Number of validation examples: ", X_valid.shape[0])
print("Image data shape =", X_train[0].shape)
print("Number of classes =", len(np.unique(y_train)))

# Plotting histograms of the count of each sign
def histogram_plot(dataset: np.ndarray, label: str):
    """ Plots a histogram of the dataset

    Args:
        dataset: The input data to be plotted as a histogram.
        label: The label of the histogram.
    """
    hist, bins = np.histogram(dataset, bins=43)
    width = 0.8 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, hist, align='center', width=width)
    plt.xlabel(label)
    plt.ylabel("Image count")
    plt.show()

histogram_plot(y_train, "Training examples")

histogram_plot(y_valid, "Validation examples")