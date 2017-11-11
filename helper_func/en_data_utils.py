import numpy as np
import csv
import re
from dir_walker import walker

import matplotlib
# Force matplotlib to not use any Xwindows backend.
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Convert the experiment data to the training set.
# The script for data preprocess.

def get_dataset(dataset_dir, x_range):
    """
    Get data from csv file, do the augmentation, seperate the training set and 
    the test set, then save to npy file for further uses.
    """
    Y = []
    X = []
    # import csv data
    for filename, subdir in walker(dataset_dir, re.compile('training(.*?)_\d+.csv')):
        y = re.findall('training(\d+)_\d+.csv', filename)[-1]
        y_one_hot = np.eye(9, dtype=int)[int(y) - 1]
        Y.append(y_one_hot)
        with open(subdir+'/'+filename, 'r') as f:
            reader = csv.reader(f)
            data_list = list(reader)[1:] # Skip the first line

        x = [float(data_list[i][1]) for i in x_range] # len(x_range)*1 list here
        X.append(x) # m*n list
    coordinates = [data_list[i][0] for i in x_range]

    X_np = np.array(X).T # n*m numpy array
    Y_np = np.array(Y).T # 1*m numpy array
    coordinates_np = np.array(coordinates).T
    #TODO Set seperated folders
    np.save(dataset_dir+'/'+'x_orig', X_np)
    np.save(dataset_dir+'/'+'y_orig', Y_np)
    np.save(dataset_dir+'/'+'coordinates', coordinates_np)

def load_dataset(dataset_dir):
    x_orig = np.load(dataset_dir+'/'+'x_orig.npy')
    y_orig = np.load(dataset_dir+'/'+'y_orig.npy')
    coordinates = np.load(dataset_dir+'/'+'coordinates.npy')
    return x_orig, y_orig, coordinates

def plot_dataset(X, Y, coordinates, save_dir):
    """
    Get n*m dimensional X, plot them to given coordinates
    """
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for i in range(np.shape(X)[1]):
        y = np.argwhere(Y[:, i] == 1)[0][0] + 1
        ax.plot(coordinates, X[:, i], label=str(y))
        ax.set_xlabel('time/ms')
        ax.set_ylabel('Current/A')
        ax.legend(loc='best')
        plt.savefig(save_dir+'/'+'test'+'.png', dpi=300)

if __name__ == '__main__':
    # dataset_dir = '/mnt/t/college/last/finaldesign/ENML/code/test/'
    dataset_dir = 'T:/college/last/finaldesign/ENML/code/test/'
    get_dataset(dataset_dir, range(7000, 8000))
    x_orig, y_orig, coordinates = load_dataset(dataset_dir)
    train_x_set = x_orig - np.mean(x_orig, axis=0)
    train_y_set = y_orig
    plot_dataset(train_x_set, train_y_set, coordinates, dataset_dir)