import numpy as np
import csv
import re
import os
from dir_walker import walker
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Convert the experiment data to the training set.
# The script for data preprocess.

def get_save_path(save_dir, filename):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    path = save_dir+'/'+filename
    return path

def get_dataset(files_dir, x_range):
    """
    Get data from csv file, do the augmentation, seperate the training set and 
    the test set, then save to npy file for further uses.
    """
    # TODO Random slice feature
    Y = []
    X = []
    # import csv data
    for filename, subdir in walker(files_dir, re.compile('training(.*?)_\d+.csv')):
        y = re.findall('training(\d+)_\d+.csv', filename)[-1]
        # y_one_hot = np.eye(9, dtype=int)[int(y) - 1]
        Y.append(int(y) - 1)
        with open(subdir+'/'+filename, 'r') as f:
            reader = csv.reader(f)
            data_list = list(reader)[1:] # Skip the first line

        x = [float(data_list[i][1]) for i in x_range] # len(x_range)*1 list here
        X.append(x) # m*n list
    coordinates = [data_list[i][0] for i in x_range]
    Y_one_hot = to_categorical(Y, num_classes=9)
    X_np = np.array(X) # m*n numpy array
    Y_np = np.array(Y_one_hot) # m*c numpy array
    coordinates_np = np.array(coordinates)

    save_dir = files_dir+'/'+'dataset'
    np.save(get_save_path(save_dir, 'x_orig'), X_np)
    np.save(get_save_path(save_dir, 'y_orig'), Y_np)
    np.save(get_save_path(save_dir, 'coordinates'), coordinates_np)

def load_dataset(dataset_dir, test_ratio=0.3):
    x_orig = np.load(dataset_dir+'/'+'x_orig.npy')
    y_orig = np.load(dataset_dir+'/'+'y_orig.npy')
    coordinates = np.load(dataset_dir+'/'+'coordinates.npy')
    x_train, x_test, y_train, y_test = train_test_split(x_orig, y_orig, test_size=test_ratio)
    return x_train, y_train, x_test, y_test, coordinates

def plot_dataset(X, Y, coordinates, save_dir):
    """
    Get n*m dimensional X, plot them to given coordinates
    """
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for i in range(np.shape(X)[0]):
        y = np.argwhere(Y[i, :] == 1)[0][0] + 1
        ax.plot(coordinates, X[i, :], label=str(y))
        ax.set_xlabel('time/ms')
        ax.set_ylabel('Current/A')
        ax.legend(loc='best')
    plt.savefig(get_save_path(save_dir, 'test.png'), dpi=300)

if __name__ == '__main__':
    # dataset_dir = '/mnt/t/college/last/finaldesign/ENML/code/test/'
    dataset_dir = 'T:/college/last/finaldesign/ENML/code/test/20171113_test'
    # dataset_dir = 'T:/college/last/finaldesign/ENML/code/test/20171112_test'
    get_dataset(dataset_dir, range(7000, 8000))
    # get_dataset(dataset_dir, range(7936, 8000))
    x_train, y_train, x_test, y_test, coordinates = load_dataset(dataset_dir+'/dataset')
    train_x_set = x_train - np.mean(x_train, axis=1).reshape(np.shape(x_train)[0], 1)
    train_y_set = y_train
    test_x_set = x_test - np.mean(x_test, axis=1).reshape(np.shape(x_test)[0], 1)
    test_y_set = y_test
    # plot_dataset(train_x_set, train_y_set, coordinates, dataset_dir+'/plot')