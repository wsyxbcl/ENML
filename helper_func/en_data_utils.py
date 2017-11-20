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
import peakutils

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
    coordinates = []
    # import csv data
    for filename, subdir in walker(files_dir, re.compile('training(.*?)_\d+.csv')):
        print(subdir+filename)
        y = re.findall('training(\d+)_\d+.csv', filename)[-1]
        # y_one_hot = np.eye(9, dtype=int)[int(y) - 1]
        Y.append(int(y) - 1)
        with open(subdir+'/'+filename, 'r') as f:
            reader = csv.reader(f)
            data_list = list(reader)[1:] # Skip the first line

        x = [float(data_list[i][1]) for i in x_range] # len(x_range)*1 list here
        coordinate = [float(data_list[i][0]) for i in x_range]
        X.append(x) # m*n list
        coordinates.append(coordinate)
    
    Y_one_hot = to_categorical(Y, num_classes=9)
    X_np = np.array(X) # m*n numpy array
    Y_np = np.array(Y_one_hot) # m*c numpy array
    coordinates_np = np.array(coordinates)
    return X_np, Y_np, coordinates_np


def save_dataset(X_np, Y_np, coordinates_np, save_dir):
    """
    Save given dataset to save_dir.(.npy)
    """
    # save_dir = files_dir+'/'+'dataset'
    np.save(get_save_path(save_dir, 'x_orig'), X_np)
    np.save(get_save_path(save_dir, 'y_orig'), Y_np)
    np.save(get_save_path(save_dir, 'coordinates'), coordinates_np)

def load_dataset(dataset_dir, test_ratio=0.3):
    x_orig = np.load(dataset_dir+'/'+'x_orig.npy')
    y_orig = np.load(dataset_dir+'/'+'y_orig.npy')
    coordinates = np.load(dataset_dir+'/'+'coordinates.npy')
    x_train, x_test, y_train, y_test = train_test_split(x_orig, y_orig, test_size=test_ratio)
    return x_train, y_train, x_test, y_test, coordinates

def remove_baseline(x, degree=3):
    """
    Use the baseline function from peakutils to remove base line of x. which 
    iteratively performs a polynomial fitting.
    
    deg (int (default: 3))
    – Degree of the polynomial that will estimate the data baseline
    """
    baseline_values = peakutils.baseline(x, deg=degree)
    x_removed = x - baseline_values
    return baseline_values, x_removed

def plot_dataset(X, Y, coordinates, save_dir, filename):
    """
    Get n*m dimensional X, plot them to given coordinates
    """
    #TODO need rewrite for coordinate has been changed
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for i in range(np.shape(X)[0]):
        y = np.argwhere(Y[i, :] == 1)[0][0] + 1
        ax.plot(coordinates[i, :], X[i, :], label=str(y))
        ax.set_xlabel('time/ms')
        ax.set_ylabel('Current/A')
        ax.legend(loc='best')
    plt.savefig(get_save_path(save_dir, filename), dpi=300)
    plt.clf()

def fft(X, coordinates):
    n = X.shape[1]
    freq = np.copy(coordinates)
    # Convert ms to s
    timestep = (coordinates_np[0,1]-coordinates_np[0,0]) * 0.001
    X_fft = np.fft.fft(X, axis=1)
    freq[:] = np.fft.fftfreq(n, d=timestep)
    return X_fft, freq

if __name__ == '__main__':
    dataset_dir = '/mnt/t/college/last/finaldesign/ENML/code/test/FFTfreq'
    # dataset_dir = 'T:/college/last/finaldesign/ENML/code/test/baseline'
    # dataset_dir = 'T:/college/last/finaldesign/ENML/code/test/20171115_test'

    # TODO combine slice and cropping in one function, seperate cropped data and
    # then add it to the train set?
    # Maybe create a larger slice at first, then 3crop the training set but only
    # 1crop the dev/test set.
    # X_np_1, Y_np_1, coordinates_np_1 = get_dataset(dataset_dir, range(4000, 4500))
    # X_np_2, Y_np_2, coordinates_np_2 = get_dataset(dataset_dir, range(4500, 5000))
    # X_np_3, Y_np_3, coordinates_np_3 = get_dataset(dataset_dir, range(5000, 5500))
    # X_np_4, Y_np_4, coordinates_np_4 = get_dataset(dataset_dir, range(5500, 6000))
    # X_np_5, Y_np_5, coordinates_np_5 = get_dataset(dataset_dir, range(6000, 6500))
    # X_np_6, Y_np_6, coordinates_np_6 = get_dataset(dataset_dir, range(6500, 7000))
    # X_np_7, Y_np_7, coordinates_np_7 = get_dataset(dataset_dir, range(7000, 7500))
    # X_np_8, Y_np_8, coordinates_np_8 = get_dataset(dataset_dir, range(7500, 8000))

    # FFT test module
    X_np, Y_np, coordinates_np = get_dataset(dataset_dir, range(4000, 8000))
    plot_dataset(X_np, Y_np, coordinates_np, dataset_dir+'/plot', 'X_orig')
    X_baseline_removed = np.copy(X_np)
    for i in range(X_np.shape[0]):
        baseline_values, X_baseline_removed[i] = remove_baseline(X_np[i], degree=1)
    plot_dataset(X_baseline_removed, Y_np, coordinates_np, dataset_dir+'/plot', 'X_baseline_removed')
    X_norm = np.copy(X_baseline_removed)
    X_norm = X_baseline_removed - np.mean(X_baseline_removed, axis=1).reshape(np.shape(X_baseline_removed)[0], 1)
    plot_dataset(X_norm, Y_np, coordinates_np, dataset_dir+'/plot', 'X_norm')
    X_fft, freq = fft(X_norm, coordinates_np)
    plot_dataset(np.abs(X_fft[:, :2000]), Y_np, freq[:, :2000], dataset_dir+'/plot', 'X_FFT')

    # save_dataset(np.concatenate((X_np_1, X_np_2, X_np_3, X_np_4, X_np_5, X_np_6, X_np_7, X_np_8)),
    #              np.concatenate((Y_np_1, Y_np_2, Y_np_3, Y_np_4, Y_np_5, Y_np_6, Y_np_7, Y_np_8)),
    #              np.concatenate((coordinates_np_1, coordinates_np_2, coordinates_np_3, coordinates_np_4, coordinates_np_5, coordinates_np_6, coordinates_np_7, coordinates_np_8)),
    #              dataset_dir+'/'+'dataset')

    # train_x_set, train_y_set, test_x_set, test_y_set, coordinates = load_dataset(dataset_dir+'/dataset')
    # print("Shape of train_x_set:")
    # print(train_x_set.shape)
    # print("Shape of test_x_set:")
    # print(test_x_set.shape)
    # print("Shape of train_y_set:")
    # print(train_y_set.shape)
    # print("Shape of test_y_set:")
    # print(test_y_set.shape)
    # plot_dataset(test_x_set, test_y_set, coordinates, dataset_dir+'/plot', 'test_x_set')
    # plot_dataset(train_x_set, train_y_set, coordinates, dataset_dir+'/plot', 'train_x_set')
    
    # Remove base line
    # for i in range(test_x_set.shape[0]):
    #     baseline_values, test_x_set[i] = remove_baseline(test_x_set[i], degree=1)
    # for i in range(train_x_set.shape[0]):
    #     baseline_values, train_x_set[i] = remove_baseline(train_x_set[i], degree=1)
    # train_x_set = train_x_set - np.mean(train_x_set, axis=1).reshape(np.shape(train_x_set)[0], 1)
    # test_x_set = test_x_set - np.mean(test_x_set, axis=1).reshape(np.shape(test_x_set)[0], 1)
    
    # plot_dataset(test_x_set, test_y_set, coordinates, dataset_dir+'/plot', 'baseline_removed_test')
    # plot_dataset(train_x_set, train_y_set, coordinates, dataset_dir+'/plot', 'baseline_removed_train')
    

    # Test baseline function here
    # baseline_values, x_removed = remove_baseline(test_x_set[0], degree=3)
    # x_removed = x_removed - np.mean(x_removed)
    # fig = plt.figure()
    # ax = fig.add_subplot(1,1,1)
    # ax.plot(coordinates, baseline_values, label='baseline')
    # ax.plot(coordinates, test_x_set[0], label='raw data')
    # ax.plot(coordinates, x_removed, label='baseline removed')
    # ax.legend(loc='best')
    # plt.savefig(get_save_path(dataset_dir+'/baseline', 'baseline.png'), dpi=300)