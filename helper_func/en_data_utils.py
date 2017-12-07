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

def get_dataset(files_dir, x_range, num_classes, y_starts_from):
    """
    Get data from csv file, do the augmentation, seperate the training set and 
    the test set, then save to npy file for further uses.
    y here only stands for class, so there must be consequenced class in filename.
    """
    Y = []
    X = []
    coordinates = []
    # import csv data
    for filename, subdir in walker(files_dir, re.compile('training(.*?)_\d+.csv')):
        # print(subdir+'/'+filename)
        y = re.findall('training(\d+)_.*?.csv', filename)[-1] # starts from 0 maybe
        # y_one_hot = np.eye(9, dtype=int)[int(y) - 1]
        if y_starts_from == 0:
            Y.append(int(y)) # where y is 0~(num_classes-1)
        else:
            Y.append(int(y) - 1) #where y is 1~num_classes
        with open(subdir+'/'+filename, 'r') as f:
            reader = csv.reader(f)
            data_list = list(reader)[1:] # Skip the first line

        x = [float(data_list[i][1]) for i in x_range] # len(x_range)*1 list here
        coordinate = [float(data_list[i][0]) for i in x_range]
        X.append(x) # m*n list
        coordinates.append(coordinate)
        # print(subdir+'/'+filename)
    Y_one_hot = to_categorical(Y, num_classes=num_classes)
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
    x_train, x_test, y_train, y_test, coordinates_train, coordinates_test = train_test_split(x_orig, y_orig, coordinates, test_size=test_ratio)
    return x_train, y_train, coordinates_train, x_test, y_test, coordinates_test

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

def plot_dataset(X, Y, coordinates, save_dir, filename, xlabel='time/ms', ylabel='Current/A', trans=0.5):
    """
    Get m*n dimensional X, plot them to given coordinates
    """
    plt.style.use('ggplot')
    # plt.style.use('Solarize_Light2')
    colors = plt.rcParams['axes.prop_cycle']
    # Try Tableau data color here
    tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),    
                 (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),    
                 (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),    
                 (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),    
                 (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]
    for i in range(len(tableau20)):    
        r, g, b = tableau20[i]    
        tableau20[i] = (r / 255., g / 255., b / 255.)


    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    labels = []
    for i in range(np.shape(X)[0]):
        y = np.argwhere(Y[i, :] == 1)[0][0]
        if y in labels:
            ax.plot(coordinates[i, :], X[i, :], color=colors.by_key()['color'][y], alpha=trans, label='')
            # ax.plot(coordinates[i, :], X[i, :], color=tableau20[y], alpha=trans, label='')
        else:
            ax.plot(coordinates[i, :], X[i, :], color=colors.by_key()['color'][y], alpha=trans, label=str(y))
            # ax.plot(coordinates[i, :], X[i, :], color=tableau20[y], alpha=trans, label=str(y))
            labels.append(y)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend(loc='best')
    plt.savefig(get_save_path(save_dir, filename), dpi=300)
    plt.clf()
    plt.close(fig)

def fft(X, coordinates):
    n = X.shape[1]
    freq = np.copy(coordinates)
    # Convert ms to s
    timestep = (coordinates[0,1]-coordinates[0,0]) * 0.001
    X_fft = np.fft.fft(X, axis=1)
    freq[:] = np.fft.fftfreq(n, d=timestep)
    return X_fft, freq

def fft_avg(test_x_set, test_y_set, num_classes):
    """
    Used to do average on x_set after FFT.
    test_x_set here refers to any x dataset
    """
    y_label = np.argwhere(test_y_set==1)[:, 1]
    y_label_sortindice = np.argsort(y_label)
    y_label_sorted = y_label[y_label_sortindice]
    test_x_set_sorted = test_x_set[y_label_sortindice]
    test_y_set_sorted = test_y_set[y_label_sortindice]

    turning_points = []
    for i, label in enumerate(y_label_sorted):
        if i == 0:
            prev = y_label_sorted[0]
        if label == prev:
            continue
        else:
            prev = label
            turning_points.append(i)

    test_x_avg = np.copy(test_x_set[:num_classes, :])
    test_y_avg = np.copy(test_y_set[:num_classes, :])
    for i, point in enumerate(turning_points):
        if i == 0:
            test_x_avg[i, :] = np.mean(test_x_set_sorted[:point, :], axis=0)
            test_y_avg[i, :] = test_y_set_sorted[0, :]
        elif i == (num_classes - 2):
            test_x_avg[i, :] = np.mean(test_x_set_sorted[turning_points[i - 1]:point, :], axis=0)
            test_y_avg[i, :] = test_y_set_sorted[turning_points[i - 1], :]
            test_x_avg[i + 1, :] = np.mean(test_x_set_sorted[point:, :], axis=0)
            test_y_avg[i + 1, :] = test_y_set_sorted[point, :]
        else:
            test_x_avg[i, :] = np.mean(test_x_set_sorted[turning_points[i - 1]:point, :], axis=0)
            test_y_avg[i, :] = test_y_set_sorted[turning_points[i - 1], :]
    return test_x_avg, test_y_avg

def get_slice_concat(raw_data_dir, num_slices, len_slice, num_classes, y_starts_from):
    """
    get data according to slice ranges, and concat to form dataset.
    """
    print("Getting slice and concatenating")
    ranges = []
    for i in range(num_slices):
        ranges.append(list(range(-(i + 1) * len_slice, -i * len_slice)))
    for i, r in enumerate(ranges):
        if i == 0:
            X_np, Y_np, coordinates_np = get_dataset(raw_data_dir, r, num_classes, y_starts_from)
        else:
            X, Y, coordinates = get_dataset(raw_data_dir, r, num_classes, y_starts_from)
            X_np = np.concatenate((X, X_np))
            Y_np = np.concatenate((Y, Y_np))
            coordinates_np = np.concatenate((coordinates, coordinates_np))
            print('%f'%(i/num_slices))
    return X_np, Y_np, coordinates_np

if __name__ == '__main__':
    dataset_dir = '/mnt/t/college/last/finaldesign/ENML/model/20171117_class5_len128'
    # dataset_dir = '/mnt/t/college/last/finaldesign/ENML/code/test/test_slice'
    # dataset_dir = 'T:/college/last/finaldesign/ENML/model/FFTfreq'
    # dataset_dir = 'T:/college/last/finaldesign/ENML/code/test/20171115_test'
    raw_data_dir = dataset_dir+'/raw'
    y_starts_from = 0 # IMPORTANT, 0 or 1 only. A temporary solution for y_starts promlem!!!
    num_slices = 43
    num_classes = 5
    len_slice = 128
    vis = 1
    FFT = 1

    X_np, Y_np, coordinates_np = get_slice_concat(raw_data_dir, num_slices, len_slice, num_classes, y_starts_from)
    save_dataset(X_np, Y_np, coordinates_np, dataset_dir+'/'+'dataset')
    # train_x_set, train_y_set, test_x_set, test_y_set, coordinates = load_dataset(dataset_dir+'/dataset')
    train_x_set, train_y_set, coordinates_train, test_x_set, test_y_set, coordinates_test = load_dataset(dataset_dir+'/dataset')
    print("Shape of train_x_set:")
    print(train_x_set.shape)
    print("Shape of test_x_set:")
    print(test_x_set.shape)
    print("Shape of train_y_set:")
    print(train_y_set.shape)
    print("Shape of test_y_set:")
    print(test_y_set.shape)

    # Visualization
    if vis:
        # num_pick = 500
        num_pick = 10 * num_classes * num_slices
        plot_dataset(test_x_set[:num_pick], test_y_set[:num_pick], coordinates_test[:num_pick], dataset_dir+'/plot', 'test_orig.png', trans=1)

        for i in range(test_x_set.shape[0]):
            baseline_values, test_x_set[i] = remove_baseline(test_x_set[i], degree=1)
        for i in range(train_x_set.shape[0]):
            baseline_values, train_x_set[i] = remove_baseline(train_x_set[i], degree=1)

        train_x_set = train_x_set - np.mean(train_x_set, axis=1).reshape(np.shape(train_x_set)[0], 1)
        test_x_set = test_x_set - np.mean(test_x_set, axis=1).reshape(np.shape(test_x_set)[0], 1)
        if FFT:
            train_x_set, freq = fft(train_x_set, coordinates_train)
            test_x_set, freq = fft(test_x_set, coordinates_test)
            half = int(train_x_set.shape[1]/2)
            train_x_set = np.abs(train_x_set[:, :half])
            test_x_set = np.abs(test_x_set[:, :half])
            freq = freq[:, :half]
            train_x_set = np.multiply(train_x_set, 1e7)
            test_x_set = np.multiply(test_x_set, 1e7) 
            plot_dataset(test_x_set[:num_pick], test_y_set[:num_pick], freq, dataset_dir+'/plot', 'test_fft.png', xlabel='Freq/Hz', ylabel='A', trans=1)   
            test_x_set_fft_avg, test_y_set_fft_avg = fft_avg(test_x_set[:num_pick], test_y_set[:num_pick], num_classes)
            plot_dataset(test_x_set_fft_avg, test_y_set_fft_avg, freq, dataset_dir+'/plot', 'test_fft_avg.png', xlabel='Freq/Hz', ylabel='A', trans=1)
            plot_dataset(test_x_set_fft_avg - test_x_set_fft_avg[0, :], test_y_set_fft_avg, freq, dataset_dir+'/plot', 'test_fft_avg_contrast.png', xlabel='Freq/Hz', ylabel='A', trans=1)

        else:
            train_x_set = np.multiply(train_x_set, 1e8)
            test_x_set = np.multiply(test_x_set, 1e8)
            plot_dataset(test_x_set[:num_pick], test_y_set[:num_pick], coordinates_test[:300], dataset_dir+'/plot', 'test_x_normed.png', trans=1)
    # Remove base line
    # for i in range(test_x_set.shape[0]):
    #     baseline_values, test_x_set[i] = remove_baseline(test_x_set[i], degree=1)
    # for i in range(train_x_set.shape[0]):
    #     baseline_values, train_x_set[i] = remove_baseline(train_x_set[i], degree=1)
    # train_x_set = train_x_set - np.mean(train_x_set, axis=1).reshape(np.shape(train_x_set)[0], 1)
    # test_x_set = test_x_set - np.mean(test_x_set, axis=1).reshape(np.shape(test_x_set)[0], 1)
    
    # plot_dataset(test_x_set, test_y_set, coordinates, dataset_dir+'/plot', 'baseline_removed_test.png')
    # plot_dataset(train_x_set, train_y_set, coordinates, dataset_dir+'/plot', 'baseline_removed_train.png')
    

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