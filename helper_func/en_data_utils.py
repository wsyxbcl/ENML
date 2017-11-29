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
    Y = []
    X = []
    coordinates = []
    # import csv data
    for filename, subdir in walker(files_dir, re.compile('training(.*?)_\d+.csv')):
        # print(subdir+'/'+filename)
        y = re.findall('training(\d+)_.*?.csv', filename)[-1]
        # y_one_hot = np.eye(9, dtype=int)[int(y) - 1]
        Y.append(int(y)) # where y is 0-9
        # Y.append(int(y) - 1) #where y is 1-9
        with open(subdir+'/'+filename, 'r') as f:
            reader = csv.reader(f)
            data_list = list(reader)[1:] # Skip the first line

        x = [float(data_list[i][1]) for i in x_range] # len(x_range)*1 list here
        coordinate = [float(data_list[i][0]) for i in x_range]
        X.append(x) # m*n list
        coordinates.append(coordinate)
        # print(subdir+'/'+filename)
    Y_one_hot = to_categorical(Y, num_classes=10)
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
    # colors = plt.rcParams['axes.prop_cycle']
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
        y = np.argwhere(Y[i, :] == 1)[0][0] + 1
        if y in labels:
            # ax.plot(coordinates[i, :], X[i, :], color=colors.by_key()['color'][y], alpha=trans, label='')
            ax.plot(coordinates[i, :], X[i, :], color=tableau20[y], alpha=trans, label='')
        else:
            # ax.plot(coordinates[i, :], X[i, :], color=colors.by_key()['color'][y], alpha=trans, label=str(y))
            ax.plot(coordinates[i, :], X[i, :], color=tableau20[y], alpha=trans, label=str(y))
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

def get_slice_concat(raw_data_dir, num_slices, len_slice):
    """
    get data according to slice ranges, and concat to form dataset.
    """
    print("Getting slice and concatenating")
    ranges = []
    for i in range(num_slices):
        ranges.append(list(range(-(i + 1) * len_slice, -i * len_slice)))
    for i, r in enumerate(ranges):
        if i == 0:
            X_np, Y_np, coordinates_np = get_dataset(raw_data_dir, r)
        else:
            X, Y, coordinates = get_dataset(raw_data_dir, r)
            X_np = np.concatenate((X, X_np))
            Y_np = np.concatenate((Y, Y_np))
            coordinates_np = np.concatenate((coordinates, coordinates_np))
            print('%f'%(i/num_slices))
    return X_np, Y_np, coordinates_np

if __name__ == '__main__':
    dataset_dir = '/mnt/t/college/last/finaldesign/ENML/model/20171117_class10_len512'
    # dataset_dir = '/mnt/t/college/last/finaldesign/ENML/code/test/test_slice'
    # dataset_dir = 'T:/college/last/finaldesign/ENML/code/test/baseline'
    # dataset_dir = 'T:/college/last/finaldesign/ENML/code/test/20171115_test'
    raw_data_dir = dataset_dir+'/raw'
    num_slices = 8
    len_slice = 512
    X_np, Y_np, coordinates_np = get_slice_concat(raw_data_dir, num_slices, len_slice)

    # # FFT test module
    # X_np, Y_np, coordinates_np = get_dataset(raw_data_dir, range(4000, 8000))
    # plot_dataset(X_np, Y_np, coordinates_np, dataset_dir+'/plot', 'X_orig')
    # X_baseline_removed = np.copy(X_np)
    # for i in range(X_np.shape[0]):
    #     baseline_values, X_baseline_removed[i] = remove_baseline(X_np[i], degree=1)
    # plot_dataset(X_baseline_removed, Y_np, coordinates_np, dataset_dir+'/plot', 'X_baseline_removed')
    # X_norm = np.copy(X_baseline_removed)
    # X_norm = X_baseline_removed - np.mean(X_baseline_removed, axis=1).reshape(np.shape(X_baseline_removed)[0], 1)
    # plot_dataset(X_norm, Y_np, coordinates_np, dataset_dir+'/plot', 'X_norm')

    # X_fft, freq = fft(X_norm, coordinates_np)
    # half = int(X_norm.shape[1]/2)
    # X_fft_plot = np.abs(X_fft[:, :half])
    # plot_dataset(X_fft_plot, Y_np, freq[:, :half], dataset_dir+'/plot', 'X_FFT', xlabel='Freq/Hz', ylabel='A')
    # # Contrast to the first
    # plot_dataset(X_fft_plot - X_fft_plot[0], Y_np, freq[:, :half], dataset_dir+'/plot', 'X_FFT_contrast', xlabel='Freq/Hz', ylabel='A')

    # TODO Really in a hurry. Package these sutff...
    # Average frenquency
    # X_fft_avg = np.copy(X_fft_plot[:7, :])
    # Y_fft_avg = np.copy(Y_np[:7, :])
    # X_fft_avg[0, :] = np.mean(X_fft_plot[:33, :], axis=0)
    # Y_fft_avg[0, :] = Y_np[0, :]
    # X_fft_avg[1, :] = np.mean(X_fft_plot[33:64, :], axis=0)
    # Y_fft_avg[1, :] = Y_np[33, :]
    # X_fft_avg[2, :] = np.mean(X_fft_plot[64:96, :], axis=0)
    # Y_fft_avg[2, :] = Y_np[64, :]
    # X_fft_avg[3, :] = np.mean(X_fft_plot[96:128, :], axis=0)
    # Y_fft_avg[3, :] = Y_np[96, :]
    # X_fft_avg[4, :] = np.mean(X_fft_plot[128:160, :], axis=0)
    # Y_fft_avg[4, :] = Y_np[138, :]
    # X_fft_avg[5, :] = np.mean(X_fft_plot[160:192, :], axis=0)
    # Y_fft_avg[5, :] = Y_np[160, :]
    # X_fft_avg[6, :] = np.mean(X_fft_plot[192:224, :], axis=0)
    # Y_fft_avg[6, :] = Y_np[192, :]

    # plot_dataset(X_fft_avg, Y_fft_avg, freq[:, :half], dataset_dir+'/plot', 'X_FFT_avg', xlabel='Freq/Hz', ylabel='A', trans=1)
    # plot_dataset(X_fft_avg - X_fft_avg[0, :], Y_fft_avg, freq[:, :half], dataset_dir+'/plot', 'X_FFT_avg_contrast', xlabel='Freq/Hz', ylabel='A', trans=1)

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
    plot_dataset(test_x_set[:300], test_y_set[:300], coordinates_test[:300], dataset_dir+'/plot', 'test_orig.png', trans=1)

    for i in range(test_x_set.shape[0]):
        baseline_values, test_x_set[i] = remove_baseline(test_x_set[i], degree=1)
    for i in range(train_x_set.shape[0]):
        baseline_values, train_x_set[i] = remove_baseline(train_x_set[i], degree=1)

    train_x_set = train_x_set - np.mean(train_x_set, axis=1).reshape(np.shape(train_x_set)[0], 1)
    test_x_set = test_x_set - np.mean(test_x_set, axis=1).reshape(np.shape(test_x_set)[0], 1)
    FFT = 1
    if FFT:
        train_x_set, freq = fft(train_x_set, coordinates_train)
        test_x_set, freq = fft(test_x_set, coordinates_test)
        half = int(train_x_set.shape[1]/2)
        train_x_set = np.abs(train_x_set[:, :half])
        test_x_set = np.abs(test_x_set[:, :half])
        freq = freq[:, :half]
        # TODO color list problem here
        # plot_dataset(test_x_set, test_y_set, freq, save_dir+'/plot', 'test_fft', xlabel='Freq/Hz', ylabel='A')
        train_x_set = np.multiply(train_x_set, 1e7)
        test_x_set = np.multiply(test_x_set, 1e7) 
        plot_dataset(test_x_set[:300], test_y_set[:300], freq, dataset_dir+'/plot', 'test_x_set_fft.png', xlabel='Freq/Hz', ylabel='A', trans=1)   
    else:
        train_x_set = np.multiply(train_x_set, 1e8)
        test_x_set = np.multiply(test_x_set, 1e8)
        plot_dataset(test_x_set[:300], test_y_set[:300], coordinates_test[:300], dataset_dir+'/plot', 'test_x_set.png', trans=1)
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