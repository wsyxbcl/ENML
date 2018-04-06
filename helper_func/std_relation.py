# Used to Compare the standard deviation between current and potential
from en_data_utils import *

dataset_dir = 'C:/code/ENML/test/std_compare_test'
current_dataset_dir = dataset_dir+'/dataset/current'
potential_dataset_dir = dataset_dir+'/dataset/potential'

current_x_orig = np.load(current_dataset_dir+'/x_orig.npy')
current_y_orig = np.load(current_dataset_dir+'/y_orig.npy')
current_coordinates_orig = np.load(current_dataset_dir+'/coordinates.npy')

potential_x_orig = np.load(potential_dataset_dir+'/x_orig.npy')
potential_y_orig = np.load(potential_dataset_dir+'/y_orig.npy')
potential_coordinates_orig = np.load(potential_dataset_dir+'/coordinates.npy')

current_x_std = np.std(current_x_orig, axis=1)
potential_x_std = np.std(potential_x_orig, axis=1)
plot_dataset(current_x_std[:], current_y_orig[:], potential_x_std[:], 
             dataset_dir+'/plot', 'std_compare.png', 
             xlabel='potential/V', 
             ylabel='std of current', 
             trans=1, 
             X_2d=True)