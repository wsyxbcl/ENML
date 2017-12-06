from inception import *
model_name = 'GPU_1060_simple_cnn4_baseline1_fft0_batch512_keepprob2_50_lr_001_on_20171201_class5_len128'
model_dir = '/mnt/t/college/last/finaldesign/ENML/model/20171201_class5_len128/GPU_1060_simple_cnn4_baseline1_fft0_batch512_keepprob2_50_lr_001/training_result'
dataset_dir = '/mnt/t/college/last/finaldesign/ENML/model/20171201_blank_class5_len128/dataset'

model_EN = load_model(model_dir + '/model.h5')
test_ratio = 0.25

train_x_set, train_y_set, coordinates_train, test_x_set, test_y_set, coordinates_test = load_dataset(dataset_dir, test_ratio=test_ratio)
for i in range(test_x_set.shape[0]):
    baseline_values, test_x_set[i] = remove_baseline(test_x_set[i], degree=1)
test_x_set = test_x_set - np.mean(test_x_set, axis=1).reshape(np.shape(test_x_set)[0], 1)
test_x_set = np.multiply(test_x_set, 1e8)
test_x_set = test_x_set.reshape(test_x_set.shape[0], test_x_set.shape[1], 1)
test_analysis(model_EN, test_x_set, test_y_set, dataset_dir+'/'+model_name, 'test_set_result')
