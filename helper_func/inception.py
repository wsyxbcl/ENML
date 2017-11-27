import numpy as np
from keras.layers.convolutional import MaxPooling1D, Conv1D, AveragePooling1D
from keras.layers import Input, Dropout, Dense, Flatten, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import concatenate
from keras import regularizers
from keras import initializers
from keras.models import Model
from keras.utils import plot_model
from en_data_utils import *

import keras.backend as K

def conv1d_bn(x, nb_filter, len_filter, padding='same', strides=1):
    """
    Utility function to apply conv + BN. 
    (Slightly modified from https://github.com/kentsommer/keras-inceptionV4/inception_v4.py)

    """
    channel_axis = -1
    x = Conv1D(nb_filter, len_filter, 
               strides=strides, 
               padding=padding,
               kernel_regularizer=regularizers.l2(0.00004),
               kernel_initializer=initializers.VarianceScaling(scale=2.0, mode='fan_in', distribution='normal', seed=None))(x)
    # x = BatchNormalization(axis=channel_axis, momentum=0.9997, scale=False)(x)
    x = Activation('relu')(x)
    return x

def block_inception_a(input):
    channel_axis= -1

    branch_0 = conv1d_bn(input, 96, 1)

    branch_1 = conv1d_bn(input, 64, 1)
    branch_1 = conv1d_bn(branch_1, 96, 3)

    branch_2 = conv1d_bn(input, 64, 1)
    branch_2 = conv1d_bn(branch_2, 96, 3)
    branch_2 = conv1d_bn(branch_2, 96, 3)

    # branch_3 = AveragePooling1D(3, strides=1, padding='same')(input)
    branch_3 = MaxPooling1D(3, strides=1, padding='same')(input)
    branch_3 = conv1d_bn(branch_3, 96, 1)

    x = concatenate([branch_0, branch_1, branch_2, branch_3], axis=channel_axis)
    return x

def block_inception_b(input):
    channel_axis = -1

    branch_0 = conv1d_bn(input, 192, 1)
    branch_0 = conv1d_bn(branch_0, 192, 3, strides=2, padding='valid')

    branch_1 = conv1d_bn(input, 256, 1)
    branch_1 = conv1d_bn(branch_1, 320, 7)
    branch_1 = conv1d_bn(branch_1, 320, 3, strides=2, padding='valid')

    branch_2 = MaxPooling1D(3, strides=2, padding='valid')(input)
    
    x = concatenate([branch_0, branch_1, branch_2], axis=channel_axis)
    return x

def block_reduction_a(input):
    channel_axis = -1

    branch_0 = conv1d_bn(input, 384, 3, strides=2, padding='valid')

    branch_1 = conv1d_bn(input, 192, 1)
    branch_1 = conv1d_bn(branch_1, 224, 3)
    branch_1 = conv1d_bn(branch_1, 256, 3, strides=2, padding='valid')

    branch_2 = MaxPooling1D(3, strides=2, padding='valid')(input)

    x = concatenate([branch_0, branch_1, branch_2], axis=channel_axis)
    return x

def block_reduction_b(input):
    channel_axis = -1

    branch_0 = conv1d_bn(input, 192, 1)
    branch_0 = conv1d_bn(branch_0, 192, 3, strides=2, padding='valid')

    branch_1 = conv1d_bn(input, 256, 1)
    branch_1 = conv1d_bn(branch_1, 320, 7)
    branch_1 = conv1d_bn(branch_1, 320, 3, strides=2, padding='valid')

    branch_2 = MaxPooling1D(3, strides=2, padding='valid')(input)

    x = concatenate([branch_0, branch_1, branch_2], axis=channel_axis)
    return x

def inception(input):
    channel_axis = -1

    # Input Shape is n * 1
    net = conv1d_bn(input, 32, 3, strides=2, padding='valid')
    net = conv1d_bn(net, 32, 3, padding='valid')
    net = conv1d_bn(net, 64, 3)

    branch_0 = MaxPooling1D(3, strides=2, padding='valid')(net)

    branch_1 = conv1d_bn(net, 96, 3, strides=2, padding='valid')

    net = concatenate([branch_0, branch_1], axis=channel_axis)

    # branch_0 = conv1d_bn(net, 64, 1)
    # branch_0 = conv1d_bn(branch_0, 96, 3, padding='valid')

    # branch_1 = conv1d_bn(net, 64, 1)
    # branch_1 = conv1d_bn(branch_1, 64, 7)
    # branch_1 = conv1d_bn(branch_1, 96, 3, padding='valid')

    # net = concatenate([branch_0, branch_1], axis=channel_axis)

    for idx in range(1):
        net = block_inception_a(net)

    net = block_reduction_a(net)

    # for idx in range(1):
    #     net = block_inception_b(net)

    # net = block_reduction_b(net)
    return net

def inception_test(input_shape, num_classes):
    inputs = Input(input_shape)
    x = inception(inputs)

    # # Final pooling and prediction
    # x = AveragePooling1D(8, padding='valid')(x)
    x = MaxPooling1D(8, padding='valid')(x)
    x = Flatten()(x)
    x = Dense(units=num_classes, activation='softmax')(x)

    model = Model(inputs, x, name='inception_test')
    return model

def simple_inception(input_shape, num_classes):
    inputs = Input(input_shape)

    net = conv1d_bn(inputs, 32, 3, strides=1, padding='same')

    branch_0 = conv1d_bn(net, 64, 1, strides=1, padding='same')

    branch_1 = conv1d_bn(net, 64, 1, strides=1, padding='same')
    branch_1 = conv1d_bn(branch_1, 64, 3, strides=1, padding='same')

    branch_2 = conv1d_bn(net, 64, 1, strides=1, padding='same')
    branch_2 = conv1d_bn(branch_2, 64, 5, strides=1, padding='same')

    branch_3 = MaxPooling1D(3, strides=1, padding='same')(net)
    branch_3 = conv1d_bn(branch_3, 64, 5, strides=1, padding='same')

    net = concatenate([branch_0, branch_1, branch_2, branch_3], axis=-1)

    branch_0 = conv1d_bn(net, 128, 1, strides=1, padding='same')

    branch_1 = conv1d_bn(net, 128, 1, strides=1, padding='same')
    branch_1 = conv1d_bn(branch_1, 128, 3, strides=1, padding='same')

    branch_2 = conv1d_bn(net, 128, 1, strides=1, padding='same')
    branch_2 = conv1d_bn(branch_2, 128, 5, strides=1, padding='same')

    branch_3 = MaxPooling1D(3, strides=1, padding='same')(net)
    branch_3 = conv1d_bn(branch_3, 128, 1, strides=1, padding='same')
    net = concatenate([branch_0, branch_1, branch_2, branch_3], axis=-1)

    net = Flatten()(net)
    net = Dense(units=num_classes, activation='softmax')(net)
    
    model = Model(inputs, net, name='simple_CNN')
    return model    

def simple_CNN(input_shape, num_classes):
    inputs = Input(input_shape)
    net = conv1d_bn(inputs, 64, 3, strides=1, padding='same')
    net = MaxPooling1D(2, strides=2, padding='valid')(net)
    net = conv1d_bn(net, 128, 3, padding='same')
    net = MaxPooling1D(2, strides=2, padding='valid')(net)
    net = conv1d_bn(net, 256, 3, padding='same')
    net = MaxPooling1D(2, strides=2, padding='valid')(net)
    # net = conv1d_bn(net, 512, 3, padding='same')
    # net = MaxPooling1D(2, strides=2, padding='valid')(net)
    net = Flatten()(net)
    net = Dense(units=num_classes, activation='softmax')(net)
    
    model = Model(inputs, net, name='simple_CNN')
    return model

def test_analysis(model, test_x_set, test_y_set, save_dir, filename):
    """
    Use trained model to predict y_estimate on test_x_set.
    And plot y_estimate against test_y_set for further analysis.
    """
    m_test = test_x_set.shape[0]
    classes = test_y_set.shape[1]
    prob = model.predict(test_x_set)
    y_estimate = prob.argmax(axis=-1)
    y_truth = test_y_set.argmax(axis=-1)
    y_correct = y_truth[np.argwhere(y_estimate==y_truth)][:, 0]
    acc_total = y_correct.shape[0]/y_truth.shape[0]
    with open(get_save_path(save_dir, filename+'.txt'), 'wt') as f:
        print("Accuracy(total):%f"%acc_total, file=f)
        print('Class\t#Samples\tTest Accuracy', file=f)
        for i in range(classes):
            num_sample = y_truth[np.argwhere(y_truth==i)].shape[0]
            acc = y_correct[np.argwhere(y_correct==i)].shape[0] / num_sample
            print('%d\t%d\t%f'%(i+1, num_sample, acc), file=f)
    # prepare the np array for visualization
    test_array = np.array([y_truth, y_estimate]).T
    test_array = test_array[np.argsort(test_array[:, 0])]
    idx = np.arange(m_test)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(idx, test_array[:, 0], label='Y', color=plt.rcParams['axes.prop_cycle'].by_key()['color'][0])
    ax.scatter(idx, test_array[:, 1],
               label='Y_estimate', 
               marker='o',
               c=plt.rcParams['axes.prop_cycle'].by_key()['color'][1], 
               alpha=0.7)
    ax.set_xlabel('Samples in test set')
    ax.set_ylabel('Classes')
    ax.legend(loc='best')
    plt.savefig(get_save_path(save_dir, filename+'.png'), dpi=300)
    plt.clf()
    plt.close(fig)

if __name__ == '__main__':
    model_name = 'simple_cnn_baseline1_fft1_batch512'
    root_dir = '/mnt/t/college/last/finaldesign/ENML/model/20171117_class5_len256'
    test_ratio = 0.25
    validation_ratio = 0.25 # splited from traning set
    training_epoch = 128
    batch_size = 512
    save_dir = root_dir+'/'+model_name
    FFT = 1

    train_x_set, train_y_set, test_x_set, test_y_set, coordinates = load_dataset(root_dir+'/'+'dataset', test_ratio=test_ratio)
    # Baseline removal
    # TODO Maybe vectorilize this.
    for i in range(test_x_set.shape[0]):
        baseline_values, test_x_set[i] = remove_baseline(test_x_set[i], degree=1)
    for i in range(train_x_set.shape[0]):
        baseline_values, train_x_set[i] = remove_baseline(train_x_set[i], degree=1)

    train_x_set = train_x_set - np.mean(train_x_set, axis=1).reshape(np.shape(train_x_set)[0], 1)
    test_x_set = test_x_set - np.mean(test_x_set, axis=1).reshape(np.shape(test_x_set)[0], 1)
    if FFT:
        train_x_set, freq = fft(train_x_set, coordinates)
        test_x_set, freq = fft(test_x_set, coordinates)
        half = int(train_x_set.shape[1]/2)
        train_x_set = np.abs(train_x_set[:, :half])
        test_x_set = np.abs(test_x_set[:, :half])
        freq = freq[:, :half]
        # TODO color list problem here
        # plot_dataset(test_x_set, test_y_set, freq, save_dir+'/plot', 'test_fft', xlabel='Freq/Hz', ylabel='A')
        train_x_set = np.multiply(train_x_set, 1e7)
        test_x_set = np.multiply(test_x_set, 1e7)    
    else:
        train_x_set = np.multiply(train_x_set, 1e8)
        test_x_set = np.multiply(test_x_set, 1e8)

    # Important here, for input shape of Conv1D is (batch_size, steps, input_dim)
    # which seems to be (m, n, 1) in this case.
    train_x_set = train_x_set.reshape(train_x_set.shape[0], train_x_set.shape[1], 1)
    test_x_set = test_x_set.reshape(test_x_set.shape[0], test_x_set.shape[1], 1)


    # model = inception_test(input_shape=(train_x_set.shape[1], 1), num_classes=train_y_set.shape[1])
    model = simple_CNN(input_shape=(train_x_set.shape[1], 1), num_classes=train_y_set.shape[1])
    # model = simple_inception(input_shape=(train_x_set.shape[1], 1), num_classes=train_y_set.shape[1])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(train_x_set, train_y_set, validation_split=validation_ratio, epochs=training_epoch, batch_size=batch_size)
    
    preds = model.evaluate(test_x_set, test_y_set)
    print ("Loss = " + str(preds[0]))
    print ("Test Accuracy = " + str(preds[1]))

    test_analysis(model, test_x_set, test_y_set, save_dir+'/'+'training_result', 'test_set_result')
    # Plot the learning curve
    plt.plot(model.history.history['loss'])
    plt.plot(model.history.history['val_loss'])
    plt.title('Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper right')
    plt.savefig(get_save_path(save_dir+'/'+'training_result', 'loss.png'), dpi=300)
    plt.clf()

    plt.plot(model.history.history['acc'])
    plt.plot(model.history.history['val_acc'])
    plt.title('Accuracy')
    plt.ylabel('acc')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(get_save_path(save_dir+'/'+'training_result', 'acc.png'), dpi=300)
    plt.clf()

    # Save model
    model.save(get_save_path(save_dir+'/'+'training_result', 'model.h5'))

    # model.summary()
    plot_model(model, to_file=get_save_path(save_dir+'/'+'training_result', 'model.png'), show_shapes=True)
    # VG(model_to_dot(model).create(prog='dot', format='svg'))