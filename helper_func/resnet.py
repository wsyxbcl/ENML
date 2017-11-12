import numpy as np
from keras import layers
from keras.layers import Input, Add, Dense, Activation, BatchNormalization, Flatten, Conv1D, ZeroPadding1D,AveragePooling1D, MaxPooling1D
from keras.models import Model, load_model
from keras.utils import layer_utils
import pydot
# from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from keras.initializers import glorot_uniform

from en_data_utils import *

import keras.backend as K

def identity_block(X, f, filters, stage, block):
    """
    Arguments:
    X -- input tensor of shape (m, n)
    f -- integer, specifying the length of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    
    Returns:
    X -- output of the identity block, tensor of shape (n, 1)
    """
    
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value. You'll need this later to add back to the main path. 
    X_shortcut = X
    
    # First component of main path
    X = Conv1D(filters=F1, kernel_size=1, strides= 1, name=conv_name_base+'2a', kernel_initializer='glorot_uniform')(X)
    # X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)
    
    ### START CODE HERE ###
    
    # Second component of main path (≈3 lines)
    X = Conv1D(filters=F2, kernel_size=f, strides= 1, padding='same', name=conv_name_base+'2b', kernel_initializer='glorot_uniform')(X)
    # X = BatchNormalization(axis=3, name=bn_name_base+'2b')(X)
    X = Activation('relu')(X)

    # Third component of main path (≈2 lines)
    X = Conv1D(filters=F3, kernel_size=1, strides= 1, name=conv_name_base+'2c', kernel_initializer='glorot_uniform')(X)
    # X = BatchNormalization(axis=3, name=bn_name_base+'2c')(X)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    
    ### END CODE HERE ###
    return X

def convolutional_block(X, f, filters, stage, block, s=2):
    """
    Arguments:
    X -- input tensor of shape (m, n)
    f -- integer, specifying the length of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    s -- Integer, specifying the stride to be used
    
    Returns:
    X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
    """
    
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value
    X_shortcut = X


    ##### MAIN PATH #####
    # First component of main path 
    X = Conv1D(F1, 1, strides=s, name=conv_name_base+'2a', kernel_initializer='glorot_uniform')(X)
    # X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)
    
    ### START CODE HERE ###

    # Second component of main path (≈3 lines)
    X = Conv1D(F2, f, strides=1, padding='same', name=conv_name_base+'2b', kernel_initializer='glorot_uniform')(X)
    # X = BatchNormalization(axis=3, name=bn_name_base+'2b')(X)
    X = Activation('relu')(X)

    # Third component of main path (≈2 lines)
    X = Conv1D(F3, 1, strides=1, name=conv_name_base+'2c', kernel_initializer='glorot_uniform')(X)
    # X = BatchNormalization(axis=3, name=bn_name_base+'2c')(X)

    ##### SHORTCUT PATH #### (≈2 lines)
    X_shortcut = Conv1D(F3, 1, strides=s, name=conv_name_base+'1', kernel_initializer='glorot_uniform')(X_shortcut)
    # X_shortcut = BatchNormalization(axis=3, name=bn_name_base+'1')(X_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    
    ### END CODE HERE ###
    
    return X

def ResNet1D50(input_shape, classes):
    """
    Implementation of 1D version the popular ResNet50 the following architecture:

    (CONV1D -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
    -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER)

    (CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
    -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER)


    Arguments:
    input_shape -- shape of the vecter of the dataset
    classes -- integer, number of classes

    Returns:
    model -- a Model() instance in Keras
    """
    
    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)

    
    # Zero-Padding
    X = ZeroPadding1D(padding=3)(X_input)
    
    # Stage 1
    X = Conv1D(64, 7, strides=2, name='conv1', kernel_initializer='glorot_uniform')(X)
    # X = BatchNormalization(axis = 3, name = 'bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling1D(3, strides=2)(X)

    # Stage 2
    X = convolutional_block(X, f=3, filters=[64, 64, 256], stage=2, block='a', s=1)
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')

    ### START CODE HERE ###

    # Stage 3 (≈4 lines)
    X = convolutional_block(X, f=3, filters=[128, 128, 512], stage=3, block='a', s=2)
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='b')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='c')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='d')

    # Stage 4 (≈6 lines)
    X = convolutional_block(X, f=3, filters=[256, 256, 1024], stage=4, block='a', s=2)
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='b')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='c')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='d')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='e')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='f')

    # Stage 5 (≈3 lines)
    X = convolutional_block(X, f=3, filters=[512, 512, 2048], stage=5, block='a', s=2)
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='b')
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='c')

    # AVGPOOL (≈1 line). Use "X = AveragePooling2D(...)(X)"
    X = AveragePooling1D(pool_size=2, name='avg_pool')(X)
    
    ### END CODE HERE ###

    # output layer
    X = Flatten()(X)
    X = Dense(classes, activation='softmax', name='fc'+str(classes), kernel_initializer='glorot_uniform')(X)
    
    
    # Create model
    model = Model(inputs = X_input, outputs = X, name='ResNet1D50')

    return model

if __name__ == '__main__':

    # K.set_image_data_format('channels_last')
    # K.set_learning_phase(1)

    dataset_dir = '/mnt/t/college/last/finaldesign/ENML/code/test/20171112_test'
    x_orig, y_orig, coordinates = load_dataset(dataset_dir)
    # TODO Random shuffle and get training/test set
    train_x_set = x_orig - np.mean(x_orig, axis=1).reshape(np.shape(x_orig)[0], 1)
    # Important here, for input shape of Conv1D is (batch_size, steps, input_dim)
    train_x_set = train_x_set.reshape(train_x_set.shape[0], train_x_set.shape[1], 1)
    train_x_set = np.multiply(train_x_set, 1e8)
    # Got question here, (m, n, 1)???
    # train_y_set = y_orig.astype(float)
    train_y_set = y_orig

    model = ResNet1D50(input_shape=(train_x_set.shape[1], 1), classes=train_y_set.shape[1])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(train_x_set, train_y_set, epochs = 64, batch_size = 32)
    
    # preds = model.evaluate(X_test, Y_test)
    # print ("Loss = " + str(preds[0]))
    # print ("Test Accuracy = " + str(preds[1]))

    # model.summary()
    # plot_model(model, to_file='model.png')
    # VG(model_to_dot(model).create(prog='dot', format='svg'))