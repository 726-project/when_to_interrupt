import sys
import numpy as np
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Bidirectional, TimeDistributed, GRU, Activation, Flatten, RepeatVector
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
import tensorflow as tf
from matplotlib import pyplot
import pandas as pd
from keras import regularizers
ENABLE_GPU = False

HIDDEN_STATE_VECTOR_DIM = 127
EPOCHS = 40
BATCHES = 512

IS_SHUFFLE = True

IS_ALL = True
IS_HAT = False
IS_OP = False
NUM_CLASSES = 3 #confuse, not confuse, uncertain

# It will return the base model with one LSTM layer
def get_lstm(backwards):
    model = Sequential()
    model.add(LSTM(HIDDEN_STATE_VECTOR_DIM,activation='relu', go_backwards=backwards))
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    return model

# It will return the base Bidirectional LSTM model
def get_bi_lstm(mode):
    model = Sequential()
    model.add(Bidirectional(LSTM(HIDDEN_STATE_VECTOR_DIM), merge_mode=mode))
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    return model

# It will return a model with Bidirectional LSTM as encoder and LSTM as decoder. Lastly, a dense layer as output layer
def get_bi_complicated_lstm(mode):
    model = Sequential()
    forward_layer = LSTM(HIDDEN_STATE_VECTOR_DIM, return_sequences=True)
    backward_layer = LSTM(HIDDEN_STATE_VECTOR_DIM, activation='tanh', return_sequences=True,
                       go_backwards=True)
    model.add(Bidirectional(forward_layer, merge_mode=mode, backward_layer=backward_layer))
    model.add(Dropout(0.5))
    # model.add(LSTM(HIDDEN_STATE_VECTOR_DIM,kernel_regularizer=regularizers.l2(2),activation='tanh'))
    model.add(LSTM(HIDDEN_STATE_VECTOR_DIM,activation='tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    return model

# This function is used to trained and fit the model
def compileAndFit(model,frame_sequences,labels, drawing):
    model.compile(optimizer=Adam(learning_rate=0.0001,decay=1e-6),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    scaler_op = MinMaxScaler(feature_range=(-2,2))
    scaler_h = MinMaxScaler(feature_range=(-2,2))
    if IS_ALL:
        for i in range(len(frame_sequences)):
            frame_sequences[i][:,:] = scaler_op.fit_transform(frame_sequences[i][:,:])# normalize openpose 2d position
    if IS_SHUFFLE:
        frame_sequences, labels = shuffle(frame_sequences, labels)
    
    if not drawing:
        history = model.fit(
            x=frame_sequences,
            y=labels,
            validation_split=0.1,
            batch_size=BATCHES,
            epochs=EPOCHS)
        return history
    else:
        loss = []
        # Fit and model EPOCHS times and plot the loss values
        for _ in range(EPOCHS):
            history = model.fit(
            x=frame_sequences,
            y=labels,
            validation_split=0.1,
            batch_size=BATCHES,
            epochs=1)
            loss.append(history.history['loss'][0])
        # print(loss)
        return loss

#Plot the accuracy and loss graph 
def plotAccVal(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

# Decide which lstm modes is the best based on the loss value
def which_lstms(frame_sequences,labels):
    res = pd.DataFrame()
    lstm = get_lstm(False)
    res['lstm_forward'] = compileAndFit(lstm,frame_sequences,labels,True)
    # lstm backwards
    lstm_back = get_lstm(True)
    res['lstm_backward'] = compileAndFit(lstm_back,frame_sequences,labels,True)
    # bidirectional concat
    bi_lstm = get_bi_lstm('concat')
    res['bi-lstm_concat'] = compileAndFit(bi_lstm,frame_sequences,labels,True)
    res.plot()
    pyplot.show()

# Decide which mode of Bidirectional LSTM we should use 
def which_lstm_mode(frame_sequences,labels):
    res = pd.DataFrame()

    bi_lstm_sum = get_bi_lstm('sum')
    res['bi_lstm_sum'] = compileAndFit(bi_lstm_sum,frame_sequences,labels,True)

    bi_lstm_concat = get_bi_lstm('concat')
    res['bi_lstm_concat'] = compileAndFit(bi_lstm_concat,frame_sequences,labels,True)

    bi_lstm_mul = get_bi_lstm('mul')
    res['bi_lstm_mul'] = compileAndFit(bi_lstm_mul,frame_sequences,labels,True)

    bi_lstm_ave = get_bi_lstm('ave')
    res['bi_lstm_ave'] = compileAndFit(bi_lstm_ave,frame_sequences,labels,True)
    res.plot()
    pyplot.show()

def main():
    # load the sequences
    frame_sequences = np.load('LSTM_input.npy')
    labels = np.load('LSTM_labels.npy')

    # Please use different function if you want to try different models 
    model = get_bi_complicated_lstm('ave')
    x = compileAndFit(model,frame_sequences,labels, False)
    plotAccVal(x)
    # The fllowing functions will plot which lstm models and which modes of BLSTM is the best
    # which_lstms(frame_sequences,labels)
    # which_lstm_mode(frame_sequences,labels)



if __name__ == '__main__':
    if ENABLE_GPU:
        physical_device = tf.config.experimental.list_physical_devices('GPU')
        print("Number of GPUs Available: ", len(physical_device))
        # use a CUDA GPU it exits
        if len(physical_device):
            tf.config.experimental.set_memory_growth(physical_device[0], True)
    main()