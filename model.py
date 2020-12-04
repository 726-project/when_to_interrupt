import sys
import numpy as np
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.optimizers import Adam
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
from keras.metrics import categorical_crossentropy
import matplotlib.pyplot as plt


ENABLE_GPU = True

HIDDEN_STATE_VECTOR_DIM = 256
EPOCHS = 30
BATCHES = 10

# this implementation is for baseline LSTM model training
def main():
    frame_sequences = np.load('LSTM_input.npy')
    labels = np.load('LSTM_labels.npy') #sequences labels not frame labels

    num_features = frame_sequences.shape[-1]
    num_classes = 4 #confuse, not confuse, uncertain
    model = Sequential()

    # normalize openpose 2d position between -1,1
    # normalize the rest feature between 0,1
    scaler_op = MinMaxScaler(feature_range=(-1,1))
    scaler_h = MinMaxScaler(feature_range=(0,1))
    for i in range(len(frame_sequences)):
        frame_sequences[i][:,0:54] = scaler_op.fit_transform(frame_sequences[i][:,0:54])
        frame_sequences[i][:, 54:] = scaler_h.fit_transform(frame_sequences[i][:, 54:])

    # input_shape=(frame_sequences.shape[1:]) #(WINDOW_SIZE, num_features)
    model.add(LSTM(HIDDEN_STATE_VECTOR_DIM, input_shape=(frame_sequences.shape[1:]), activation='relu', return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(HIDDEN_STATE_VECTOR_DIM, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(num_classes, activation='softmax'))
    # mean_squared_error or categorical_crossentropy
    # using mean_squared_error results in bad accuracy
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(
        x=frame_sequences,
        y=labels,
        validation_split=0.1,
        batch_size=BATCHES,
        epochs=EPOCHS,
        shuffle=False)

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


if __name__ == '__main__':
    if ENABLE_GPU:
        physical_device = tf.config.experimental.list_physical_devices('GPU')
        print("Number of GPUs Available: ", len(physical_device))
        # use a CUDA GPU it exits
        if len(physical_device):
            tf.config.experimental.set_memory_growth(physical_device[0], True)
    main()

