import sys
import numpy as np
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle

ENABLE_GPU = False

HIDDEN_STATE_VECTOR_DIM = 127
EPOCHS = 60
BATCHES = 500

IS_SHUFFLE = True

# this implementation is for baseline LSTM model training
def main():
    frame_sequences = np.load('LSTM_input.npy')
    labels = np.load('LSTM_labels.npy') #sequences labels not frame labels

    num_features = frame_sequences.shape[-1]
    # num_classes = 3 #confuse, not confuse, uncertain
    num_classes = 3 #confuse, not confuse, uncertain

    model = Sequential()

    # input_shape=(frame_sequences.shape[1:]) #(WINDOW_SIZE, num_features)
    # tanh activation get better accuracy than relu
    model.add(LSTM(HIDDEN_STATE_VECTOR_DIM, activation='tanh', return_sequences=True))
    model.add(Dropout(0.2))

    # two layers are better than single layer of LSTM
    model.add(LSTM(HIDDEN_STATE_VECTOR_DIM, activation='tanh'))
    model.add(Dropout(0.2))

    model.add(Dense(num_classes, activation='softmax'))
    # mean_squared_error or categorical_crossentropy
    # using mean_squared_error results in bad accuracy
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    scaler_op = MinMaxScaler(feature_range=(-2,2))
    scaler_h = MinMaxScaler(feature_range=(-2,2))
    for i in range(len(frame_sequences)):
        frame_sequences[i][:,0:54] = scaler_op.fit_transform(frame_sequences[i][:,0:54])# normalize openpose 2d position
        frame_sequences[i][:, 54:] = scaler_h.fit_transform(frame_sequences[i][:, 54:]) # normalize the rest feature

    if IS_SHUFFLE:
        frame_sequences, labels = shuffle(frame_sequences, labels)

    history = model.fit(
        x=frame_sequences,
        y=labels,
        validation_split=0.1,
        batch_size=BATCHES,
        epochs=EPOCHS)


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

