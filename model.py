import sys
import numpy as np
import tensorflow as tf
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Dropout, Input, concatenate
from keras.optimizers import Adam
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
from keras.metrics import categorical_crossentropy
import matplotlib.pyplot as plt

IS_SHUFFLE = False
HIDDEN_STATE_VECTOR_DIM = 16
EPOCHS = 40
BATCHES = 100
IS_ALL = True
IS_HAT = False
IS_OP = False
IS_MULTIPLE_INPUT = True
# this implementation is for baseline LSTM model training
def main():
    frame_sequences = np.load('LSTM_input.npy')
    labels = np.load('LSTM_labels.npy') #sequences labels not frame labels

    num_features = frame_sequences.shape[-1]
    num_classes = 3 #confuse, not confuse, uncertain

    # normalize openpose 2d position between -1,1
    # normalize the rest feature between 0,1
    scaler_op = MinMaxScaler(feature_range=(-1,1))
    scaler_h = MinMaxScaler(feature_range=(0,1))
    if IS_ALL:
        for i in range(len(frame_sequences)):
            frame_sequences[i][:,0:54] = scaler_op.fit_transform(frame_sequences[i][:,0:54])# normalize openpose 2d position
            frame_sequences[i][:, 54:] = scaler_h.fit_transform(frame_sequences[i][:, 54:]) # normalize the rest feature
    elif IS_HAT:
        for i in range(len(frame_sequences)):
            frame_sequences[i][:, 0:] = scaler_h.fit_transform(frame_sequences[i][:, 0:])  # normalize the rest feature
    elif IS_OP:
        for i in range(len(frame_sequences)):
            frame_sequences[i][:, 0:] = scaler_op.fit_transform(frame_sequences[i][:, 0:])  # normalize openpose 2d position
    print(frame_sequences.shape)
    # shuffle data if is shuffle
    if IS_SHUFFLE:
        frame_sequences, labels = shuffle(frame_sequences, labels)

    if IS_ALL:
        openpose_input = []
        head_ori_input = []
        for i in range(len(frame_sequences)):
            openpose_input.append(frame_sequences[i][:,0:54])
        openpose_input = np.array(openpose_input)
        for i in range(len(frame_sequences)):
            head_ori_input.append(frame_sequences[i][:,54:])
        head_ori_input = np.array(head_ori_input)

    if not IS_MULTIPLE_INPUT:
        model = Sequential()
        # input_shape=(frame_sequences.shape[1:]) #(WINDOW_SIZE, num_features)
        model.add(LSTM(HIDDEN_STATE_VECTOR_DIM, input_shape=(frame_sequences.shape[1:]), activation='relu', return_sequences=True))
        model.add(Dropout(0.5))

        model.add(LSTM(HIDDEN_STATE_VECTOR_DIM, activation='relu'))
        model.add(Dropout(0.5))

        model.add(Dense(num_classes, activation='softmax'))

        model.compile(optimizer=Adam(lr=1e-4, decay=1e-5),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        history = model.fit(
            x=frame_sequences,
            y=labels,
            validation_split=0.1,
            batch_size=BATCHES,
            epochs=EPOCHS,
            shuffle=IS_SHUFFLE)
    else:
        inputA = Input(shape=(openpose_input.shape[1:]))
        inputB = Input(shape=(head_ori_input.shape[1:]))

        x = LSTM(HIDDEN_STATE_VECTOR_DIM, activation='relu')(inputA)
        x = Dropout(0.5)(x)
        x = Dense(8, activation="relu")(x)
        x = Model(inputs=inputA, outputs=x)

        y = LSTM(HIDDEN_STATE_VECTOR_DIM, activation='relu')(inputB)
        y = Dropout(0.5)(y)
        y = Dense(8, activation="relu")(y)
        y = Model(inputs=inputB, outputs=y)

        combined = concatenate([x.output, y.output])

        z = Dense(16, activation="relu")(combined)
        z = Dense(num_classes, activation="softmax")(z)
        model = Model(inputs=[x.input, y.input], outputs=z)

        model.compile(optimizer=Adam(lr=1e-3, decay=1e-5),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        history = model.fit(
            x=[openpose_input,head_ori_input],
            y=labels,
            validation_split=0.1,
            batch_size=BATCHES,
            epochs=EPOCHS,
            shuffle=IS_SHUFFLE)


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
    gpus = tf.config.experimental.list_physical_devices('GPU')
    # use a CUDA GPU it exits
    if gpus:
        try:
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)
    main()

