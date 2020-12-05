import sys
import numpy as np
import glob
import json
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle

ENABLE_GPU = False

HIDDEN_STATE_VECTOR_DIM = 127
EPOCHS = 40
BATCHES = 50

IS_SHUFFLE = True


# this implementation is for baseline LSTM model training
def main():
    data = []
    labels = []
    train_path = glob.glob("processed_data/all/*.json")
    for path in train_path:
        with open(path) as f:
            raw_data = json.loads(f.read())
        for frame in raw_data:
            data.append(frame[1:])
            labels.append(frame[0])


    num_features = len(data[0])
    num_classes = 3 #confuse, not confuse, uncertain

    data_np = np.array(data)
    labels_np = np.array(labels)


    model = Sequential()

    model.add(Dense(HIDDEN_STATE_VECTOR_DIM, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(HIDDEN_STATE_VECTOR_DIM, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(num_classes, activation='softmax'))



    model.compile(optimizer=Adam(learning_rate=0.00001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    scaler_op = MinMaxScaler(feature_range=(-2,2))
    scaler_h = MinMaxScaler(feature_range=(-2,2))

    data_np[:,0:54] = scaler_op.fit_transform(data_np[:,0:54])
    data_np[:,54:] = scaler_h.fit_transform(data_np[:,54:])

    if IS_SHUFFLE:
        data_np, labels_np = shuffle(data_np, labels_np)

    history = model.fit(
        x=data_np,
        y=labels_np,
        validation_split=0.2,
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

