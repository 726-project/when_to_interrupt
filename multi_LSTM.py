import sys
import numpy as np
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, concatenate
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle

ENABLE_GPU = False

HIDDEN_STATE_VECTOR_DIM = 127
EPOCHS = 30
BATCHES = 700
DROP_OUT = 0.5

IS_SHUFFLE = True


# this implementation is for baseline LSTM model training
def main():
    frame_sequences = np.load('LSTM_input.npy')
    labels = np.load('LSTM_labels.npy') #sequences labels not frame labels

    num_features = frame_sequences.shape[-1]
    num_classes = 3 #confuse, not confuse, uncertain


    scaler_op = MinMaxScaler(feature_range=(-2,2))
    scaler_h = MinMaxScaler(feature_range=(-2,2))
    for i in range(len(frame_sequences)):
        frame_sequences[i][:,0:54] = scaler_op.fit_transform(frame_sequences[i][:,0:54])# normalize openpose 2d position
        frame_sequences[i][:, 54:] = scaler_h.fit_transform(frame_sequences[i][:, 54:]) # normalize the rest feature

    if IS_SHUFFLE:
        frame_sequences, labels = shuffle(frame_sequences, labels)

    # Extract feature input
    openpose_input = []
    head_ori_input = []
    for i in range(len(frame_sequences)):
        openpose_input.append(frame_sequences[i][:, 0:54])
        head_ori_input.append(frame_sequences[i][:, 54:])
    openpose_input = np.array(openpose_input)
    head_ori_input = np.array(head_ori_input)

    print(openpose_input.shape)
    print(head_ori_input.shape)

    op_input_obj = keras.Input(shape=openpose_input.shape[1:], name="op")
    head_input_obj = keras.Input(shape=head_ori_input.shape[1:], name="head")

    op_features = LSTM(HIDDEN_STATE_VECTOR_DIM, activation='tanh', return_sequences=True)(op_input_obj)
    op_features = Dropout(DROP_OUT)(op_features)
    op_features = LSTM(HIDDEN_STATE_VECTOR_DIM, activation='tanh')(op_features)
    op_features = Dropout(DROP_OUT)(op_features)
    op_features = Dense(64, activation='relu')(op_features)

    head_features = LSTM(32, activation='tanh', return_sequences=True)(head_input_obj)
    head_features = Dropout(DROP_OUT)(head_features)
    head_features = LSTM(32, activation='tanh')(head_features)
    head_features = Dropout(DROP_OUT)(head_features)
    head_features = Dense(64, activation='relu')(head_features)

    x = concatenate([op_features, head_features])

    confusion_pred = Dense(num_classes, activation='softmax', name="confusion_pred")(x)

    model = keras.Model(
        inputs=[op_input_obj, head_input_obj],
        outputs=confusion_pred,
    )    

    # keras.utils.plot_model(model, "multi_input_model.png", show_shapes=True)



    model.compile(optimizer=Adam(learning_rate=0.0005),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])



    history = model.fit(
        {"op": openpose_input, "head": head_ori_input},
        {"confusion_pred": labels},
        validation_split=0.1,
        epochs=EPOCHS,
        batch_size=BATCHES,
    )


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

