import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import json
import glob
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Activation, Dense, LSTM
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler

# enable GPU for training
physical_device = tf.config.experimental.list_physical_devices('GPU')
print("Number of GPUs Available: ", len(physical_device))
# use a CUDA GPU it exits
if len(physical_device):
    tf.config.experimental.set_memory_growth(physical_device[0], True)

train_data = []
train_label = []

# obtain path for training data
train_path = glob.glob("combined_jsons/*.json")

for path in train_path:
    with open(path) as f:
        raw_data = json.loads(f.read())
    train_data.append(raw_data)

num_features = len(train_data[0][0])
loaded_train_data = []
loaded_train_label = []
# parameter for sliding window
window_size = 3
slide = window_size

for lst in train_data:
    window_left = 0
    window_right = window_size - 1
    while window_right < len(lst):
        window = []
        for i in range(window_left, window_right + 1):
            window.append(lst[i][1:])
            if i == window_right:
                loaded_train_label.append(lst[i][0])
        loaded_train_data.append(window)
        window_left += slide
        window_right += slide

loaded_train_data = np.array(loaded_train_data)
loaded_train_label = np.array(loaded_train_label)
print(loaded_train_data.shape)
print(loaded_train_label.shape)
print(loaded_train_data.shape[1:])

# need to preprocess and shuffle the data
# before passing into model
loaded_train_label, loaded_train_data = shuffle(loaded_train_label, loaded_train_data)

num_features = loaded_train_data.shape[-1]
# three classes: confuse, not confuse, uncertain
num_classes = 4

# initialize the sequential model
model = Sequential([
    LSTM(64, input_shape=(loaded_train_data.shape[1:])),
    Dense(units=16, activation='relu'),
    Dense(units=32, activation='relu'),
    Dense(units=64, activation='relu'),
    Dense(units=128, activation='relu'),
    Dense(units=256, activation='relu'),
    Dense(units=num_classes, activation='softmax')])

# print the model summary
model.summary()

# compile the model
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# train the model with given data and parameters
EPOCHS = 30
BATCHES = 10
# split some data as validation set
validation_percent = 0.1
history = model.fit(
    x=loaded_train_data,
    y=loaded_train_label,
    validation_split=validation_percent,
    batch_size=BATCHES,
    epochs=EPOCHS,
    shuffle=True,
    verbose=2)

# plot and save the model accuracy and loss
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
#plt.savefig('model_accuracy.png')


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
#plt.savefig('model_loss.png')
