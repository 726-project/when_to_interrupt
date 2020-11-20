import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler

# feature value for every data
train_data = []
# data label
train_labels = []

# add data to corresponding array
# ...

train_data = np.array(train_data)
train_labels = np.array(train_labels)

# need to preprocess and shuffle the data
# before passing into model
# ...

num_features = train_data.shape[-1]
# three classes: confuse, not confuse, uncertain
num_classes = 3;

# enable GPU for training
physical_device = tf.config.experimental.list_physical_devices('GPU')
print("Number of GPUs Available: ", len(physical_device))
# check whether a CUDA GPU exits
if len(physical_device):
    tf.config.experimental.set_memory_growth(physical_device[0], True)

# initialize the sequential model
model = Sequential([
    Dense(units=16, input_shape=(num_features,), activation='relu'),
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
validation_precent = 0.1
history = model.fit(
    x=train_data,
    y=train_labels,
    validation_split=validation_precent,
    batch_size=BATCHES,
    epochs=EPOCHS,
    shuffle=True,
    verbose=2)

# plot amd save the model accuracy and loss
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
plt.savefig('model_accuracy.png')


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
plt.savefig('model_loss.png')

