# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Dropout
from keras.layers import ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam


#from keras.callbacks import CSVLogger

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

from keras import backend as K
K.tensorflow_backend._get_available_gpus()
print(K.tensorflow_backend._get_available_gpus())

# Initialising the CNN
classifier = Sequential()

classifier.add(ZeroPadding2D((1,1),input_shape = (64, 64, 3)))

# Step 1 - Convolution
classifier.add(Conv2D(48, (5, 5), activation = 'relu'))


# Step 2 - Convolution
classifier.add(ZeroPadding2D((1,1)))
classifier.add(Conv2D(128, (3, 3), activation = 'relu'))


# step 3 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 4 - Convolution
classifier.add(ZeroPadding2D((1,1)))
classifier.add(Conv2D(192, (2, 2), activation = 'relu'))


# step 5 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 6 - Convolution
classifier.add(ZeroPadding2D((1,1)))
classifier.add(Conv2D(192, (2, 2), activation = 'relu'))

# Step 6 - Convolution
classifier.add(ZeroPadding2D((1,1)))
classifier.add(Conv2D(128, (2, 2), activation = 'relu'))


# step 7 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 8 - Flattening
classifier.add(Flatten())

# Step 14 - Full connection
classifier.add(Dense(units = 1024, activation = 'relu'))
classifier.add(Dropout(rate=0.20))

classifier.add(Dense(units = 1024, activation = 'relu'))
classifier.add(Dropout(rate=0.20))

classifier.add(Dense(units = 200, activation = 'softmax'))

# Compiling the CNN
Adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0, amsgrad=False)
classifier.compile(optimizer = Adam, loss = 'categorical_crossentropy', metrics = ['accuracy'])

#classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])


# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                    shear_range = 0.2,
                                    zoom_range = 0.2,
                                    horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('tiny-imagenet-200/train',
                                                    target_size = (64, 64),
                                                    batch_size = 256,
                                                    class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('tiny-imagenet-200/validation',
                                            target_size = (64, 64),
                                            batch_size = 256,
                                            class_mode = 'categorical')

history = classifier.fit_generator(training_set,
                            epochs = 25,
                            validation_data = test_set,
                            verbose=1)

import matplotlib.pyplot as plt

print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
