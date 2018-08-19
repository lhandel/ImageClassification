
# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import BatchNormalization
from keras.optimizers import Adam
from keras import regularizers
from keras.layers import ZeroPadding2D

from keras.callbacks import CSVLogger

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

from keras import backend as K
K.tensorflow_backend._get_available_gpus()
print(K.tensorflow_backend._get_available_gpus())

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
# Step 1 - Convolution
classifier.add(Conv2D(32, (2, 2), input_shape = (64, 64, 3), activation = 'relu'))
classifier.add(BatchNormalization())

classifier.add(Conv2D(32, (2, 1), activation = 'relu'))
classifier.add(BatchNormalization())

classifier.add(Conv2D(32, (1, 2), activation = 'relu'))
classifier.add(BatchNormalization())

# Adding more convolutional layer
classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Conv2D(48, (2, 2), activation = 'relu'))
classifier.add(BatchNormalization())

classifier.add(Conv2D(48, (2, 2), activation = 'relu'))
classifier.add(BatchNormalization())

classifier.add(Conv2D(48, (2, 2), activation = 'relu'))
classifier.add(BatchNormalization())

# Adding more convolutional layer
classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Conv2D(64, (2, 2), activation = 'relu'))
classifier.add(BatchNormalization())

classifier.add(Conv2D(64, (2, 2), activation = 'relu'))
classifier.add(BatchNormalization())

classifier.add(Conv2D(64, (2, 2), activation = 'relu'))
classifier.add(BatchNormalization())

# Adding more convolutional layer
classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Conv2D(80, (2, 2), activation = 'relu'))
classifier.add(BatchNormalization())

classifier.add(Conv2D(80, (2, 2), activation = 'relu'))
classifier.add(BatchNormalization())

classifier.add(Conv2D(80, (2, 2), activation = 'relu'))
classifier.add(BatchNormalization())

# Adding more convolutional layer
classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(ZeroPadding2D((1,1)))
classifier.add(Conv2D(96, (2, 2), activation = 'relu'))
classifier.add(BatchNormalization())

classifier.add(ZeroPadding2D((1,1)))
classifier.add(Conv2D(96, (2, 2), activation = 'relu'))
classifier.add(BatchNormalization())

classifier.add(ZeroPadding2D((1,1)))
classifier.add(Conv2D(96, (2, 2), activation = 'relu'))
classifier.add(BatchNormalization())

classifier.add(MaxPooling2D(pool_size = (2, 2)))


# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units = 800, activation = 'relu', kernel_initializer='glorot_uniform', kernel_regularizer=regularizers.l2(0.00001)))
classifier.add(Dense(units = 800, activation = 'relu', kernel_initializer='glorot_uniform', kernel_regularizer=regularizers.l2(0.00001)))
classifier.add(Dense(units = 200, activation = 'softmax'))

# Compiling the CNN

from keras.callbacks import LearningRateScheduler
learning_rate = 1e-4
lr = 0.001
def updateLR(epoch, lr):
    if(epoch % 10 == 0 and epoch !=0):
        lr *= 0.5
        print("update",lr)
    return lr

lrate = LearningRateScheduler(updateLR)

Adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.00014, amsgrad=False)
classifier.compile(optimizer = Adam, loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('tiny-imagenet-200/train',
                                                 target_size = (64, 64),
                                                 batch_size = 23,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('tiny-imagenet-200/validation',
                                            target_size = (64, 64),
                                            batch_size = 23,
                                            class_mode = 'categorical')
csv_logger = CSVLogger('training.log')
history = classifier.fit_generator(training_set,
                         epochs = 30,
                         validation_data = test_set,
                         verbose=1,
                         callbacks=[lrate])


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
