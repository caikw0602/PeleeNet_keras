from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, SeparableConv2D
from keras.layers import Input, Add, Activation, BatchNormalization, Reshape, Lambda
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.callbacks import ModelCheckpoint,LearningRateScheduler,EarlyStopping
from keras.optimizers import Adam, SGD
from keras.datasets import cifar10
from peleenet import PeleeNet
from matplotlib import pyplot as plt
import os
import keras
from keras import backend as K
import numpy as np
np.random.seed(42)

# Add GPU option
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

from keras.preprocessing.image import ImageDataGenerator


# The data, split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
num_classes = 10
# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)



class PlotLosses(keras.callbacks.Callback):
        def on_train_begin(self, logs={}):
            self.i = 0
            self.x = []
            self.losses = []
            self.val_losses = []
        
            self.fig = plt.figure()
        
            self.logs = []

        def on_epoch_end(self, epoch, logs={}):
        
            self.logs.append(logs)
            self.x.append(self.i)
            self.losses.append(logs.get('loss'))
            self.val_losses.append(logs.get('val_loss'))
            self.i += 1
        
            #clear_output(wait=True)
            fig_loss = plt.figure(figsize=(10, 10))
            plt.plot(self.x, self.losses, label="loss")
            plt.plot(self.x, self.val_losses, label="val_loss")
            plt.title("Learning Curve")
            plt.xlabel("Epoch #")
            plt.ylabel("Loss")
            plt.legend()
            fig_loss.savefig("learning_curve_peleenet_cifar10.jpg") 
            #plt.savefig('learning_curve.jpg')
            #plt.show()



# uncomment below if training from scratch
#tr_model = PeleeNet()

# load existing model
tr_model = load_model('model_pelee_cifar10.h5')
tr_model.summary()

model_name ='model_pelee_cifar10.h5'
plot_lr = PlotLosses()
    
# model saving
checkpoint = ModelCheckpoint(model_name,monitor='val_acc',verbose=1,save_best_only=True)
early_stop = EarlyStopping(monitor='val_acc',min_delta=0,patience=200,verbose=1,mode='auto')

# Compile the model
#sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

tr_model.compile(loss='categorical_crossentropy',optimizer=Adam(1e-5),metrics=['accuracy'])


x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255


data_augmentation = True

if not data_augmentation:
    print('Not using data augmentation.')
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True)
else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        zca_epsilon=1e-06,  # epsilon for ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        # randomly shift images horizontally (fraction of total width)
        width_shift_range=0.1,
        # randomly shift images vertically (fraction of total height)
        height_shift_range=0.1,
        shear_range=0.,  # set range for random shear
        zoom_range=0.,  # set range for random zoom
        channel_shift_range=0.,  # set range for random channel shifts
        # set mode for filling points outside the input boundaries
        fill_mode='nearest',
        cval=0.,  # value used for fill_mode = "constant"
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False,  # randomly flip images
        # set rescaling factor (applied before any other transformation)
        rescale=None,
        # set function that will be applied on each input
        preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        data_format=None,
        # fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=0.0)

    # Compute quantities required for feature-wise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train)

    # Fit the model on the batches generated by datagen.flow().
    tr_model.fit_generator(datagen.flow(x_train, y_train,
                                     batch_size=32),
                        epochs=1000,
                        steps_per_epoch=1400,
                        validation_data=(x_test, y_test),
                        workers=4,
                        callbacks = [checkpoint,plot_lr,early_stop])

# plot the results
plt.figure()
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model_accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','test'])
plt.savefig('peleenet_acc_cifar.jpg')
