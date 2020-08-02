import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

tf.keras.backend.set_floatx('float32')
tf.compat.v1.disable_eager_execution()

import pickle
import numpy as np
import matplotlib.pyplot as plt

c_InputNumber = 32,32,3
c_OutputNumber = 10
c_Lr = 0.0005
c_Batchsize = 32
c_Epochs = 80
c_FolderPath = ''
c_Filepath_Train = f'data_batch_'
c_Filepath_Test = f'test_batch'

class DenseModel:
    def __init__(self, InputNumber, OutputNumber, Lr):
        self.InputNumber = InputNumber
        self.OutputNumber = OutputNumber
        self.model = self.CreateModel()
        self.opt = tf.keras.optimizers.Adam(Lr, amsgrad=True)
        self.tensorboard_callback = keras.callbacks.TensorBoard(log_dir='logs')
    
    def Scheduler(self, epoch):
        if epoch < 10:
            return 0.005
        else:
            return 0.005 * tf.math.exp(0.1 * (10 - epoch))
    
    def CreateModel(self):
        channels = [16, 32, 64]
        input = tf.keras.layers.Input(self.InputNumber)
        x = tf.keras.layers.Conv2D(channels[0], kernel_size=(3, 3), padding='same',
                                   kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(input)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation(tf.nn.relu)(x)
        
        for c in channels:
            for i in range(9):
                subsampling = i == 0 and c > 16
                strides = (2, 2) if subsampling else (1, 1)
                y = tf.keras.layers.Conv2D(c, kernel_size=(3, 3), padding='same', strides=strides,
                                           kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
                y = tf.keras.layers.BatchNormalization()(y)
                y = tf.keras.layers.Activation(tf.nn.relu)(y)
                y = tf.keras.layers.Conv2D(c, kernel_size=(3, 3), padding='same',
                                           kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(y)
                y = tf.keras.layers.BatchNormalization()(y)
                if subsampling:
                    x = tf.keras.layers.Conv2D(c, kernel_size=(1, 1), strides=(2, 2), padding='same', 
                                               kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
                x = tf.keras.layers.Add()([x, y])
                x = tf.keras.layers.Activation(tf.nn.relu)(x)
        
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Flatten()(x)
        labels = tf.keras.layers.Dense(self.OutputNumber, activation='softmax', kernel_initializer='he_normal')(x)
        return tf.keras.Model(inputs=[input], outputs=[labels])
    
    def CompileModel(self):
        self.model.compile(loss='categorical_crossentropy', optimizer=self.opt, metrics=['accuracy'])
    
    def TrainModel(self, x_train, x_valid, y_train, y_valid, batch_size, epochs):
        igen = ImageDataGenerator(rotation_range=10, horizontal_flip=True, width_shift_range=0.1, height_shift_range=0.1)
        igen.fit(x_train)
        callback = [tf.keras.callbacks.LearningRateScheduler(self.Scheduler),
                    ] #self.tensorboard_callback, tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, mode='auto', restore_best_weights=True)
        history = self.model.fit_generator(igen.flow(x_train, y_train, batch_size=batch_size), steps_per_epoch=len(x_train)/batch_size,
                                           validation_data=(x_valid, y_valid), epochs=epochs, callbacks=callback, verbose=2)
        return history
    
    def EvaluateModel(self, x_test, y_test):
        score = self.model.evaluate(x_test, y_test)
        return score[1]

class Agent:
    def __init__(self, InputNumber, OutputNumber, Lr):
        self.InputNumber = InputNumber
        self.OutputNumber = OutputNumber
        self.Lr = Lr
        self.dense = DenseModel(self.InputNumber, self.OutputNumber, self.Lr)

    def Unpickle(self, file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict
    
    def Pixelize(self, flat):
        return np.reshape(flat, (32,32,3))
    
    def Onehot(self, value):
        onehot = np.zeros((10))
        onehot[value] = 1
        return onehot
    
    def Encode(self, dict):
        x = np.zeros((10000,32,32,3))
        y = np.zeros((10000,10))
        for i in range(10000):
            x[i][:][:][:] =  self.Pixelize(dict[b'data'][i])
            y[i][:] = self.Onehot(dict[b'labels'][i])
        return x, y
    
    def Run(self, Filepath_Test, Filepath_Train, Batchsize, Epochs):
        self.dense.CompileModel()
        dict_test = self.Unpickle(Filepath_Test)
        x_test, y_test = self.Encode(dict_test)
        
        for i in range(1, 6):
            dict_train = self.Unpickle(Filepath_Train + str(i))
            if i == 1:
                x_train, y_train = self.Encode(dict_train)
            else:
                x, y = self.Encode(dict_train)
                x_train = np.vstack((x_train,x))
                y_train = np.vstack((y_train,y))
        
        x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.1, shuffle=True)
        history = self.dense.TrainModel(x_train, x_valid, y_train, y_valid, Batchsize, Epochs)
        print('Epoch Test')
        accuracy = self.dense.EvaluateModel(x_test, y_test)
        print('Accuracy: ', accuracy)
        
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()

## Main Start ##
agent = Agent(c_InputNumber, c_OutputNumber, c_Lr)
agent.Run(c_Filepath_Test, c_Filepath_Train, c_Batchsize, c_Epochs)