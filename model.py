from tensorflow import keras
#from keras import models,layers, regularizers
#from keras.callbacks import ModelCheckpoint,EarlyStopping
#from keras.models import model_from_json
#from keras.layers import Dropout
import json

import matplotlib as plt


## Simple NN using keras ##
class NN():
    '''Simple neural network that learns to evaluate a given preprocessed board.'''
    def build_model(self,X_shape):
        model = keras.models.Sequential()
        # input layer
        model.add(keras.layers.Dropout(0.2, input_shape=(X_shape,)))
        model.add(keras.layers.Dense(768, input_shape=(X_shape,), activation='relu'))
        # hidden layer
        model.add(keras.layers.Dropout(0.2))
        model.add(keras.layers.Dense(768, activation='relu', kernel_regularizer = keras.regularizers.l2(0.01)))
        model.add(keras.layers.Dropout(0.2))
        model.add(keras.layers.Dense(768, activation='relu', kernel_regularizer = keras.regularizers.l2(0.01)))
        # output layer
        model.add(keras.layers.Dense(1, activation='tanh'))

        model.compile(optimizer='adam',
                        loss='mse',
                        metrics=['mean_squared_error'])
        return model

    def train(self, model, X_train, y_train, X_val, y_val, batch_size):
        print('\n### Starting training on {} chess moves. ###'.format(len(y_train)))
        checkpoint = keras.callbacks.ModelCheckpoint("trained_nets/best_model_dropout2_big.hdf5", monitor='loss', verbose=1,
            save_best_only=True, mode='auto', period=1)
        earlystopping=keras.callbacks.EarlyStopping(monitor="mean_squared_error", patience=0, verbose=0, mode='auto')
        history = model.fit(X_train, y_train,
                  batch_size=batch_size,
                  epochs=20,
                  verbose=1,
                  validation_data=(X_val, y_val),
                  callbacks=[checkpoint])
        # Save the model architecture
        with open('trained_nets/model_architecture_dropout2_big.json', 'w') as f:
            f.write(model.to_json())
        print('### Finished training. ###')
        # Plot training & validation accuracy values
        plt.plot(history.history['mean_squared_error'])
        plt.plot(history.history['val_mean_squared_error'])
        plt.title('Model accuracy')
        plt.ylabel('mean_squared_error')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()

        # Plot training & validation loss values
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()
    def load_model(self):
        # Model reconstruction from JSON file
        with open('trained_nets/model_architecture_dropout2_big.json', 'r') as f:
            model = keras.models.model_from_json(f.read())
        # Load weights into the new model
        model.load_weights('trained_nets/best_model_dropout2_big.hdf5')
        # Check models
        model.compile(optimizer='adam',
                        loss='mse',
                        metrics=['mean_squared_error'])
        return model

## predict board state
def predict(model,X_test):
    '''This function evaluates a given preprocessed board state using the trained NN
    returns score [-1,1], where 1 = White wins
    '''
    # Evaluate board
    score = model.predict(X_test, verbose=0)
    return score
