from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM

def lstm():
    model = Sequential()
    model.add(LSTM(units=100, input_shape=(1027, 81), return_sequences=True))
    model.add(Dense(2, activation='softmax'))

    return model
