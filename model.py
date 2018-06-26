from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM

def lstm():
    model = Sequential()
    model.add(LSTM(units=100, input_shape=(1027, 81), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(Dense(2, activation='softmax'))

    return model
