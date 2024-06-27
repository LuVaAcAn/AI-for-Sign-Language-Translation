from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import LSTM, Dense
from constants import MAX_FRAMES, KEYPOINT_COUNT
NUM_EPOCH = 150

def modelo(output_lenght: int):
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(MAX_FRAMES, KEYPOINT_COUNT)))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(128, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(output_lenght, activation='softmax'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model