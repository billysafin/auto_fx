from tensorflow.keras.layers import Activation, Dense
def build_model():
   model = Sequential()
   model.add(LSTM(50, activation='tanh', input_shape=(4, 104), return_sequences=True))
   model.add(LSTM(104, return_sequences=False))
   model.add(Dropout(0.2))
   model.add(Dropout(0.2))
   model.add(Dense(8, activation="relu", input_shape=(4, 104)))
   model.add(Dense(8, activation="relu"))
   model.add(Dense(16, activation="relu"))
   model.add(Dense(1))
   return model