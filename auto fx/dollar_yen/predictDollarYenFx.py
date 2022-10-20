import pandas as pd
import numpy as np
import os
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.models import Sequential

def build_modelmodel_name='LSTM'):
    simple_model=Sequential()
    simple_model.add(Dense(8, activation="relu", input_shape=(x_train.shape[1], self.H_HIDDEN)))
    simple_model.add(Dense(8, activation="relu"))
    simple_model.add(Dense(self.UNITS_03, activation="relu"))
    simple_model.add(Dense(1))

    return simple_model


def main():
    #モデルの読み込み
    model = build_simple_model()

    #学習した重みを読み込み
    #先にget_data_and_train.pyを実行していないと、param.hdf5が存在しないのでエラーになる
    hdf5_path = os.path.dirname(os.path.abspath(__file__)) + '/graphs/hdf5s/'
    model.load_weights(hdf5_path + 'param.hdf5')



if __name__ == '__main__':

    main()
