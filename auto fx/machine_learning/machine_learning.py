import math
import pandas_datareader as web
import numpy as np
import pandas as pd
import datetime
import os
import sys

sys.path.append('../dolloar_yen')
from settings import ml_settings

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import SimpleRNN
from tensorflow.keras.layers import GRU
from tensorflow.keras.optimizers import Adam

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import japanize_matplotlib
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.ticker as mticker

from pprint import pprint

class machine_learning():
    def __init__(self):
        # 乱数シードを固定する
        tf.random.set_seed(1234)

        # 機械学習用設定読み込み
        mlst = ml_settings.ml_settings()
        self.mlst = mlst.get_MACHINE_SETTINGS()

    def create_train_data(self, only_end_price_data):
        self.mlst['USED_FOR_TRAINING'] = math.ceil(len(only_end_price_data) * self.mlst['PERCENT_FOR_TRAINING'])

        # 正規化する
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(only_end_price_data)

        # 正規化したデータから訓練で使用する行数分のデータを抽出する
        train_data = scaled_data[0 : self.mlst['USED_FOR_TRAINING'], :]

        return train_data

    def train_data(self, x_train, y_train, hdf5_path):
        history = self.mlst['MODEL'].fit(x_train, y_train, batch_size = self.mlst['BATCH_SIZE'], epochs = self.mlst['EPOCHS'], validation_split = self.mlst['VALIDATION_SPLIT'])

        #学習結果をファイルに保存
        #param.hdf5を生成
        self.mlst['MODEL'].save_weights(hdf5_path + 'param.hdf5')
        return history

    def graph_training(self, history, traing_path):
        plt.plot(history.history['mae'], label='train mae')
        plt.plot(history.history['val_mae'], label='val mae')
        plt.xlabel('epoch')
        plt.ylabel('mae')
        plt.legend(loc='best')
        plt.ylim([0, 5])
        plt.savefig(traing_path + "train.png")
        #plt.show()

    def create_lstm_data(self, train_data):
        self.mlst['H_HIDDEN'] = math.ceil(len(train_data) * self.mlst['PERCENT_FOR_H_HIDDEN'])

        x_train = []
        y_train = []
        for i in range(self.mlst['H_HIDDEN'], len(train_data)):
            xset = []
            for j in range(train_data.shape[1]):
                a = train_data[i - self.mlst['H_HIDDEN'] : i, j]
                xset.append(a)
            x_train.append(xset)
            y_train.append(train_data[i, 0])

        x_train, y_train = np.array(x_train), np.array(y_train)

        # 訓練データのNumpy配列について、奥行を訓練データの数、行をself.mlst['use_as_train日分のデータ、列を抽出したFXデータの種類数、の3次元に変換する
        x_train_3D = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], x_train.shape[2]))

        return x_train_3D, y_train

    def create_test_data(self, dataset, train_data):
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(dataset)

        # 検証データを用意する
        test_data = scaled_data[self.mlst['USED_FOR_TRAINING'] - self.mlst['H_HIDDEN'] : , : ]

        return test_data, scaler

    def build_model(self, x_train, y_train, model_name='LSTM'):
        model_body = ''

        # RNN,LSTM、GRUを選択できるようにする
        if model_name == 'RNN':
            self.mlst['MODEL'].add(SimpleRNN(self.mlst['NURO_01'], input_shape=(self.mlst['USED_FOR_TRAINING'], 1), return_sequences=True))
            self.mlst['MODEL'].add(Dropout(self.mlst['DROPOUT_01']))
            self.mlst['MODEL'].add(SimpleRNN(self.mlst['NURO_02'], return_sequences=True))
            self.mlst['MODEL'].add(Dropout(self.mlst['DROPOUT_02']))

        if model_name == 'LSTM':
            self.mlst['MODEL'].add(LSTM(self.mlst['NURO_01'], activation='tanh', input_shape=(x_train.shape[1], self.mlst['H_HIDDEN']), return_sequences=True))
            self.mlst['MODEL'].add(LSTM(self.mlst['H_HIDDEN'], return_sequences=False))
            self.mlst['MODEL'].add(Dropout(self.mlst['DROPOUT_01']))
            self.mlst['MODEL'].add(Dropout(self.mlst['DROPOUT_02']))
            self.mlst['MODEL'].add(Dense(self.mlst['UNITS_01'], activation="relu", input_shape=(x_train.shape[1], self.mlst['H_HIDDEN'])))
            self.mlst['MODEL'].add(Dense(self.mlst['UNITS_02'], activation="relu"))
            self.mlst['MODEL'].add(Dense(self.mlst['UNITS_03'], activation="relu"))
            self.mlst['MODEL'].add(Dense(1))

            model_body = ('   model = Sequential()' + "\n"
            '   model.add(LSTM(' + str(self.mlst['NURO_01']) + ", activation='tanh', input_shape=(" + str(x_train.shape[1]) + ', ' + str(self.mlst['H_HIDDEN']) + '), return_sequences=True))' + "\n"
            '   model.add(LSTM(' + str(self.mlst['H_HIDDEN']) + ', return_sequences=False))' + "\n"
            '   model.add(Dropout(' + str(self.mlst['DROPOUT_01']) + '))' + "\n"
            '   model.add(Dropout(' + str(self.mlst['DROPOUT_02']) + '))' + "\n"
            '   model.add(Dense(' + str(self.mlst['UNITS_01']) + ', activation="relu", input_shape=(' + str(x_train.shape[1]) + ', ' + str(self.mlst['H_HIDDEN']) + ')))' + "\n"
            '   model.add(Dense(' + str(self.mlst['UNITS_02']) + ', activation="relu"))' + "\n"
            '   model.add(Dense(' + str(self.mlst['UNITS_03']) + ', activation="relu"))' + "\n"
            '   model.add(Dense(1))' + "\n")

        if model_name == 'GRU':
            self.mlst['MODEL'].add(GRU(self.mlst['NURO_01'], input_shape=(self.mlst['USED_FOR_TRAINING'], 1), return_sequences=True))
            self.mlst['MODEL'].add(Dropout(self.mlst['DROPOUT_01']))
            self.mlst['MODEL'].add(GRU(self.mlst['NURO_02'], return_sequences=True))
            self.mlst['MODEL'].add(Dropout(self.mlst['DROPOUT_02']))

        self.mlst['MODEL'].add(Activation("linear"))
        self.mlst['MODEL'].compile(loss="mean_squared_error", optimizer="sgd")
        self.mlst['MODEL'].compile(loss="mse", optimizer=Adam(learning_rate = self.mlst['lr']), metrics=["mae"])

        model_body + ('   model.add(Activation("linear"))' + "\n"
            '   modelcompile(loss="mean_squared_error", optimizer="sgd")' + "\n"
            '   model.compile(loss="mse", optimizer=Adam(learning_rate=' + str(self.mlst['lr']) + '), metrics=["mae"])')

        #self.mlst['MODEL'].compile(optimizer='adam', loss='mse', metrics=["accuracy"])

        #モデルを書く
        self.create_model_for_prediction(model_body)

    def make_prediction(self, test_data):
        x_test = []
        y_test = []

        for i in range(self.mlst['H_HIDDEN'], len(test_data)):
            xset = []
            for j in range(test_data.shape[1]):
                a = test_data[i - self.mlst['H_HIDDEN'] : i, j]
                xset.append(a)
            x_test.append(xset)
            y_test.append(test_data[i, 0])

        # 検証データをNumpy配列に変換する
        x_test = np.array(x_test)

        # 検証データのNumpy配列について、奥行を訓練データの数、行をself.mlst['use_as_train分のデータ、列を抽出した株価データの種類数、の3次元に変換する
        x_test_3D = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], x_test.shape[2]))

        # モデルに検証データを代入して予測を行う
        predictions = self.mlst['MODEL'].predict(x_test_3D)

        return predictions, y_test

    def get_model_summary(self, predictions, y_test):
        # モデルの精度を評価する
        # 決定係数とRMSEを計算する
        # 決定係数は1.0に、RMSEは0.0に近いほど、モデルの精度は高い
        r2_score_result = r2_score(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        scores = f'r2_score: {r2_score_result:.4f}  &  ' + f'rmse: {rmse:.4f}'

        return self.mlst['MODEL'].summary(), scores

    def get_loses(self, history):
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        plt.plot(np.arange(len(loss)), loss, label='loss')
        plt.plot(np.arange(len(val_loss)), val_loss, label='val_loss')
        plt.grid()
        plt.legend()
        plt.show()

    def plot_result(self, dataset, predictions, scaler, dates, save_file, intraday_df, x_ticker=None):
        # 作業用のnumpy配列を用意する
        temp_column = np.zeros(dataset.shape[1] - 1)

        # 予測データは正規化されているので、元の株価に戻す
        predictions = scaler.inverse_transform(self.padding_array(predictions, temp_column))

        #推論値
        prediction = predictions[ : , 0]
        print("AIが予測する次のUSDJPYの終値は以下です。")
        print(round(float(prediction[-1]), 2))

        # 訓練期間と検証期間を抽出する
        dates_train = dates[:self.mlst['USED_FOR_TRAINING']]
        dates_valid = dates[self.mlst['USED_FOR_TRAINING']:]

        # 終値を抽出
        closing_price = intraday_df.iloc[:, 4]

        # グラフを表示する領域をfigとする
        fig = plt.figure(figsize=(self.mlst['FONTSIZE'], self.mlst['FONTSIZE'] / 2))

        ax1 = plt.subplot()

        ax1.set_title('終値の履歴と予測結果 {} - {}'.format(dates[0], dates[self.mlst['H_HIDDEN'] - 1]), fontsize = self.mlst['FONTSIZE'] + 4)
        ax1.set_xlabel('日付', fontsize = self.mlst['FONTSIZE'], color='coral')
        ax1.set_ylabel('終値', fontsize = self.mlst['FONTSIZE'], color='blue')

        ax1.plot(dates, closing_price)
        ax1.plot(dates_valid, predictions[ : , 0])

        plt.xticks(rotation=60)

        ax1.legend(['実際の価格', '予測の価格'], loc='lower right')
        ax1.grid()

        if x_ticker == 'hour':
            data_0_mdates = mdates.HourLocator(byhour=range(0, 24, 4))
        elif x_ticker == 'day':
            data_0_mdates = mdates.DayLocator([1, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 31])
        elif x_ticker == 'month':
            data_0_mdates = mdates.MonthLocator([1, 3, 6, 9, 12])
        elif x_ticker == 'year':
            data_0_mdates = mdates.YearLocator(1, month=3, day=1)
        else:
            data_0_mdates = mdates.AutoDateLocator()

        data_0_mdates_fmt = mdates.DateFormatter('%Y-%m-%d')
        ax1.xaxis.set_major_locator(data_0_mdates)
        ax1.xaxis.set_major_formatter(data_0_mdates_fmt)

        fig.savefig(save_file)

        plt.show()

    def simple_graph(self, df_u):
        ticks = 1
        xticks = ticks * 5

        plt.plot(df_u['日付'][::ticks], df_u['終値'][::ticks], label='usd/jpy')
        plt.grid()
        plt.legend()
        plt.xticks(df_u['日付'][::xticks], rotation=60)
        plt.show()

    def padding_array(self, val, temp_column):
        """
        正規化する前にdatasetと同じnumpy形式に変換する
        """
        xset = []
        for x in val:
            a = np.insert(temp_column, 0, x)
            xset.append(a)

        xset = np.array(xset)
        return xset

    def create_model_for_prediction(self, model_body, full_file_path = '/../dollar_yen/for_prediction/created_model.py'):
        mode_head = 'from tensorflow.keras.layers import Activation, Dense' + "\n"
        mode_head + 'from tensorflow.keras.models import Sequential' + "\n"
        mode_head + "\n"
        model_build = str('def') + ' build_model():' + "\n"
        model_string = ('   model = Sequential()' + "\n" + '    ')
        model_string_re = ('   return model')

        file_path = os.path.dirname(os.path.abspath(__file__)) + full_file_path
        if(os.path.isfile(file_path) == True):
            os.remove(file_path)

        with open(file_path, 'a+') as f:
            f.write(mode_head)
            f.write(model_build)
            f.write(model_body)
            f.write(model_string_re)
            f.close()
