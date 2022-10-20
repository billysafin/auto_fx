import math
import pandas_datareader as web
import datetime

import requests
import pandas as pd
import datetime
import os
import sys
import numpy as np

## for testing
from pprint import pprint

## 設定ファイル読み込み
from settings import alphavantage as alp
from common import support

## 機械学習をインポート
sys.path.append('../')
from machine_learning import machine_learning as ml

def pandas_data(df, fx):
    column = []
    column.append(df) #日時
    column.append(float(fx[df]['1. open']))#始値
    column.append(float(fx[df]['2. high'])) #高値
    column.append(float(fx[df]['3. low'])) #安値
    column.append(float(fx[df]['4. close'])) #終値

    return column

def get_data(from_date):
    """ APIからデータ取得するクラス
    from_data: データを取得開始する日付
    """

    alps_inst = alp.alphavantage()
    alp_fx_data = alps_inst.get_FX_DAILY()

    # alp URL
    intradayParams = ''
    for k, v in alp_fx_data.items():
        intradayParams = intradayParams + k + '=' + v + '&'
    intradayUrl = alps_inst.get_url() + intradayParams[:-1]
    intradayR = requests.get(intradayUrl)
    intradayData = intradayR.json()
    intradayData = intradayData['Time Series FX (' + alp_fx_data['interval'] + ')']

    # 本日の日付
    dt_now = datetime.datetime.now()
    this_year = dt_now.strftime('%Y-%m') + '-01'
    this_year = datetime.datetime.strptime(this_year, '%Y-%m-%d')

    # 取得開始する日付を変換
    if from_date is not None and isinstance(from_date, str) == True:
        from_date = datetime.datetime.strptime(from_date, '%Y-%m-%d')

    # 入れ物
    intradayPandas = []

    # データ整形
    for i,df in enumerate(intradayData):
        fx = intradayData
        df_date = datetime.datetime.strptime(df, '%Y-%m-%d')

        if from_date is not None:
            if df_date >= from_date:
                intradayPandas.append(pandas_data(df, fx))
        else:
            intradayPandas.append(pandas_data(df, fx))

    intraday_df = pd.DataFrame(intradayPandas)
    intraday_df = intraday_df.set_axis(['日付', '始値', '高値', '安値', '終値'], axis=1)

    # データを返す
    return intraday_df

def print_summaries(mlinstance, x_train, y_train, predictions, y_test):
    """
    モデルとデータの状態をプリントする
    """
    model_summary, scores = mlinstance.get_model_summary(predictions, y_test)
    print(model_summary)
    print('x_train shape : ', x_train.shape)
    print('y_train shape : ', y_train.shape)
    print(scores)

def save_file_path():
    """
    保存先ファイル名
    """
    t_delta = datetime.timedelta(hours=9)
    JST = datetime.timezone(t_delta, 'JST')
    now = datetime.datetime.now(JST)
    now_date = now.date().strftime('%Y%m%d')
    dir = os.path.dirname(os.path.abspath(__file__)) + '/graphs/predictions/'
    #file = dir + now_date + '.png'
    file = dir + 'prediction.png'

    return file

def main(from_date, model_name = "RNN", display_loses = False, save_file = True, simple_only = False, x_ticker=None):
    """ メインのクラス
    refrence: https://wayama.io/article/ml/lec/text/lstm_fx/
    """

    # 整形済みデータと日付リスト
    intraday_df = get_data(from_date)
    intraday_df.sort_values(by=['日付'], ascending=True, inplace=True)

    # 抽出したFXデータをdatasetに代入する
    dates = intraday_df.iloc[:, 0]
    datesset = [datetime.datetime.strptime(s, '%Y-%m-%d') for s in dates]
    data = intraday_df.filter(['始値', '高値', '安値', '終値'])
    dataset = data.values

    # 機械学習インスタンス
    mlinstance = ml.machine_learning()

    #取得データグラフのみ
    if support.strtobool(simple_only) == 1:
        pprint(intraday_df)
        mlinstance.simple_graph(intraday_df)
        exit()

    # 訓練データを作成する
    train_data = mlinstance.create_train_data(dataset)
    x_train, y_train = mlinstance.create_lstm_data(train_data)

    # モデル構築
    mlinstance.build_model(x_train, y_train)

    # データを用いて訓練
    hdf5_path = os.path.dirname(os.path.abspath(__file__)) + '/for_prediction/hdf5s/'
    history = mlinstance.train_data(x_train, y_train, hdf5_path)

    #学習中の評価値の推移
    print('学習結果をグラフで出力')
    traing_path = os.path.dirname(os.path.abspath(__file__)) + '/graphs/training/'
    mlinstance.graph_training(history, traing_path)

    # 検証データを用意する
    test_data, scaler = mlinstance.create_test_data(dataset, train_data)

    if support.strtobool(display_loses) == 1:
        mlinstance.get_loses(history)

    #予想結果
    predictions, y_test = mlinstance.make_prediction(test_data)

    # サマリーを出力
    print('サマリーを出力')
    print_summaries(mlinstance, x_train, y_train, predictions, y_test)

    #保存先とファイル名
    if support.strtobool(save_file) == 1:
        file = save_file_path()
    else:
        file = None

    #グラフの作成
    print('グラフを出力')
    mlinstance.plot_result(dataset, predictions, scaler, datesset, file, intraday_df, x_ticker)

    print('グラフ作成終了')

if __name__ == '__main__':
    """
    args[1]:　日付
    args[2]:　モデル種類
    args[3]:　損失関数の可視化
    args[4]:　ファイル保存
    """
    args = sys.argv

    args_len = len(args)

    if args_len >= 2:
        from_date = args[1]
    else:
        from_date = None

    if args_len >= 3:
        model = args[2]
    else:
        model = "LSTM"

    if args_len >= 4:
        display_lose = args[3]
    else:
        display_lose = False

    if args_len >= 5:
        save_file = args[4]
    else:
        save_file = True

    if args_len >= 6:
        simple_only = args[5]
    else:
        simple_only = False

    if args_len >= 7:
        x_ticker = args[6]
    else:
        x_ticker = None

    main(from_date, model, display_lose, save_file, simple_only, x_ticker)
