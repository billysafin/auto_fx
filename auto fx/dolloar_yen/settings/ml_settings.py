from tensorflow.keras.models import Sequential

class ml_settings():
    def __init__(self):
        self.MACHINE_SETTINGS = {
            'PERCENT_FOR_TRAINING': .8, # 訓練の割合
            'USED_FOR_TRAINING': 0,     # 訓練数　
            'PERCENT_FOR_H_HIDDEN': .8, # 隠れ層の割合
            'H_HIDDEN': 0,              # 隠れ層の数
            'MODEL': Sequential(),      # モデル
            'NURO_01': 100,              # ニューラルネット01
            'NURO_02': 50,              # ニューラルネット02
            'UNITS_01': 4,              # 第1層の出力数
            'UNITS_02': 4,              # 第2層の出力数
            'UNITS_03': 8,             # 第3層の出力数
            'DROPOUT_01': 0.1,          # Dropoutレート01
            'DROPOUT_02': 0.1,          # Dropoutレート02
            'BATCH_SIZE': 100,          # バッチサイズ
            'EPOCHS': 400,              # 訓練の回数
            'VALIDATION_SPLIT': 0.2,    #validation_split
            'FONTSIZE': 12,             #フォントサイズ
            'lr': 0.001
        }

    def get_MACHINE_SETTINGS(self):
        return self.MACHINE_SETTINGS
