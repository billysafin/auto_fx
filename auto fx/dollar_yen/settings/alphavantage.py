
class alphavantage():
    def __init__(self):
        self.url = 'https://www.alphavantage.co/query?'

        self.FX_INTRADAY_PARAMS =  {
            'function':         'FX_INTRADAY',
            'from_symbol':      'USD',
            'to_symbol':        'JPY',
            'interval':         '30min',
            'outputsize':       'full',
            'datatype':         'json',
            'apikey':           '8GCTR30AOLE6JTUD'
        }

        self.FX_DAILY = {
            'function':         'FX_DAILY',
            'from_symbol':      'USD',
            'to_symbol':        'JPY',
            'interval':         'Daily',
            'outputsize':       'full',
            'datatype':         'json',
            'apikey':           '8GCTR30AOLE6JTUD'
        }

    def get_url(self):
        return self.url

    def get_FX_INTRADAY_PARAMS(self):
        return self.FX_INTRADAY_PARAMS

    def get_FX_DAILY(self):
        return self.FX_DAILY
