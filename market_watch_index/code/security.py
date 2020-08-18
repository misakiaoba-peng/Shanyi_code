#-*-coding:utf-8-*-
# @title: 市场监控指数
# @author: Brian Shan
# @date: 2020.08.11

import numpy as np
import pandas as pd
import datetime
import logging
from datetime import datetime, timedelta
from Dolphindb_Data import GetData

class security(object):
    def __init__(self, security_code:str, lookback_days: 'int > 0' = -500):
        self.name  = security_code
        self.get = GetData()
        self.preLoad(lookback_days)
        self.sharpe = None
        self.momentum = None
        self.vol = None
        self.num_point = 200

        self.tick_time = None
        self.tick_high = None
        self.tick_low = None
        self.tick_close  = None
    
    def preLoad(self, lookback_days):
        today = datetime.now().date()
        data = self.get.Future_hist_Mcandle(
                    self.name, 
                    (today + timedelta(days = lookback_days)).strftime("%Y.%m.%d"),
                    (today + timedelta(days = -1)).strftime("%Y.%m.%d"))
        data.replace(to_replace = [0, np.nan], method = 'ffill', inplace = True)
        data.set_index('date', inplace = True)
        data.sort_index(inplace = True)
        self.time = list(data.index)
        self.high = data['high']
        self.low = data['low']
        self.close = data['close']
        
    def handler(self, tick):
        symbol = tick[0]
        assert symbol == self.name
        cur_time = pd.Timestamp.combine(pd.to_datetime(tick[3]).date(), pd.to_datetime(tick[4]).time())
        cur_high = tick[7]
        cur_low = tick[8]
        cur_last = tick[9]
        if self.tick_time is None or self.time[-1].minute == cur_time.minute:
            self.tick_time = cur_time
            self.tick_high = cur_high if self.tick_high is None else max(cur_high, self.tick_high)
            self.tick_low = cur_low if self.tick_low is None else min(cur_low, self.tick_low)
            self.tick_close = cur_last
        else: 
            temp = self.tick_time.ceil(freq = 'T')
            self.time.append(temp)
            self.high[temp] = self.tick_high
            self.low[temp] = self.tick_low
            self.close[temp] = self.tick_close

            self.time = self.time[1:]
            self.high = self.high.iloc[1:]
            self.low = self.low.iloc[1:]
            self.close = self.close.iloc[1:]

            self.tick_time = cur_time
            self.tick_high = cur_high
            self.tick_low = cur_low
            self.tick_close = cur_last

    def get_trend(self, cycle:str, N:int):
        if self.sharpe is None:
            self.trend_high = pd.Series(self.high, index = self.time).groupby(
                                pd.Grouper(freq = cycle)).max().dropna()[(-self.num_point-N):]
            self.trend_low = pd.Series(self.low, index = self.time).groupby(
                                pd.Grouper(freq = cycle)).min().dropna()[(-self.num_point-N):]
            self.trend_close = pd.Series(self.close, index = self.time).groupby(
                                pd.Grouper(freq = cycle)).agg('last').dropna()[(-self.num_point-N):]
            self.trend_TP = (self.trend_high + self.trend_low + self.trend_close)/3
            trend_std = self.trend_TP.rolling(window = N).std(ddof = 0)[(-self.num_point):]
            temp = (self.trend_close - self.trend_close.shift(N))[(-self.num_point):]
            self.sharpe = temp/trend_std
            self.momentum = temp/(self.trend_high + self.trend_low - 2 * self.trend_close).abs().rolling(window = N).sum()[(-self.num_point):]
            
            # keep length only N
            self.trend_high = self.trend_high[-N:]
            self.trend_low = self.trend_low[-N:]
            self.trend_TP = self.trend_TP[-N:]
            self.trend_close = self.trend_close[-N:]
        else:
            if cycle[-3:] == 'min': 
                num = int(cycle[:-3])
                new_high = self.high[-num:].max()
                new_low = self.low[-num:].min()
            elif cycle[-1:] == 'D':
                new_high = self.high[self.high.index.dt.date() == self.high.index[-1].date()].max()
                new_low = self.low[self.low.index.dt.date() == self.high.index[-1].date()].min()
            else: 
                raise AttributeError(f"cycle can only be in min or D")
            
            new_close = self.close.iloc[-1]
            new_TP = (new_high + new_low + new_close) / 3
            
            self.trend_high[self.time[-1]] = new_high
            self.trend_low[self.time[-1]] = new_low
            self.trend_close[self.time[-1]] = new_close
            self.trend_TP[self.time[-1]] = new_TP

            self.trend_high = self.trend_high[-N:]
            self.trend_low = self.trend_low[-N:]
            self.trend_TP = self.trend_TP[-N:]
            self.trend_close = self.trend_close[-N:]
            
            self.sharpe[self.time[-1]] = (self.trend_close.iloc[-1] - self.trend_close.iloc[0])/self.trend_TP.std(ddof = 0)
            self.momentum[self.time[-1]] = (self.trend_close.iloc[-1] - self.trend_close.iloc[0])/\
                (self.trend_high + self.trend_low - 2 * self.trend_close).abs().sum()
            self.sharpe = self.sharpe[-self.num_point:]
            self.momentum = self.momentum[-self.num_point:]
        return self.sharpe, self.momentum
                

    def get_vol_single(self, cycle, N):
        if self.vol is None:
            self.vol_high = pd.Series(self.high, index = self.time).groupby(
                        pd.Grouper(freq = cycle)).max().dropna()[(-self.num_point-N):]
            self.vol_low = pd.Series(self.low, index = self.time).groupby(
                        pd.Grouper(freq = cycle)).min().dropna()[(-self.num_point-N):]
            self.vol_close = pd.Series(self.close, index = self.time).groupby(
                        pd.Grouper(freq = cycle)).agg('last').dropna()[(-self.num_point-N):]
            self.vol_TP = (self.vol_high + self.vol_low + self.vol_close)/3
            self.vol = self.vol_TP.rolling(window = N).std(ddof = 0)[(-self.num_point):]
            
        else:
            if cycle[-3:] == 'min': 
                num = int(cycle[:-3])
                new_high = self.high[-num:].max()
                new_low = self.low[-num:].min()
            elif cycle[-1:] == 'D':
                new_high = self.high[self.high.index.dt.date() == self.high.index[-1].date()].max()
                new_low = self.low[self.low.index.dt.date() == self.high.index[-1].date()].min()
            else: 
                raise AttributeError(f"cycle can only be in min or D")
            
            new_close = self.close[-1]
            new_TP = (new_high + new_low + new_close) / 3
            
            self.vol_TP[self.time[-1]] = new_TP
            self.vol_TP = self.vol_TP[-N:]
            self.vol[self.time[-1]] = self.vol_TP.std(ddof = 0)
            self.vol = self.vol[-self.num_point:]

        return self.vol