from datetime import datetime, timedelta
from security import security
from typing import List
import numpy as np
import pandas as pd
import logging


def str_to_time(cycle:str) -> tuple:
    if cycle[-3:] == 'min': 
        day = 0
        hour = 0
        minute = int(cycle[:-3])
    elif cycle[-1:] == 'D':
        day = 1
        hour = 0
        minute = 0
    else:
        raise AttributeError(f"cycle can only be in min or D")
    return day, hour, minute



class market_watch_index(object):
    def __init__(self, 
                stk_ls: List[str], stk_idx_future_ls: List[str], cmdty_future_ls: List[str], 
                cycle_trend: str = '15min', N_trend:int = 200, 
                cycle_vol:str = 'D', N_vol:int = 60,
                cycle_whl:str = 'D', N_mlty:int = 60):
        self.stk_ls = stk_ls
        self.stk_idx_future_ls = stk_idx_future_ls
        self.cmdty_future_ls = cmdty_future_ls
        self.future_ls = stk_idx_future_ls + cmdty_future_ls
        self.sec_dict = {}
        self.trend_schedule = {}
        self.vol_schedule = {}
        for i in (stk_ls + stk_idx_future_ls + cmdty_future_ls):
            sec = security(i)
            sharpe, momentum = sec.get_trend(cycle_trend, N_trend)
            vol = sec.get_vol_single(cycle_vol, N_vol)
            self.sec_dict[i] = sec

        self.period_trend = str_to_time(cycle_trend)
        self.cycle_trend = cycle_trend
        self.N_trend = N_trend
        self.period_vol = str_to_time(cycle_vol)
        self.N_vol = N_vol
        self.cycle_vol = cycle_vol
        self.cycle_whl = cycle_whl
        self.period_whl = str_to_time(cycle_whl)
        self.N_mlty = N_mlty
        
    def handler(self, tick):
        cur_time = pd.Timestamp.combine(pd.to_datetime(tick[3]).date(), pd.to_datetime(tick[4]).time())
        product = tick[0].upper()
        if product in self.sec_dict.keys():
            self.sec_dict[product].handler(tick)
            if product in self.trend_schedule:
                if cur_time >= self.trend_schedule[product]:
                    sharpe, momentum = self.sec_dict[product].get_trend(self.cycle_trend, self.N_trend)
                    logging.debug(f"{product}-{sharpe.iloc[-2:]}")
                    logging.debug(f"{product}-{momentum.iloc[-2:]}")
                    self.trend_schedule[product] = cur_time + timedelta(days = self.period_trend[0], minutes = self.period_trend[2])
            else:
                self.trend_schedule[product] = cur_time + timedelta(days = self.period_trend[0], minutes = self.period_trend[2])
            if product in self.vol_schedule:
                if cur_time >= self.vol_schedule[product]:
                    vol = self.sec_dict[product].get_vol_single(self.cycle_vol, self.N_vol)
                    logging.debug(f"{product}-{vol.iloc[-2:]}")
                    self.vol_schedule[product] = cur_time + timedelta(days = self.period_vol[0], minutes = self.period_vol[2])
            else:
                self.vol_schedule[product] = cur_time + timedelta(days = self.period_vol[0], minutes = self.period_vol[2])

    
    def vol_index_whole(self, cycle, M):
        N = 60 * M
        rate = self.close_df.pct_change().iloc[-N:, :] 
        CO = rate.cov()
        weight = np.full(self.num_prod, 1/N)
        market_vol = weight.T @ CO @ weight


from Constant import stk_ls, stk_idx_future_ls, cmdty_future_ls
logging.basicConfig(level = logging.DEBUG, format = '%(asctime)s - %(levelname)s - %(message)s', 
		handlers = [logging.StreamHandler()])


array = market_watch_index(stk_ls, stk_idx_future_ls, cmdty_future_ls, cycle_trend = '1min', cycle_vol = '1min')
from threading import Event
import dolphindb as ddb
s=ddb.session()
s.enableStreaming(30001)
s.subscribe("10.0.40.33", 8505,array.handler,"Future_stream","action_x",-1,True)
# s.subscribe('10.0.60.56', 8503,array.handler,"rtq_stock_stream_quick","action_x",-1,True)
Event().wait()
