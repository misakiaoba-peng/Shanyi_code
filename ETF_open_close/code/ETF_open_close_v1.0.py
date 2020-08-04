# %%
import os
import csv
import numpy as np
np.seterr(divide = 'ignore', invalid='ignore')
import pandas as pd
from datetime import datetime, timedelta
from matplotlib import pyplot as plt
from typing import Set
from typing import List

from Dolphindb_Data import GetData


# %%
stock = '510050'
c = GetData()
data = c.Stock_candle(stock, '2012.01.01', '2020.07.01', 'D')
data.set_index('date', inplace = True)
# %%
data['execute'] = data['close'] * (data['low'] >= data['open'] * 0.99) 
data['execute'] += (data['open'] * 0.99 * 1000).astype('int')/1000 * (data['low'] < data['open'] * 0.99)

data['ret'] = (data['execute'] - data['open'])/data['open']
data['cum_ret'] = data['ret'].cumsum()
# %%
data['cum_ret'].plot(title = r'510050 open-buy-close-sell Cumulative Return')
plt.savefig('华夏上证50早买晚卖累计收益率.png')

# %%
VictoryRatio = np.sum(data['ret'] > 0)/(np.sum(data['ret'] > 0) + np.sum(data['ret'] < 0)) # 胜率
		
data['nav'] = 1 + data['cum_ret']
profit_loss_ratio = -data[data['ret']>0]['ret'].mean()/data[data['ret']<0]['ret'].mean()  # 盈亏比（日）
# 最大日收益
daily_max = data['ret'].max()
# 最大日亏损率
daily_min = data['ret'].min()
rety = (data['nav'][-1] - 1) * (252 / data.shape[0]) # 年化收益率
sharp = rety / (data['ret'].std() * np.sqrt(252))
MDD = max(1 - data['nav'] / data['nav'].cummax())
if MDD != 0:
    MAR = rety / MDD
else:
    MAR = 0
result = {
    '累计收益率': data['nav'][-1] - 1, 
    'Sharpe': sharp, 
    '年化收益': rety,
    '胜率': VictoryRatio,
    '盈亏比': profit_loss_ratio,
    '最大日收益率': daily_max, 
    '最大日亏损率': daily_min,
    '最大回撤': MDD, 
    'MAR': MAR,
    }
result_df = pd.DataFrame.from_dict(result, orient='index').T


# %%
with pd.ExcelWriter(f"{stock}_open_close.xlsx") as writer_excel:
    data.to_excel(writer_excel, sheet_name = 'data')
    result_df.to_excel(writer_excel, sheet_name = 'result')

# %%
