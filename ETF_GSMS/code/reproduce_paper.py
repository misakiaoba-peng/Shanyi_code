# %%
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from Constant import MS_file_path, output_path
from Dolphindb_Data import GetData
from matplotlib import pyplot as plt
# %%
MS = pd.read_csv(MS_file_path, index_col = 0, parse_dates = [0])
GSMS = MS.rolling(window = 25).apply(lambda x: x.sum(axis = 0)/x.std(axis = 0, ddof = 0))
GSMS.dropna(inplace = True, axis = 0, how = 'all')
GSMS_rank = GSMS.rank(axis = 1)
zsyh_rank = GSMS_rank['600036']
get = GetData()
start = (zsyh_rank.index[0] + timedelta(days = -3)).strftime('%Y.%m.%d') 
end = zsyh_rank.index[-1].strftime('%Y.%m.%d')
zsyh_df = get.Stock_candle('600036', start, end, 'D')[['date', 'close']]
zsyh_df['date'] = zsyh_df['date'].dt.date
zsyh_df.set_index('date', inplace = True)
zsyh_df['ret'] = zsyh_df['close'].pct_change()
hs300_df = get.Stock_index_candle('000300', start, end, 'D')[['date', 'close']]
hs300_df['date'] = hs300_df['date'].dt.date
hs300_df.set_index('date', inplace = True)
hs300_df['ret'] = hs300_df['close'].pct_change()
excess_ret = (zsyh_df['ret'] - hs300_df['ret']).loc[zsyh_rank.index]

# %%
fig, ax1 = plt.subplots()

ax2 = ax1.twinx()
ax1.plot(zsyh_rank, 'g-')
ax2.plot(excess_ret,'b-')

ax1.set_xlabel('time')
ax1.set_ylabel('rank', color='g')
ax2.set_ylabel('excess ret', color='b')

plt.show()


# %%
