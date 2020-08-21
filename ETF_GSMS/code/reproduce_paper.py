# %%
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from Constant import MS_file_path, output_path
from Dolphindb_Data import GetData
from matplotlib import pyplot as plt
# %%
MS = pd.read_csv(MS_file_path, index_col = 0, parse_dates = [0])
MS.fillna(value = 0, inplace = True)
GSMS = MS.rolling(window = 25).apply(lambda x: x.sum(axis = 0)/x.std(axis = 0, ddof = 0))
GSMS.dropna(inplace = True, axis = 0, how = 'all')
GSMS_rank = GSMS.rank(axis = 1)
zsyh_rank = GSMS_rank['600036']
# %%
get = GetData()
start = (zsyh_rank.index[0] + timedelta(days = -3)).strftime('%Y.%m.%d') 
end = zsyh_rank.index[-1].strftime('%Y.%m.%d')
zsyh_df = get.Stock_candle('600036', start, end, 'D')[['date', 'close']]
zsyh_df['date'] = zsyh_df['date'].dt.date
zsyh_df.set_index('date', inplace = True)
zsyh_ret = zsyh_df.rolling(window = 30).apply(lambda x: (x[-1] - x[0])/x[0]).shift(-29).iloc[:-29]

# %%
hs300_df = get.Stock_index_candle('000300', start, end, 'D')[['date', 'close']]
hs300_df['date'] = hs300_df['date'].dt.date
hs300_df.set_index('date', inplace = True)
hs300_ret = hs300_df.rolling(window = 30).apply(lambda x: (x[-1] - x[0])/x[0]).shift(-29).iloc[:-29]

# %%
ret_df = pd.DataFrame()
ret_df['600036'] = zsyh_ret['close']
ret_df['hs300'] = hs300_ret['close']
ret_df['excess_ret'] = ret_df['600036'] - ret_df['hs300']

# %%
fig, ax1 = plt.subplots()
fig.autofmt_xdate()
ax2 = ax1.twinx()
ax1.plot(zsyh_rank, 'g-')
ax2.plot(ret_df['excess_ret'],'b-')

ax1.set_xlabel('time')
ax1.set_ylabel('rank', color='g')
ax2.set_ylabel('excess ret', color='b')
plt.title(r"600036's 25-day GSMS ranks and 30-day excess return over HS300")
plt.savefig("600036.png")
plt.show()


# %%
with pd.ExcelWriter('result.xlsx') as writer_excel:
    MS.to_excel(writer_excel, sheet_name = 'MS')
    GSMS.to_excel(writer_excel, sheet_name = 'GSMS')
    GSMS_rank.to_excel(writer_excel, sheet_name = 'GSMS_rank')
    ret_df.to_excel(writer_excel, sheet_name = 'ret')

# %%
