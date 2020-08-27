# %%
import numpy as np
import pandas as pd
from Dolphindb_Data import GetData
from matplotlib import pyplot as plt
import datetime

data = GetData().Future_hist_Mcandle("T", "2016.01.01", "2020.08.25", type = "main", ktype = "1")[['date','close']]
data.replace(to_replace = [0, np.nan], method = 'ffill', inplace = True)
data.set_index('date', inplace = True)

# %%
StochLen = 9
M1 = 3
M2 = 3
money = 10_000_000
margin_percent = 0.02
margin_multiplier = 3
future_multiplier = 10000

commission = 3

vol_ratio = 0.05
# %%
def KDJ(data, StochLen, Sm1, Sm2):
	var0 = data.rolling(StochLen).min()
	var1 = data.rolling(StochLen).max()
	var2 = data - var0
	var3 = var1 - var0
	FastKCustom = var2/var3 * 100
	idx = np.where(var3 == 0)
	for i,j in zip(idx[0], idx[1]):
		if np.isnan(FastKCustom.iat[i-1, j]):
			FastKCustom.iat[i, j] = 50
		else:
			FastKCustom.iat[i, j] = FastKCustom.iat[i-1, j]
	Kvalue = FastKCustom.ewm(com = Sm1 - 1, adjust = False).mean()
	Dvalue = Kvalue.ewm(com = Sm2 - 1, adjust = False).mean()
	Jvalue = 3 * Kvalue - 2 * Dvalue
	return FastKCustom, Kvalue, Dvalue, Jvalue

# %%
RSV, K, D, J= KDJ(data[['close']], StochLen, M1, M2)
data['RSV'] = RSV
data['Kvalue'] = K
data['Dvalue'] = D
data['Jvalue'] = J
for g_key, g_df in data.groupby(data.index.date):
	if g_key == datetime.date(2017,4,24):
		print(5)
	g_df['pre_Jvalue'] =  g_df['Jvalue'].shift()
	g_df['prepre_J'] = g_df['Jvalue'].shift(2)
	g_df['J_cumMax'] = g_df['Jvalue'].cummax()
	local_max_bool = (g_df['pre_Jvalue'] >= g_df['prepre_J']) & (g_df['Jvalue'] < g_df['pre_Jvalue'] * (1 - vol_ratio))
	local_min_bool = (g_df['pre_Jvalue'] <= g_df['prepre_J']) & (g_df['Jvalue'] > g_df['pre_Jvalue'] * (1 + vol_ratio))
	local_min_val = g_df['Jvalue'].copy()
	local_min_val[~local_min_bool] = np.nan
	local_min_val.fillna(method = 'ffill', inplace = True)
	buy_signal = local_max_bool & (g_df['pre_Jvalue'] < g_df['J_cumMax']) & (g_df['pre_Jvalue'] > local_min_val * (1+vol_ratio))
	sell_signal = local_min_bool
	sell_signal.iloc[-1] = True
	g_df['buy'] = buy_signal.astype(int) * (money // (g_df['close'] * future_multiplier * margin_percent * margin_multiplier))
	g_df['sell_signal'] = sell_signal
	cur_hand = 0
	g_df['trade'] = 0
	for idx, row in g_df.iterrows():
		if row['sell_signal'] and cur_hand < 0:
			g_df.loc[idx, 'trade'] = -cur_hand
			cur_hand = 0
		elif row['buy'] != 0 and cur_hand == 0:
			cur_hand += row['buy']
			g_df.loc[idx, 'trade'] = row['buy']
	if g_df.loc[g_df.index[-1], 'trade'] < 0: # 盘尾不买
		g_df.loc[g_df.index[-1], 'trade'] = 0
	data.loc[g_df.index, 'trade'] = g_df['trade']
		
data['lot'] = data['trade'].cumsum()
# %%
temp =  data['trade'].copy()
temp[temp > 0] = 0
data['commission'] = -temp * commission

data['pnl'] = data['lot'].shift() * (data['close'] - data['close'].shift()) * future_multiplier  - data['commission']
# %%
data['total asset'] = data['pnl'].cumsum() + money
data['ret'] = data['pnl'] / money
plt.figure(figsize = (16,12))
plt.plot(data['total asset'])
plt.ylabel('Yuan')
plt.title(f"Total Asset Time Series Graph during {data.index[0].strftime('%y%m%d')}-{data.index[-1].strftime('%y%m%d')}")
plt.savefig("Total_asset.png")
plt.close()

# %%
data.to_csv('result.csv')
# %%
