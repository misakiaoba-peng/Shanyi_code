#-*-coding:utf-8-*-
# @title: ETF 无量尺多空策略
# @author: Brian Shan
# @date: 2020.07.08
# 注： 交易时间为9点30 - 11点30， 1点 - 3点
# ToDo: 
# 1. 目前是从dolphindb_data里取一分钟的数据重新组合成其他分钟
#    原因是其他分钟的数据目前有问题，等待技术部门更新（也许他们不更新了）
# 2. 如果参数组数较小可使用并行跑，不然会out of memory

import os
import csv
import numpy as np
np.seterr(divide = 'ignore', invalid='ignore')
import pandas as pd
from datetime import datetime, timedelta
from matplotlib import pyplot as plt
from typing import Set
from typing import List
import gc
import shutil
import glob
import itertools
from guppy import hpy

from Dolphindb_Data import GetData
from Generate_Min_Data import generate_any_minute_data
from Constant import future_multiplier, margin_percent, margin_multiplier, output_path, log_path
from Constant import output_path, log_path, select_ls, future_ls, whole_ls, money
from Constant import ETF_dict, commission_index, commission_bond, commission_multiplier

import time
from multiprocessing import Pool, freeze_support, Lock, get_logger,Queue
import logging
from logging.handlers import QueueHandler, QueueListener
 
class ETF_KDJ_LongShort(object):
	def __init__(self, ETF_sym:List[str], future_sym: List[str], 
				start:str, end:str, cycle:int = 1):
		"""
		Constuctor: 

		args:
			ETF_sym: list of str, ETF代码列表
			future_sym: list of str, 期货代码列表
			start: str, 开始日期，格式为'yyyy.mm.dd' e.g. '2020.07.06'
			end: str，结束日期， 格式为'yyyy.mm.dd' e.g. '2020.07.06', 
					dataEnd必须要比dataStart晚
			circle: int,  策略所需要的分钟数
		"""
		self.ETF_sym = ETF_sym
		self.future_sym = future_sym
		self.num_ETF = len(ETF_sym)
		self.num_future = len(future_sym)
		self.start = (pd.to_datetime(start) + timedelta(days = -180)).strftime("%Y.%m.%d") # 预读半年数据
		self.start_backtest = pd.to_datetime(start)
		self.end = end
		self.cycle: int = cycle


		
		self.get = GetData()
		self.close_df = pd.DataFrame()
		self.get.clearcache()
		self.load_data()
		

	def load_data(self):
		# cur_dir = os.path.dirname(os.path.realpath(__file__))
		# if f"data_{self.cycle}min.csv" in os.listdir(cur_dir):
		#     # read from csv from tinysoft
		#     data = pd.read_csv(cur_dir + "\\data_15min.csv", index_col = 0)
		#     data.index = pd.to_datetime(data.index)
		#     data.columns = [str(i) for i in data.columns]
		#     data = data.loc[(pd.to_datetime(self.start) <= data.index) & (data.index <= pd.to_datetime(self.end) + timedelta(days = 1))]
		#     self.close_df = data[self.ETF_sym]
		# else:

		# read from dolphin_db
		for sym in self.ETF_sym:
			data=self.get.Stock_candle(sym, self.start, self.end, 1)
			if len(data) > 0:
				self.etf_handler(generate_any_minute_data(data, self.cycle))
			else:
				if self.close_df is None:
					self.close_df = pd.DataFrame(columns = [sym])
				else:
					self.close_df[sym] = np.nan
		
		self.close_buy_df = self.close_df.copy() # close price for buying ETF and future
		self.close_sell_df = self.close_df.copy() # close price for selling ETF and future
		self.switch_dict = {}
		for prod in self.future_sym:
			switch_df = self.get.Future_hist_Mswitch(prod, self.start, self.end)
			self.switch_dict[prod] = switch_df
			dateStart = self.start
			dateStart_buy = None
			dateStart_sell = None
			if len(switch_df) > 0: # 主力合约换过
				for row in switch_df.values:
					premain = row[4]
					pre_tradingday = row[9]
					data = self.get.Future_hist_candle(premain, dateStart, pre_tradingday.strftime("%Y.%m.%d"), 1)
					if len(data) > 0:
						dateStart_buy, dateStart_sell = self.future_handler(
							generate_any_minute_data(data, self.cycle), 
							dateStart_buy, 
							dateStart_sell
							)
					dateStart = (pre_tradingday - timedelta(days = 3)).strftime("%Y.%m.%d")
				
				data = self.get.Future_hist_candle(
						switch_df['main'].iloc[-1], 
						switch_df['pre_tradingday'].iloc[-1].strftime("%Y.%m.%d"), 
						self.end, 1
					)
				if len(data) > 0:
					dateStart_buy, dateStart_sell = self.future_handler(
						generate_any_minute_data(data, self.cycle), 
						dateStart_buy, 
						dateStart_sell
						)
			else: # 主力合约没有换过
				data = self.get.Future_hist_Mcandle(prod, self.start, self.end)
				if len(data) > 0:
					dateStart_buy, dateStart_sell = self.future_handler(
						generate_any_minute_data(data, self.cycle), 
						dateStart_buy, 
						dateStart_sell
						)
		for prod in self.future_sym:
			idx_nan = np.where(self.close_sell_df[prod].isnull())[0]
			if len(idx_nan) > 0 and len(idx_nan) != idx_nan[-1] + 1:
				logging.warning(f"there are missing close price in future: {prod} between {self.start}-{self.end}")
		
		# deal with missing data
		self.close_buy_df.fillna(method = 'ffill', inplace = True)
		self.close_sell_df.fillna(method = 'ffill', inplace = True)

		# basic matrix of cost and num per hand
		temp_df = self.close_buy_df[self.close_buy_df.index >= self.start_backtest]
		self.cost_perhand_mat = np.full(temp_df.shape, 100)
		if self.num_future > 0:
			self.cost_perhand_mat[:, -self.num_future:] = np.array(
					[margin_percent[prod] * margin_multiplier * future_multiplier[prod] for prod in self.future_sym]
				)
		self.cost_perhand_mat = self.cost_perhand_mat * temp_df.fillna(0).values
		
		self.num_perhand_mat = np.full(temp_df.shape, 100)
		if self.num_future > 0:
			self.num_perhand_mat[:, -self.num_future:] = np.array([future_multiplier[prod] for prod in self.future_sym])

		# commission matrix
		self.commission_pertrade = np.zeros(temp_df.shape)
		for i in range(self.num_ETF):
			self.commission_pertrade[:, i] = ETF_dict[self.ETF_sym[i]][1] * commission_multiplier
		for i in range(self.num_future):
			if self.future_sym[i] in ('IF', 'IC', 'IH'):
				self.commission_pertrade[:, self.num_ETF+i] = commission_index * commission_multiplier
				
		self.commission_perhand = np.zeros(temp_df.shape)
		for i in range(self.num_future):
			if self.future_sym[i] in ('TF', 'T'):
				self.commission_perhand[:, self.num_ETF+i] = commission_bond * commission_multiplier

	def etf_handler(self, data: pd.DataFrame):
		"""
		处理单个ETF数据
		"""
		sym = data['symbol'].iloc[0]
		data.set_index('date', inplace = True)
		data = data.rename(columns = {'close': sym})[[sym]]
		for g, g_df in data.groupby([data.index.hour, data.index.minute]):
			if not (g[0] == 10 or 13 <= g[0] <= 14 or \
				(g[0] == 9 and g[1] >= 31) or \
				(g[0] == 11 and g[1] <= 30) or \
				(g[0] == 15 and g[1] == 0)):
					data.drop(g_df.index, inplace = True)
		if self.close_df.empty:
			self.close_df = data.copy()
		else:
			self.close_df = self.close_df.join(data, how = "outer", sort = True)  
		idx_nan = np.where(self.close_df[sym].isnull())[0]
		if len(idx_nan) > 0 and len(idx_nan) != idx_nan[-1] + 1:
			logging.warning(f"there are missing close price in stock: {sym} between {self.start}-{self.end}")

	def future_handler(self, data: pd.DataFrame, dateStart_buy: datetime, dateStart_sell:datetime):
		"""
		处理单个期货品种数据
		"""
		prod = data['product'].iloc[0]
		data.set_index('date', inplace = True)
		data = data.rename(columns = {'close': prod})[[prod]]
		# 去掉非ETF交易时间的数据
		for g, g_df in data.groupby([data.index.hour, data.index.minute]):
			if not (g[0] == 10 or 13 <= g[0] <= 14 or \
				(g[0] == 9 and g[1] >= 31) or \
				(g[0] == 11 and g[1] <= 30) or \
				(g[0] == 15 and g[1] == 0)):
					data.drop(g_df.index, inplace = True)

		# attach data to the close price for buy and the close price for sell 
		if self.close_buy_df.empty or prod == self.close_buy_df.columns[0]: # future only
			if dateStart_buy != None:
				self.close_buy_df.at[dateStart_buy, prod] = data.at[dateStart_buy, prod]
			if dateStart_sell != None:
				dateStart_sell = data.index[data.index.get_loc(dateStart_sell)+1]
			# self.close_buy_df = pd.concat([self.close_buy_df, data.loc[dateStart_sell:data.index[-2], [prod]]])
			# self.close_sell_df = pd.concat([self.close_sell_df, data.loc[dateStart_sell:data.index[-1], [prod]]])
			self.close_buy_df = self.close_buy_df.append(data.loc[dateStart_sell:data.index[-2], [prod]])
			self.close_sell_df = self.close_sell_df.append(data.loc[dateStart_sell:data.index[-1], [prod]])
			self.close_buy_df.at[data.index[-1], prod] = np.nan
			dateStart_buy, dateStart_sell = data.index[-1], data.index[-1]
		else:
			self.close_buy_df.loc[dateStart_buy:data.index[-2], prod] = data.loc[dateStart_buy:data.index[-2], prod]
			self.close_sell_df.loc[dateStart_sell:data.index[-1], prod] = data.loc[dateStart_sell:data.index[-1], prod]

			dateStart_buy = data.index[-1]
			idx = self.close_sell_df.index.get_loc(data.index[-1])
			if idx + 1 < len(self.close_sell_df):
				idx += 1
			dateStart_sell = self.close_sell_df.index[idx]
		return dateStart_buy, dateStart_sell

	def Backtest(self, **args):
		"""
		跑策略的回测

		args:
			**args： dict, 储存strategy会要用到的参数

		return：
			None
		"""
		self.strategy(**args)
		self.generate_lots()
		self.performance()

	def strategy(self, **args):
		"""
		策略，生成每天的money ratio

		"""
		self.StochLen1 = args['StochLen1']
		self.StochLen2 = args['StochLen2']
		self.SmoothingLen1 = args['SmoothingLen1']
		self.SmoothingLen2 = args['SmoothingLen2']
		self.weight = args['weight']
		# self.risk_exposure = args['risk_exposure']
		
		def KDJ(data, StochLen, SmoothingLen):
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
			Kvalue = FastKCustom.ewm(span = SmoothingLen, adjust = False).mean()
			Dvalue = Kvalue.ewm(com = SmoothingLen - 1, adjust = False).mean()
			Jvalue = 3 * Kvalue - 2 * Dvalue
			return Kvalue, Dvalue, Jvalue
		
		_, _, self.Jvalue_s = KDJ(self.close_buy_df, self.StochLen1, self.SmoothingLen1)
		_, _, self.Jvalue_l = KDJ(self.close_buy_df, self.StochLen2, self.SmoothingLen2)
		self.indicator = self.weight * self.Jvalue_s + (1-self.weight) * self.Jvalue_l + 200 # 强弱指标
		self.indicator = self.indicator[self.indicator.index >= self.start_backtest]
		self.avg_indicator = self.indicator.mean(axis = 1) # 各品种强弱指标均值
		self.std_indicator = self.indicator.std(axis = 1, ddof = 0) # 各品种强弱指标标准差
		self.indUp = self.avg_indicator + 1.5 * self.std_indicator # 指标上轨
		self.indDn = self.avg_indicator - 1.5 * self.std_indicator # 指标下轨
		self.indicator.mask(self.indicator.gt(self.indUp, axis = 0), self.indUp, axis = 0, inplace = True)
		self.indicator.mask(self.indicator.lt(self.indDn, axis = 0), self.indDn, axis = 0, inplace = True)
		self.ReIndicator = self.indicator.divide(self.avg_indicator, axis = 0) - 1
		self.ReIndicator[self.ReIndicator[self.ETF_sym] < 0] = 0 # 不空ETF
		# self.ReIndicator[self.future_sym] = 0 # 有future但不做future
		self.MoneyRatio0 = self.ReIndicator.divide(self.ReIndicator.abs().sum(axis = 1), axis = 0)
		# # 修改投资比例
		# if self.num_ETF > 0 and self.risk_exposure != 0:
		# 	ETF_mat = self.MoneyRatio0[self.ETF_sym]
		# 	self.MoneyRatio0[self.ETF_sym] = ETF_mat / np.sum(ETF_mat, axis = 1)[:, None] * (1-self.risk_exposure)
		# if self.num_future > 0:
		# 	pos_ratio = self.risk_exposure/(1+2*self.risk_exposure)
		# 	neg_ratio = (1+self.risk_exposure)/(1+2*self.risk_exposure)
		# 	future_mat = self.MoneyRatio0[self.future_sym].values
		# 	future_mat_pos = future_mat * (future_mat > 0)
		# 	future_mat_neg = future_mat * (future_mat < 0)
		# 	self.MoneyRatio0[self.future_sym] = future_mat_pos / np.sum(future_mat_pos, axis = 1, keepdims=True) * pos_ratio + \
		# 										future_mat_neg / -np.sum(future_mat_neg, axis = 1, keepdims=True)* neg_ratio
		# 修改极值
		if self.num_ETF > 0:
			self.MoneyRatio0[self.MoneyRatio0 > 0.2] = 0.2
			self.MoneyRatio0[self.MoneyRatio0 < -0.2] = -0.2

	def generate_lots(self):
		"""
		生成每天的仓位
		"""
		MoneyRatio0_mat = self.MoneyRatio0.fillna(0).values
		close_buy_mat = self.close_buy_df[self.close_buy_df.index >= self.start_backtest].fillna(0).values
		close_sell_mat = self.close_sell_df[self.close_sell_df.index >= self.start_backtest].fillna(0).values
		
		hand_mat = MoneyRatio0_mat * money // self.cost_perhand_mat
		np.nan_to_num(hand_mat, copy = False)
		
		pnl = np.sum((close_sell_mat[1:, :] - close_buy_mat[:-1, :]) * hand_mat[:-1, :] * self.num_perhand_mat[:-1, :], axis = 1)

		trade_mat = np.concatenate((np.zeros((1,hand_mat.shape[1])), np.diff(hand_mat, axis = 0)))
		trade_mat_pos = trade_mat.copy()
		trade_mat_neg = trade_mat.copy()
		trade_mat_pos[trade_mat_pos < 0] = 0
		trade_mat_neg[trade_mat_neg > 0] = 0
		turnover_mat = (trade_mat_pos * close_buy_mat + (-trade_mat_neg) * close_sell_mat) * self.num_perhand_mat
		# 计算换仓导致的交易额
		for i in range(self.num_future):
			prod = self.future_sym[i]
			switch_df = self.switch_dict[prod]
			for row in switch_df.values:
				preclose_main = row[6]
				preclose_premain = row[7]
				pre_tradingday = row[9]
				pre_time = pre_tradingday + timedelta(hours=15)
				if pre_time not in self.MoneyRatio0.index:
					continue
				idx = self.MoneyRatio0.index.get_loc(pre_time)
				hold = hand_mat[idx - 1, self.num_ETF + i]
				hold2 = hand_mat[idx, self.num_ETF + i]
				if hold2 > hold: # 买
					trade_mat_pos[idx, self.num_ETF + i] += abs(hold) * 2
				else: # 卖
					trade_mat_pos[idx, self.num_ETF + i] += abs(hold2) * 2
				turnover_mat[idx, self.num_ETF + i] += abs(hold) * (preclose_premain + preclose_main) * future_multiplier[prod]

		commission_mat = turnover_mat * self.commission_pertrade + (trade_mat_pos - trade_mat_neg) * self.commission_perhand

		self.lots = pd.DataFrame(
			hand_mat,
			index = self.MoneyRatio0.index, 
			columns = [f"{i}-{ETF_dict[i][0]}" for i in self.ETF_sym] + self.future_sym
			)
		self.lots['commission'] = np.sum(commission_mat, axis = 1)
		self.lots['PnL'] = np.concatenate(([0], pnl)) - self.lots['commission'].values
		self.lots['total asset'] = self.lots['PnL'].cumsum() + money

		self.summary = pd.DataFrame(
			columns = self.lots.columns[:self.MoneyRatio0.shape[1]], 
			index = ['交易额', '最大手数', '手数中间值', '平均手数', '最小手数'])
		self.summary.loc['交易额', :] = np.sum(turnover_mat, axis = 0)
		self.summary.iloc[1:, :] = self.lots.iloc[:, :self.MoneyRatio0.shape[1]].agg(['max', 'median', 'mean', 'min'], axis = 0).values.tolist()
				
	def performance(self):
		"""
		生成报告
		"""
		self.lots['ret'] = self.lots['PnL'] / money
		
		outdir = os.path.join(output_path, f"{self.cycle}min", f"{self.start_backtest.strftime('%Y.%m.%d')}_{self.end}")
		if not os.path.exists(outdir):
			os.makedirs(outdir, exist_ok = True)

		plt.figure(figsize = (16,12))
		plt.plot(self.lots['total asset'])
		plt.ylabel('Yuan')
		plt.title(f"Total Asset Time Series Graph during {self.lots.index[0].strftime('%y%m%d')}-{self.lots.index[-1].strftime('%y%m%d')}")
		plt.savefig(os.path.join(outdir, 
			f"total_asset_{self.StochLen1}_{self.StochLen2}_{self.SmoothingLen1}_{self.SmoothingLen2}_{self.weight}" + \
				f"_{self.lots.index[0].strftime('%y%m%d')}_{self.lots.index[-1].strftime('%y%m%d')}.png"))
		plt.close()

		VictoryRatio = np.sum(self.lots['PnL'] > 0)/(np.sum(self.lots['PnL'] > 0) + np.sum(self.lots['PnL'] < 0)) # 胜率
		
		source = self.lots[['ret']].groupby(self.lots.index.date).sum()
		source['nav'] = source.ret.cumsum() + 1
		profit_loss_ratio = -source[source['ret']>0]['ret'].mean()/source[source['ret']<0]['ret'].mean()  # 盈亏比（日）
		# 最大日收益
		daily_max = source['ret'].max()
		# 最大日亏损率
		daily_min = source['ret'].min()
		rety = (source['nav'][-1] - 1) * (252 / source.shape[0]) # 年化收益率
		sharp = rety / (source['ret'].std() * np.sqrt(252))
		MDD = max(1 - source['nav'] / source['nav'].cummax())
		if MDD != 0:
			MAR = rety / MDD
		else:
			MAR = 0
		result = {
			'累计收益率': source['nav'][-1] - 1, 
			'Sharpe': sharp, 
			'年化收益': rety,
			'胜率': VictoryRatio,
			'盈亏比': profit_loss_ratio,
			'最大日收益率': daily_max, 
			'最大日亏损率': daily_min,
			'最大回撤': MDD, 
			'MAR': MAR,
			'累计手续费': self.lots['commission'].sum()
			}
		self.result = pd.DataFrame.from_dict(result, orient='index').T
		self.source = source
		# self.nav_permonth = source['nav'].resample('1M').last() / source[
		#     'nav'].resample('1M').first() - 1
		
	# seems useless in backtesting
	def clean_memory(self):
		del self.Jvalue_s
		del self.Jvalue_l
		del self.indicator
		del self.avg_indicator
		del self.std_indicator
		del self.indDn
		del self.indUp
		del self.ReIndicator
		del self.MoneyRatio0
		del self.lots
		del self.source
		del self.result
		del self.switch_dict
		gc.collect()

def select_params(nums):
	for d in os.listdir(output_path):
		c_dir = os.path.join(output_path, d)
		if not os.path.isdir(c_dir) or c_dir == output_path:
			continue
		df_ls = []
		des = os.path.join(c_dir, 'selected')
		os.makedirs(des, exist_ok = True)
		for date_dir, _, _ in os.walk(c_dir):
			if date_dir == c_dir or date_dir == des:
				continue
			for f in os.listdir(date_dir):
				if f[:7] == 'summary':
					data = pd.read_csv(os.path.join(date_dir, f))
					data = data[data['最大回撤'] < data['最大回撤'].quantile(0.25)].nlargest(nums, ['MAR', 'Sharpe'])
					df_ls.append(data)
					break
			# 拷贝图片
			# for row in data.values:
			# 	if row[5] == 0 or row[5] == 1:
			# 		file_name = os.path.join(date_dir, f"*{int(row[0])}_{int(row[1])}_{int(row[2])}_{int(row[3])}_{row[4]}_{int(row[5])}*")
			# 	else:
			# 		file_name = os.path.join(date_dir, f"*{int(row[0])}_{int(row[1])}_{int(row[2])}_{int(row[3])}_{row[4]}_{row[5]}*")
			# 	for file in glob.glob(file_name):
			# 		shutil.copy(file, des)

		df_total = pd.concat(df_ls)
		res = df_total.groupby(['StochLen1', 'StochLen2', 'SmoothingLen1', 'SmoothingLen2', 'weight']).size().reset_index(name = 'counts')
		res.sort_values('counts', ascending = False, inplace = True)
		res = res[res['counts'] == 3]
		res.to_csv(os.path.join(c_dir, 'para_summary.csv'), encoding='utf_8_sig', index = False)

def run_parameters(c, args, outdir, output_excel):
	start = c.start_backtest.strftime('%Y.%m.%d')
	end = c.end
	start_time = time.time()
	try:
		c.Backtest(**args)
	except Exception as e:
		logging.error(f"Error: {e} at args: {','.join([str(i) for i  in args.values()])} between {start}-{end}")
		return
	logging.info(f"Time used: {time.time() - start_time:.3f}s for args: " + \
					f"{','.join([str(i) for i  in args.values()])} between {start}-{end} using cycle {c.cycle}")
	# 导出数据
	lock.acquire()
	with open(os.path.join(outdir, f'summary_{start}_{end}.csv'), 'a', encoding='utf_8_sig', newline='') as csvfile:
		csv.writer(csvfile).writerow(list(args.values()) + list(c.result.values[0]))
	lock.release()

	if output_excel:
		with pd.ExcelWriter(os.path.join(
				outdir,
				f"KDJ_Arg_{'_'.join([str(i) for i  in args.values()])}_{start}_{end}.xlsx"
				)) as writer_excel:
			pd.DataFrame(args.items()).to_excel(writer_excel, sheet_name = "参数表")
			c.Jvalue_s.to_excel(writer_excel, sheet_name = '短期J值')
			c.Jvalue_l.to_excel(writer_excel, sheet_name = '长期J值')
			c.close_buy_df.to_excel(writer_excel, sheet_name = "买入价")
			c.close_sell_df.to_excel(writer_excel, sheet_name = "卖出价")
			c.indicator.to_excel(writer_excel, sheet_name = "Indicator")
			# c.avg_indicator.to_excel(writer_excel, sheet_name = "AvgIndicator")
			# c.ReIndicator.to_excel(writer_excel, sheet_name = "ReIndicator")
			c.MoneyRatio0.to_excel(writer_excel, sheet_name = 'MoneyRatio0')
			c.lots.to_excel(writer_excel, sheet_name = '仓位')
			c.source.to_excel(writer_excel, sheet_name = '利润表')
			c.result.to_excel(writer_excel, sheet_name = '指标')
			# c.nav_permonth.to_excel(writer_excel, sheet_name = '月度净值')
			c.summary.to_excel(writer_excel, sheet_name = '总结')

def init(l, q):
	global lock
	lock = l
	qh = QueueHandler(q)
	logger = logging.getLogger()
	logger.setLevel(logging.INFO)
	logger.addHandler(qh)

def run(start: str, end: str, arg_mat:list, cycle:"int > 0" = 15, ETF_ls: list = select_ls, \
		future_ls:list = future_ls, output_excel = False):
	"""
	主程序
	"""

	# 生成输出的文件夹
	outdir = os.path.join(output_path, f"{cycle}min", f"{start}_{end}")
	if not os.path.exists(outdir):
		os.makedirs(outdir, exist_ok = True)
	
	if not os.path.exists(log_path):
		os.makedirs(log_path, exist_ok = True)

	# logging setup
	q = Queue()
	logger = logging.getLogger()
	logger.setLevel(logging.INFO)
	ch = logging.StreamHandler()
	ch.setLevel(logging.INFO)
	fh = logging.handlers.RotatingFileHandler(
			os.path.join(log_path, f'log.txt'), maxBytes=10 * 1024 * 1024, backupCount=20)
	fh.setLevel(logging.INFO)
	formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
	fh.setFormatter(formatter)
	ch.setFormatter(formatter)
	ql = QueueListener(q, fh, ch)
	ql.start()
	logger.addHandler(fh)
	logger.addHandler(ch)

	c = ETF_KDJ_LongShort(ETF_ls, future_ls, start, end, cycle)
	logger.info(f"Load data {start}-{end} finished")

	with open(os.path.join(outdir, f'summary_{start}_{end}.csv'), 'a', encoding='utf_8_sig', newline='') as csvfile:
		writer_csv = csv.writer(csvfile)
		writer_csv.writerow(['StochLen1', 'StochLen2', 'SmoothingLen1', 'SmoothingLen2', 'weight', \
			'累计收益率', 'Sharpe', '年化收益', '胜率', '盈亏比', '最大日收益率', '最大日亏损率', '最大回撤', 'MAR', '累计手续费'])

	l = Lock()
	pool = Pool(maxtasksperchild = 400, initializer  = init, initargs = (l, q))
	for row in arg_mat:
		if row[1] < row[0] or (row[0] == row[1] and row[2] == row[3] and row[4] != 0.1): # hard code 0.1
			continue
		else:
			args  = {}
			args['StochLen1'] = int(row[0])
			args['StochLen2'] = int(row[1])
			args['SmoothingLen1'] = int(row[2])
			args['SmoothingLen2'] = int(row[3])
			args['weight'] = row[4]
			pool.apply_async(run_parameters, args = (c, args, outdir, output_excel))
	pool.close()
	pool.join()
	ql.stop()

if __name__ == '__main__':
	freeze_support() # prevent raising run-time error from running the frozen executable

	# 参数
	StochLen = [5, 9, 18, 25, 34, 46, 59, 72, 89]
	SmoothingLen = [3, 7, 11, 15, 19, 24]
	weight = [0.1, 0.3, 0.5, 0.7, 0.9]
	# risk_exposure = [0]
	# risk_exposure = [0, 0.2, 0.4, 0.6, 0.8, 1]

	arg_mat = list(itertools.product(StochLen, StochLen,SmoothingLen, SmoothingLen, weight))

	start_train = ['2016.01.01', '2017.01.01', '2018.01.01']
	# end_train = ['2016.02.01', '2017.02.01', '2018.02.01']
	end_train = ['2018.01.01', '2019.01.01', '2020.01.01']
	cycle = [15, 30, 60, 120, 240]

	
	# ---------------------------------------跑Train的参数------------------------------------------------------------------
	
	# for c in cycle:
	# 	for start, end in zip(start_train, end_train):
	# 		run(start, end, arg_mat, c, select_ls, [], False)

	
	# for c in cycle:
	# 	for start, end in zip(start_train, end_train):
	# 		run(start, end, arg_mat, c, [], future_ls, False)

	#---------------------------------------生成train里共同的优质参数组-------------------------------------------

	# select_params(150)

	# ---------------------------------------test------------------------------------------------------------------
	# 
	# for c in cycle:
	#     params = pd.read_csv(os.path.join(output_path, f"{c}min", 'para_summary.csv'))
	#     run('2020.01.01', '2020.07.01', params.values, c, select_ls, [], False)

	# ---------------------------------------删去test集里不好的参数组-------------------------------------------------------
	# 
	# for c in cycle:
	#     test_result = pd.read_csv(os.path.join(output_path, f"{c}min",'2020.01.01_2020.07.01', 'summary_2020.01.01_2020.07.01.csv'))
	#     final_params = test_result[test_result['MAR'] > 2].iloc[:,:5]
	#     if len(test_result) > 0:
	# 			print(f"Reserve Ratio: {len(final_params)}/{len(test_result)} = {len(final_params)/len(test_result)}")
	#     final_params.to_csv(os.path.join(output_path, f"{c}min", 'final_params.csv'), index = False)

	# ---------------------------------------生成全周期报告--------------------------------------------------------------
	#
	# cycle = [15, 240] 
	# for c in cycle:
	# 	params = pd.read_csv(os.path.join(output_path, f"{c}min", 'final_params.csv'))
	# 	run('2016.01.01', '2020.07.01', params.values, c, select_ls, [], True)
	
	
	# ---------------------------------------DEBUG----------------------------------------------------------------
	# run('2016.01.01', '2020.07.01', [[34, 46, 18, 13, 0.4, 0]], 15, select_ls, [], True)
	