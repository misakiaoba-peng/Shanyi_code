#-*-coding:utf-8-*-
# @title: 资金强弱(MS)
# @author: Brian Shan
# @date: 2020.08.10

import os
import logging
import numpy as np
np.seterr(divide = 'ignore', invalid='ignore')
import pandas as pd
from typing import List
from datetime import time, timedelta, datetime
from Constant import MS_file_path
from pandas.tseries.offsets import BDay


import dolphindb as ddb
s=ddb.session()
s.connect('10.0.60.55',8509,'admin','123456')


class ETF_MS(object):
	def __init__(self, stock_syms:List[str], start:str, end:str):
		"""
		Constuctor: 

		args:
			stock: str, 股票代码
			start: str, 开始日期，格式为'yyyy.mm.dd' e.g. '2020.07.06'
			end: str，结束日期， 格式为'yyyy.mm.dd' e.g. '2020.07.06', 
					end必须要比start晚
		"""
		self.stock_syms = stock_syms
		self.start = pd.to_datetime(start)
		self.end = pd.to_datetime(end)

	def load_data(self, stock_ls:List[str], date:str, field_ls: List[str] = ['*']):
		sql = f"unionAll((select {','.join(field_ls)} from loadTable('dfs://STOCK_SZL2_TAQ', 'SZL2_TAQ') \
			where symbol in {stock_ls}, date = {date}, time >= 09:30:00, time <= 15:00:00 order by symbol, time),\
			(select {','.join(field_ls)} from loadTable('dfs://STOCK_SHL2_TAQ', 'SHL2_TAQ') \
			where symbol in {stock_ls}, date = {date}, time >= 09:30:00, time <= 15:00:00 order by symbol, time))"
		try:
			data = s.run(sql)
			return data
		except Exception as e:
			logging.ERROR(e)


	def handler(self, data):
		for g_key, g_df in data.groupby('symbol'):
			g_df.replace(to_replace = [0, np.nan], method = 'ffill', inplace = True)
			g_df.reset_index(drop = True, inplace = True)
			g_df.loc[:, 'curTurnover'] = g_df['turnover'].diff()
			g_df.loc[g_df.index[0], 'curTurnover'] = g_df['turnover'].iloc[0]
			
			self.preClose[g_key] = g_df['preClose']
			self.latest[g_key] = g_df['last']
			self.turnover[g_key] = g_df['curTurnover']
			self.askPrice1[g_key] = g_df['askPrice1']
			self.bidPrice1[g_key] = g_df['bidPrice1']

	def makeup_data(self, stock_ls:List[str], start = None, end = None, date_ls: List[datetime] = None):
		if start is None and end is None and date_ls is None:
			raise AttributeError(f"start & end and date_ls must be specified at least one")
		if date_ls is None:
			date_ls = pd.date_range(start, end, freq = BDay())
		res = pd.DataFrame(columns = stock_ls)
		fields_ls = ['symbol', 'time', 'preClose', 'last', 'turnover', 'askPrice1', 'bidPrice1']
		for d in date_ls:
			self.preClose = pd.DataFrame()
			self.limitUp = pd.DataFrame()
			self.limitDown = pd.DataFrame()
			self.latest = pd.DataFrame()
			self.turnover = pd.DataFrame()
			self.askPrice1 = pd.DataFrame()
			self.bidPrice1 = pd.DataFrame()
			logging.info(f"start loading")
			data = self.load_data(self.stock_syms, d.strftime("%Y.%m.%d"), fields_ls) 
			logging.info(f"{d} loading finished")
			if len(data) > 0:
				self.handler(data)
			self.limitUp = self.preClose * 1.1
			self.limitDown = self.preClose * 0.9
			if not self.limitUp.empty:
				res.loc[d,:] = self.MS()
		return res

	def get_MS(self):
		if os.path.isfile(MS_file_path):
			ms_df = pd.read_csv(MS_file_path, index_col = 0, parse_dates=[0])
			stock_miss = set(self.stock_syms) - set(ms_df.columns)
			if len(stock_miss) > 0: # 补足stock
				res = self.makeup_data(stock_miss, date_ls = ms_df.index)
				ms_df.loc[:, res.columns] = res
			# 补足日期
			if self.end.date() > ms_df.index[-1].date():
				res = self.makeup_data(ms_df.columns, ms_df.index[-1].date() + timedelta(days = 1), self.end)
				for idx, row in res.iterrows():
					ms_df.loc[idx, :] = row
			if self.start.date() < ms_df.index[0].date(): 
				res = self.makeup_data(ms_df.columns, self.start, ms_df.index[0].date() + timedelta(days = -1))
				for idx, row in res.iterrows():
					ms_df.loc[idx, :] = row
			ms_df.sort_index(inplace = True)
			ms_df.to_csv(MS_file_path)
			return ms_df[(ms_df.index >= self.start) & (ms_df.index <= self.end)].loc[:, self.stock_syms]
		else:
			MS_dir = os.path.dirname(MS_file_path)
			if not os.path.isdir(MS_dir):
				os.makedirs(MS_dir, exist_ok = True)
			ms_df = self.makeup_data(self.stock_syms, self.start, self.end)
			ms_df.to_csv(MS_file_path)
			return ms_df

	def MS(self):
		previous = self.latest.shift()
		previous.iloc[0, :] = self.preClose.iloc[0, :]
		# 1) 若两次报价不等则定义期间资金强弱等于资金流量
		res = ((self.latest > previous) * self.turnover).sum(axis = 0)
		res += ((self.latest < previous) * -self.turnover).sum(axis = 0)

		# 2) 若两次报价相等，且新报价非涨跌停（买1和卖1报价均非控），则新报价不低于卖1价时，记期间
		# 资金强弱为成交金额，若新报价不高于买1价时，记期间资金强弱为-1*成交金额，以上均不满足时记
		# 期间资金强弱为零。
		res += (((self.latest == previous) & ((self.latest > self.limitDown) | (self.latest < self.limitUp)) \
				& (self.latest >= self.askPrice1)) * self.turnover).sum(axis = 0)
		res += (((self.latest == previous) & ((self.latest > self.limitDown) | (self.latest < self.limitUp)) \
				& (self.latest <= self.bidPrice1)) * -self.turnover).sum(axis = 0)

		# 3) 若两次报价相等，且新报价为涨停或者跌停状态，则将期间成交金额记录至尾盘判断。
		# 若新报价为涨停切收盘为涨停时，记期间资金强弱为成交金额，否则记期间资金强弱为-1*成交金额，
		res += (((self.latest == previous) & (self.latest >= self.limitUp) \
					& (self.latest.iloc[-1] >= self.limitUp)) * self.turnover).sum(axis = 0)
		res += (((self.latest == previous) & (self.latest >= self.limitUp) \
					& (self.latest.iloc[-1] < self.limitUp)) * -self.turnover).sum(axis = 0)
		# 若新报价为跌停且收盘价为跌停时记期间资金强弱为-1 * 成交金额， 否则记期间资金强弱为成交金额
		res += (((self.latest == previous) & (self.latest <= self.limitDown) \
					& (self.latest.iloc[-1] <= self.limitDown)) * self.turnover).sum(axis = 0)
		res += (((self.latest == previous) & (self.latest <= self.limitDown) \
					& (self.latest.iloc[-1] > self.limitDown)) * -self.turnover).sum(axis = 0)
		return res


if __name__ == '__main__':
	logging.basicConfig(
			level = logging.DEBUG, 
			format = '%(asctime)s - %(levelname)s - %(message)s', 
			handlers = [logging.StreamHandler()])

	from Constant import hs300
	c = ETF_MS(select_ls, '2020.07.01', '2020.07.31')
	res = c.get_MS()
	print(res)