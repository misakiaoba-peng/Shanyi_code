#-*-coding:utf-8-*-
# @title: 国信资金强弱指标（GSMS）
# @author: Brian Shan
# @date: 2020.08.06
# 注： 交易时间为9点30 - 11点30， 1点 - 3点

import os
import csv
import numpy as np
np.seterr(divide = 'ignore', invalid='ignore')
import pandas as pd
from datetime import datetime, timedelta
from matplotlib import pyplot as plt
from typing import List
import itertools
from pandas.tseries.offsets import BDay


from ETF_MS import ETF_MS
from Dolphindb_Data import GetData
from Constant import future_multiplier, margin_percent, margin_multiplier, output_path, log_path
from Constant import output_path, log_path, select_ls, future_ls, whole_ls
from Constant import ETF_dict, commission_index, commission_bond, commission_multiplier

import time
from multiprocessing import Pool, freeze_support, Lock, get_logger,Queue
import logging
from logging.handlers import QueueHandler, QueueListener

class ETF_GSMS(object):
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
		self.start = start
		self.end = end
		self.get = GetData()
		self.ms = ETF_MS(stock_syms, start, end)

	def Backtest(self, **args):
		"""
		跑策略的回测

		args:
			**args： dict, 储存strategy会要用到的参数

		return：
			None
		"""
		self.strategy(**args)
		# self.generate_lots()
		# self.performance()

	def strategy(self, **args):
		self.period = args['period']
		res = self.ms.get_MS()
		self.gsms = res.rolling(window = self.period).apply(lambda x: x.sum(axis = 0)/x.std(axis = 0))

if __name__ == '__main__':
	from Constant import hs300
	c = ETF_GSMS(hs300, '2020.06.01', '2020.07.01')
	args = {}
	args['period'] = 25

	start_time = time.time()
	c.Backtest(**args)
	print(time.time() - start_time)
	c.gsms.to_csv('hs300_gsms.csv')
