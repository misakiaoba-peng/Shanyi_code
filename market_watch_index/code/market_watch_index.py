#-*-coding:utf-8-*-
# @title: 市场监控指数
# @author: Brian Shan
# @date: 2020.08.07

from datetime import datetime, timedelta, time
from security import security
from typing import List
import numpy as np
import pandas as pd
import logging
from PyQt5.QtWidgets import  QApplication, QTableWidget, QTableWidgetItem
from PyQt5.QtCore import *
from ApplicationWindow import ApplicationWindow
from Constant import future_name_dict
from SubscribeData import SubscribeData

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
		self.total_ls = stk_ls + stk_idx_future_ls + cmdty_future_ls
		self.sec_dict = {}
		self.sec_idx = {}
		self.trend_schedule = {}
		self.vol_schedule = {}
		self.whl_schedule = None
		self.period_trend = str_to_time(cycle_trend)
		self.cycle_trend = cycle_trend
		self.N_trend = N_trend
		self.period_vol = str_to_time(cycle_vol)
		self.N_vol = N_vol
		self.cycle_vol = cycle_vol
		self.cycle_whl = cycle_whl
		self.period_whl = str_to_time(cycle_whl)
		self.N_mlty = N_mlty

		self.sharpe_dict = {}
		self.momentum_dict = {}
		self.vol_dict = {}
		self.vol_whl = None

		self.window = ApplicationWindow(
			len(self.total_ls) + 1, 
			4, 
			["全商品"] + [f"{future_name_dict[i]}({i})" for i in self.total_ls],
			['夏普比率', '动量效率', '波动率', '选择'],
			self)
		
		idx = 1
		for i in (self.total_ls):
			sec = security(i)
			sharpe, momentum = sec.get_trend(cycle_trend, N_trend)
			vol = sec.get_vol_single(cycle_vol, N_vol)
			self.sharpe_dict[i] = sharpe
			self.momentum_dict[i] = momentum
			self.vol_dict[i] = vol
			self.trend_schedule[i] = sharpe.index[-1] + timedelta(days = self.period_trend[0], minutes = self.period_trend[2])
			self.vol_schedule[i] = vol.index[-1] + timedelta(days = self.period_vol[0], minutes = self.period_vol[2])
			self.sec_dict[i] = sec
			self.sec_idx[i] = idx
			self.window.table.setItem(idx, 0, QTableWidgetItem(f"{sharpe.iloc[-1]:.3f}"))
			self.window.table.setItem(idx, 1, QTableWidgetItem(f"{momentum.iloc[-1]:.3f}"))
			self.window.table.setItem(idx, 2, QTableWidgetItem(f"{vol.iloc[-1]:.3f}")) 
			item = QTableWidgetItem("")
			item.setCheckState(Qt.Unchecked)
			self.window.table.setItem(idx, 3, item)
			idx += 1
		self.window.table.setItem(0, 0, QTableWidgetItem("-"))
		self.window.table.setItem(0, 1, QTableWidgetItem("-"))
		self.vol_whl = self.vol_index_whole(True)
		self.whl_schedule = self.vol_whl.index[-1] + timedelta(days = self.period_whl[0], minutes = self.period_whl[2])

		self.window.table.setItem(0, 2, QTableWidgetItem(f"{self.vol_whl.iloc[-1]}"))
		item = QTableWidgetItem("")
		item.setCheckState(Qt.Unchecked)
		self.window.table.setItem(0, 3, item)

		work = SubscribeData(self)
		self.window.threadpool.start(work)

		
		
	def handler(self, tick):
		print(tick)
		cur_time = pd.Timestamp.combine(pd.to_datetime(tick[3]).date(), pd.to_datetime(tick[4]).time())
		product = tick[0].upper()
		if product in self.sec_dict.keys():
			self.sec_dict[product].handler(tick)
			if product in self.trend_schedule and cur_time >= self.trend_schedule[product]:
				sharpe, momentum = self.sec_dict[product].get_trend(self.cycle_trend, self.N_trend)
				self.window.table.setItem(self.sec_idx[product], 0, QTableWidgetItem(f"{sharpe.iloc[-1]:.3f}"))
				self.window.table.setItem(self.sec_idx[product], 1, QTableWidgetItem(f"{momentum.iloc[-1]:.3f}"))
				self.sharpe_dict[product] = sharpe
				self.momentum_dict[product] = momentum
				
				if self.window.graphWindow is not None and self.sec_dict[product] in self.window.selected_ls:
					self.window.updateGraphWindow(self.sec_dict[product], 'sharpe')
					self.window.updateGraphWindow(self.sec_dict[product], 'momentum')

				self.trend_schedule[product] = cur_time + timedelta(days = self.period_trend[0], minutes = self.period_trend[2])
					

			if product in self.vol_schedule and cur_time >= self.vol_schedule[product]:
				vol = self.sec_dict[product].get_vol_single(self.cycle_vol, self.N_vol)
				self.window.table.setItem(self.sec_idx[product], 2, QTableWidgetItem(f"{vol.iloc[-1]:.5f}"))
				self.vol_dict[product] = vol

				if self.window.graphWindow is not None and self.sec_dict[product] in self.window.selected_ls:
					self.window.updateGraphWindow(self.sec_dict[product], 'vol')

				self.vol_schedule[product] = cur_time + timedelta(days = self.period_vol[0], minutes = self.period_vol[2])

		if self.whl_schedule is None or cur_time >= self.whl_schedule:
			res = self.vol_index_whole()
			self.window.table.setItem(0, 2, QTableWidgetItem(f"{res.iloc[0]}"))

			self.vol_whl = self.vol_whl[:-1]
			self.vol_whl[res.index[0]] = res.iloc[0]
			if self.window.graphWindow is not None and self.sec_dict[product] in self.window.selected_ls:
				self.window.updateGraphWindow(None, 'vol_whl')

			self.whl_schedule = cur_time + timedelta(days = self.period_whl[0], minutes = self.period_whl[2])
				
	
	def vol_index_whole(self, init = False):
		if self.period_whl[0] == 1:
			M = 1
		else:
			M = 360 / int(self.period_whl[2])
		N = int(M * self.N_mlty)
		weight = np.full(len(self.cmdty_future_ls), 1/N)
		close_df = pd.DataFrame()
		if init:
			for i in self.cmdty_future_ls:
				close_df = close_df.join(self.sec_dict[i].close, how = 'outer', rsuffix = i)
			close_df = close_df[
				((close_df.index.time >= time(hour = 9)) & (close_df.index.time <= time(hour = 15)) |
				(close_df.index.time >= time(hour = 21)) & (close_df.index.time <= time(hour = 23)))]
			close_df = close_df.groupby(pd.Grouper(freq = self.cycle_whl)).agg('last')
			close_df.dropna(inplace = True, axis = 0, how = 'all')
			close_df.fillna(inplace = True, method = 'ffill')
			rate = close_df.pct_change()
			market_vol = (rate.rolling(window = N).cov() @ weight).unstack() @ weight.T
			return market_vol.iloc[-200:]
		else:
			length = N * (self.period_whl[0] * 360 + self.period_whl[2])
			for i in self.cmdty_future_ls:
				close_df = close_df.join(self.sec_dict[i].close[-length:], how = 'outer', rsuffix = i)
			close_df = close_df[
				((close_df.index.time >= time(hour = 9)) & (close_df.index.time <= time(hour = 15)) |
				(close_df.index.time >= time(hour = 21)) & (close_df.index.time <= time(hour = 23)))]
			close_df = close_df.groupby(pd.Grouper(freq = self.cycle_whl)).agg('last')
			close_df.dropna(inplace = True, axis = 0, how = 'all')
			close_df.fillna(inplace = True, method = 'ffill')
			rate = close_df.pct_change().iloc[-N:, :]
			CO = rate.cov()
			market_vol = weight.T @ CO @ weight
			return pd.Series(market_vol,index = [rate.index[-1]])


from Constant import stk_ls, stk_idx_future_ls, cmdty_future_ls
logging.basicConfig(level = logging.DEBUG, format = '%(asctime)s - %(levelname)s - %(message)s', 
		handlers = [logging.StreamHandler()])
try:
	qapp = QApplication([])
	array = market_watch_index(stk_ls, stk_idx_future_ls, cmdty_future_ls, cycle_trend = '1min', cycle_vol = '1min', cycle_whl = 'D')
	qapp.exec_()
except Exception as e:
	print(e)