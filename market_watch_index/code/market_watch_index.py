#-*-coding:utf-8-*-
# @title: 市场监控指数主程序
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
	"""
	转换回溯长度成tuple(日，小时，分钟)

	Args:
		cycle: str, 回测长度，支持格式: '%{x}min' 和 'D'

	Return:
		tuple, (日，小时，分钟)
	"""
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


class market_watch_index(QObject):
	"""
	市场观测主程序

	"""
	dataSignal = pyqtSignal(list) # 新数据的信号

	def __init__(self, 
				stk_ls: List[str] = [], stk_idx_future_ls: List[str]= [], cmdty_future_ls: List[str] = [], 
				cycle_trend: str = '15min', N_trend:int = 200, 
				cycle_vol:str = 'D', N_vol:int = 60,
				cycle_whl:str = 'D', N_mlty:int = 60) -> None:

		super().__init__()
		self.stk_ls = stk_ls
		self.stk_idx_future_ls = stk_idx_future_ls
		self.cmdty_future_ls = cmdty_future_ls
		self.total_ls = stk_ls + stk_idx_future_ls + cmdty_future_ls
		
		self.period_trend = str_to_time(cycle_trend)
		self.cycle_trend = cycle_trend
		self.N_trend = N_trend
		self.period_vol = str_to_time(cycle_vol)
		self.N_vol = N_vol
		self.cycle_vol = cycle_vol
		self.cycle_whl = cycle_whl
		self.period_whl = str_to_time(cycle_whl)
		self.N_mlty = N_mlty

		# 存生成的数据
		self.sec_dict = {} # 所有的证券数据
		self.sec_idx = {} # 证券对应的标号从1开始
		self.trend_schedule = {} # 每个证券生成趋势度的下个时间
		self.vol_schedule = {} # 每个证券生成波动率的下个时间
		self.whl_schedule = None # 生成全商品波动率的下个时间
		self.sharpe_dict = {} # 存每个证券的夏普比率
		self.momentum_dict = {} # 存每个证券的动量效率
		self.vol_dict = {}  # 存每个证券的波动率
		self.vol_whl = None # 存全商品的波动率

		# 生成表格窗口
		self.window = ApplicationWindow(
						len(self.total_ls) + 1, 
						4, 
						["全商品"] + [f"{future_name_dict[i]}({i})" for i in self.total_ls],
						['夏普比率', '动量效率', '波动率', '选择'],
						self
					)
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
			idx += 1

		self.window.table.setItem(0, 0, QTableWidgetItem("-"))
		self.window.table.setItem(0, 1, QTableWidgetItem("-"))
		self.vol_whl = self.vol_index_whole(True)
		self.whl_schedule = self.vol_whl.index[-1] + timedelta(days = self.period_whl[0], minutes = self.period_whl[2])
		self.window.table.setItem(0, 2, QTableWidgetItem(f"{self.vol_whl.iloc[-1]:.5g}"))
		
		# 生成右下角的标签
		labelText = "计算周期:\n"
		labelText += f"夏普比率:{'1D' if self.period_trend[0] > 0 else str(self.period_trend[2]) + 'min'}\n"
		labelText += f"动量效率:{'1D' if self.period_trend[0] > 0 else str(self.period_trend[2]) + 'min'}\n"
		labelText +=  f"单品种波动率:{'1D' if self.period_vol[0] > 0 else str(self.period_vol[2]) + 'min'}\n"
		labelText += f"全商品波动率:{'1D' if self.period_whl[0] > 0 else str(self.period_whl[2]) + 'min'}\n"
		self.window.label.setText(labelText)

		# 连接信号和槽
		self.dataSignal.connect(self.handler)

		# 订阅数据
		work = SubscribeData(self.dataSignal)
		self.window.threadpool.start(work)
		
	def handler(self, tick:list) -> None:
		"""	
		主程序处理新到的ticker

		Args：
			tick: list
		"""
		cur_time = pd.Timestamp.combine(pd.to_datetime(tick[3]).date(), pd.to_datetime(tick[4]).time())
		product = tick[0].upper()
		# 更新每个证券内部数据，夏普，动量，以及表格
		if product in self.sec_dict.keys():
			self.sec_dict[product].handler(tick)
			if product in self.trend_schedule and cur_time >= self.trend_schedule[product]:
				sharpe, momentum = self.sec_dict[product].get_trend(self.cycle_trend, self.N_trend)
				self.window.table.item(self.sec_idx[product], 0).setText(f"{sharpe.iloc[-1]:.3f}")
				self.window.table.item(self.sec_idx[product], 1).setText(f"{momentum.iloc[-1]:.3f}")
				self.sharpe_dict[product] = sharpe
				self.momentum_dict[product] = momentum
				
				if self.window.graphWindow is not None and self.sec_idx[product] in self.window.selected_ls:
					self.window.updateGraphWindow(self.sec_idx[product], 'sharpe')
					self.window.updateGraphWindow(self.sec_idx[product], 'momentum')
				
				self.trend_schedule[product] = cur_time + timedelta(days = self.period_trend[0], minutes = self.period_trend[2])
					
			if product in self.vol_schedule and cur_time >= self.vol_schedule[product]:
				vol = self.sec_dict[product].get_vol_single(self.cycle_vol, self.N_vol)
				self.window.table.item(self.sec_idx[product], 2).setText(f"{vol.iloc[-1]:.5f}")
				self.vol_dict[product] = vol

				if self.window.graphWindow is not None and self.sec_idx[product] in self.window.selected_ls:
					self.window.updateGraphWindow(self.sec_idx[product], 'vol')

				self.vol_schedule[product] = cur_time + timedelta(days = self.period_vol[0], minutes = self.period_vol[2])

		if self.whl_schedule is None or cur_time >= self.whl_schedule:
			res = self.vol_index_whole()
			self.window.table.item(0, 2).setText(f"{res.iloc[0]:.5g}")

			self.vol_whl = self.vol_whl[:-1]
			self.vol_whl[res.index[0]] = res.iloc[0]

			if self.window.graphWindow is not None and self.sec_idx[product] in self.window.selected_ls:
				self.window.updateGraphWindow(None, 'vol_whl')

			self.whl_schedule = cur_time + timedelta(days = self.period_whl[0], minutes = self.period_whl[2])
				
	
	def vol_index_whole(self, init = False) -> pd.Series:
		"""
		生成全商品波动率

		Args:
			init: bool, 是否是第一次生成
		Return:
			pd.Series: 第一次生成，长度为200个点，后面都仅生成最新的值
		"""
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

if __name__ == '__main__':
	from Constant import stk_ls, stk_idx_future_ls, cmdty_future_ls
	from Constant import  cycle_trend, N_trend, cycle_vol, N_vol, cycle_whl, N_mlty
	
	logging.basicConfig(level = logging.DEBUG, format = '%(asctime)s - %(levelname)s - %(message)s', 
			handlers = [logging.StreamHandler()])
	try:
		qapp = QApplication([])
		array = market_watch_index(
			stk_ls = stk_ls, 
			stk_idx_future_ls = stk_idx_future_ls, 
			cmdty_future_ls = cmdty_future_ls, 
			cycle_trend = cycle_trend,
			N_trend = N_trend,
			cycle_vol = cycle_vol,
			N_vol = N_vol,
			cycle_whl = cycle_whl,
			N_mlty = N_mlty
			)
		qapp.exec_()
	except Exception as e:
		logging.error(e)