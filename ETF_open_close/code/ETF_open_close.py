#-*-coding:utf-8-*-
# @title: ETF单品种早买晚卖策略
# @author: Brian Shan
# @date: 2020.08.03
# 注： 交易时间为9点30 - 11点30， 1点 - 3点


import os
import csv
import numpy as np
np.seterr(divide = 'ignore', invalid='ignore')
import pandas as pd
from datetime import datetime, timedelta
from matplotlib import pyplot as plt
from typing import Set
from typing import List
import itertools

from Dolphindb_Data import GetData
from Generate_Min_Data import generate_any_minute_data

import time
from multiprocessing import Pool, freeze_support, Lock, get_logger,Queue
import logging
from logging.handlers import QueueHandler, QueueListener


from Constant import money, commission, commission_multiplier, output_path, log_path


class ETF_open_close(object):
	def __init__(self, ETF_symbol:str, start: str, end: str):
		self.ETF_symbol = ETF_symbol
		self.start = start
		self.end = end
		self.get = GetData()
		self.load_data()

	def load_data(self):
		self.data = self.get.Stock_candle(self.ETF_symbol, self.start, self.end, 1)
		self.data.set_index('date', inplace = True)
		self.handler()

	def handler(self):
		"""
		Turnover在2018.02.01前后出现不一致，前为每分钟的交易额，后为累计交易额

		"""
		temp = self.data[self.data.index >= datetime(2018,2,1)]
		for g_key, g_df in temp.groupby(temp.index.date):
			first_min = g_df['turnover'].iloc[0]
			g_df['turnover'] = g_df['turnover'].diff()
			g_df['turnover'].iloc[0] = first_min
			self.data.loc[g_df.index, 'turnover'] = g_df['turnover']
	
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
		self.trade_period = args['trade_period']
		assert self.trade_period > 0
		self.barrier = args['barrier']
		assert self.barrier > 0 and self.barrier < 1
		g_date = self.data.groupby(self.data.index.date)
		self.lots = pd.DataFrame(index = [k for k,_ in g_date], columns = ['buy vwap', 'touch barrier', 'sell vwap'])
		for g_key, g_df in g_date:
			day_open_p = g_df['open'].iloc[0]
			bar = day_open_p * (1 - self.barrier)
			turnover_buy, turnover_sell = 0, 0
			volume_buy, volume_sell = 0, 0
			touch_barrier = False
			idx, idx_sell = 0, 0
			for row in g_df.values:
				close_p = row[4]
				volume = row[5]
				turnover = row[6]
				if close_p <= bar:
					touch_barrier = True
				if idx < self.trade_period and not touch_barrier:
					volume_buy += volume
					turnover_buy += turnover
				idx += 1
				if (touch_barrier or idx >= len(g_df) - self.trade_period) and idx_sell < self.trade_period:
					volume_sell += volume
					turnover_sell += turnover
					idx_sell += 1
				if idx_sell == self.trade_period:
					break
			self.lots.loc[g_key, :] = turnover_buy/volume_buy, touch_barrier, turnover_sell/volume_sell
	
	def generate_lots(self):
		self.lots['hand'] = money //(100 * self.lots['buy vwap'])
		self.lots['commission'] = (self.lots['buy vwap'] + self.lots['sell vwap']) * self.lots['hand']  * 100 * commission * commission_multiplier
		self.lots['PnL'] = (self.lots['sell vwap'] - self.lots['buy vwap']) * self.lots['hand'] * 100 - self.lots['commission']
		self.lots['total asset'] = self.lots['PnL'].cumsum() + money

	def performance(self):
		"""
		生成报告
		"""
		self.lots['ret'] = self.lots['PnL'] / money
		
		outdir = os.path.join(output_path, f"{self.start}_{self.end}")
		if not os.path.exists(outdir):
			os.makedirs(outdir, exist_ok = True)

		plt.figure(figsize = (16,12))
		plt.plot(self.lots['total asset'])
		plt.ylabel('Yuan')
		plt.title(f"Total Asset Time Series Graph during {self.lots.index[0].strftime('%y%m%d')}-{self.lots.index[-1].strftime('%y%m%d')}")
		plt.savefig(os.path.join(outdir, 
			f"total_asset_{self.trade_period}_{self.barrier}_{self.lots.index[0].strftime('%y%m%d')}_{self.lots.index[-1].strftime('%y%m%d')}.png"))
		plt.close()

		VictoryRatio = np.sum(self.lots['PnL'] > 0)/(np.sum(self.lots['PnL'] > 0) + np.sum(self.lots['PnL'] < 0)) # 胜率
		
		self.lots['nav'] = self.lots.ret.cumsum() + 1
		profit_loss_ratio = -self.lots[self.lots['ret']>0]['ret'].mean()/self.lots[self.lots['ret']<0]['ret'].mean()  # 盈亏比（日）
		# 最大日收益
		daily_max = self.lots['ret'].max()
		# 最大日亏损率
		daily_min = self.lots['ret'].min()
		rety = (self.lots['nav'][-1] - 1) * (252 / self.lots.shape[0]) # 年化收益率
		sharp = rety / (self.lots['ret'].std() * np.sqrt(252))
		MDD = max(1 - self.lots['nav'] / self.lots['nav'].cummax())
		if MDD != 0:
			MAR = rety / MDD
		else:
			MAR = 0
		result = {
			'累计收益率': self.lots['nav'][-1] - 1, 
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
		# self.nav_permonth = source['nav'].resample('1M').last() / source[
		#     'nav'].resample('1M').first() - 1


def run_parameters(c, args, outdir, output_excel):	
	start = c.start
	end = c.end
	start_time = time.time()
	try:
		c.Backtest(**args)
	except Exception as e:
		logging.error(f"Error: {e} at args: {','.join([str(i) for i  in args.values()])} between {start}-{end}")
		return
	logging.info(f"Time used: {time.time() - start_time:.3f}s for args: " + \
					f"{','.join([str(i) for i  in args.values()])} between {start}-{end}")
	# 导出数据
	lock.acquire()
	with open(os.path.join(outdir, f'summary_{start}_{end}.csv'), 'a', encoding='utf_8_sig', newline='') as csvfile:
		csv.writer(csvfile).writerow(list(args.values()) + list(c.result.values[0]))
	lock.release()
	
	if output_excel:
		with pd.ExcelWriter(os.path.join(
					outdir,
					f"open_close_Arg_{'_'.join([str(i) for i  in args.values()])}_{start}_{end}.xlsx"
					)) as writer_excel:
				pd.DataFrame(args.items()).to_excel(writer_excel, sheet_name = "参数表")
				c.lots.to_excel(writer_excel, sheet_name = '仓位')
				c.result.to_excel(writer_excel, sheet_name = '指标')

def init(l, q):
	global lock
	lock = l
	qh = QueueHandler(q)
	logger = logging.getLogger()
	logger.setLevel(logging.INFO)
	logger.addHandler(qh)


def run(start: str, end: str, arg_mat:list, stock = '510050', output_excel = False):
	"""
	主程序
	"""

	# 生成输出的文件夹
	outdir = os.path.join(output_path, f"{start}_{end}")
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

	c = ETF_open_close(stock, start, end)
	logger.info(f"Load data {start}-{end} finished")

	with open(os.path.join(outdir, f'summary_{start}_{end}.csv'), 'a', encoding='utf_8_sig', newline='') as csvfile:
		writer_csv = csv.writer(csvfile)
		writer_csv.writerow(['交易周期', 'Bar值' \
			'累计收益率', 'Sharpe', '年化收益', '胜率', '盈亏比', '最大日收益率', '最大日亏损率', '最大回撤', 'MAR', '累计手续费'])

	l = Lock()
	pool = Pool(maxtasksperchild = 500, initializer  = init, initargs = (l, q))
	for row in arg_mat:
		args  = {}
		args['trade_period'] = int(row[0])
		args['barrier'] = row[1]
		pool.apply_async(run_parameters, args = (c, args, outdir, output_excel))
	pool.close()
	pool.join()
	ql.stop()


if __name__ == '__main__':
	freeze_support() # prevent raising run-time error from running the frozen executable

	# 参数
	trade_period = [5, 10, 15, 20, 25, 30]
	barrier = [0.005, 0.0075, 0.01, 0.0125, 0.015, 0.0175, 0.02]

	arg_mat = list(itertools.product(trade_period, barrier))

	start_train = ['2016.01.01', '2017.01.01', '2018.01.01']
	# end_train = ['2016.02.01', '2017.02.01', '2018.02.01']
	end_train = ['2018.01.01', '2019.01.01', '2020.01.01']
	
	# ---------------------------------------跑Train的参数------------------------------------------------------------------
	
	for start, end in zip(start_train, end_train):
		run(start, end, arg_mat, '510050', False)

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
		
				
					
				

