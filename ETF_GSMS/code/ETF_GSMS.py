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
from Constant import future_multiplier, output_path, money
from Constant import output_path, log_path, select_ls, whole_ls, hs300
from Constant import ETF_dict, commission_multiplier, commission_stock

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
		self.start = (pd.to_datetime(start) + timedelta(days = -180)).strftime("%Y.%m.%d") # 预读半年数据
		self.start_backtest = pd.to_datetime(start)
		self.end = end
		self.get = GetData()
		self.ETF_MS = ETF_MS(stock_syms, self.start, end)
		self.ms = self.ETF_MS.get_MS()
		self.ms.fillna(value = 0, inplace = True)
		self.open_df = pd.DataFrame()
		self.load_data()

	def load_data(self):
		for sym in self.stock_syms:
			data=self.get.Stock_candle(sym, (self.start_backtest+timedelta(days = -10)).strftime("%Y.%m.%d"), self.end, 'D')
			if len(data) > 0:
				self.etf_handler(data)
			else:
				self.open_df[sym] = np.nan
		self.open_df.replace(to_replace = [0, np.nan], method = 'ffill', inplace = True)

		self.commission_pertrade = np.zeros(self.open_df.shape)
		for i in range(len(self.stock_syms)):
			if self.stock_syms[i] in ETF_dict:
				self.commission_pertrade[:, i] = ETF_dict[self.stock_syms[i]][1] * commission_multiplier
			else:
				self.commission_pertrade[:, i] = commission_stock

	def etf_handler(self, data):
		"""
		处理单个股票数据
		"""
		sym = data['symbol'].iloc[0]
		data['date'] = data['date'].dt.date
		data.set_index('date', inplace = True)
		data['close'] = data['close'].shift()
		data.open.replace(to_replace = [0], value = np.nan, inplace = True)
		data.open.fillna(data.close, inplace = True) # fill the 0 or nan in open price with previous close
		data = data.rename(columns = {'open': sym})[[sym]]
		data = data[data.index >= self.start_backtest]
		self.open_df = self.open_df.join(data, how = "outer", sort = True)

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
		self.hold_number = int(args['hold_number'])
		self.period = int(args['period'])
		self.hold_period = int(args['hold_period'])
		self.gsms = self.ms.rolling(window = self.period).apply(lambda x: x.sum(axis = 0)/x.std(axis = 0))
		self.gsms = self.gsms[self.gsms.index >= self.start_backtest]
		self.gsms_rank = self.gsms.rank(axis = 1)
		num_stock = self.gsms_rank.count(axis = 1)
		if self.hold_number > num_stock.min():
			self.hold_number = num_stock.min()
		self.MoneyRatio = self.gsms_rank.gt(num_stock - self.hold_number, axis = 0).astype(int)/self.hold_number


	def generate_lots(self):
		open_buy_mat = self.open_df.values

		self.trade = money / self.hold_period * self.MoneyRatio // (open_buy_mat * 100) * 100
		self.trade += -1 * self.trade.shift(self.hold_period).fillna(value = 0)
		self.lots = self.trade.cumsum()
		lots_mat = self.lots.values
		pnl = np.nansum((open_buy_mat[1:] - open_buy_mat[:-1]) * lots_mat[:-1], axis = 1)

		trade_mat = np.concatenate(([lots_mat[0]], np.diff(lots_mat, axis = 0)))
		trade_mat_pos = trade_mat.copy()
		trade_mat_neg = trade_mat.copy()
		trade_mat_pos[trade_mat_pos < 0] = 0
		trade_mat_neg[trade_mat_neg > 0] = 0
		turnover_mat = (trade_mat_pos-trade_mat_neg) * open_buy_mat
		commission_mat = turnover_mat * self.commission_pertrade


		self.lots['commission'] = np.nansum(commission_mat, axis = 1)
		self.lots['PnL'] = np.concatenate(([0], pnl)) - self.lots['commission'].values
		self.lots['total asset'] = self.lots['PnL'].cumsum() + money

		self.summary = pd.DataFrame(
			columns = self.lots.columns[:self.MoneyRatio.shape[1]], 
			index = ['交易额', '最大手数', '手数中间值', '平均手数', '最小手数'])
		self.summary.loc['交易额', :] = np.sum(turnover_mat, axis = 0)
		self.summary.iloc[1:, :] = self.lots.iloc[:, :self.MoneyRatio.shape[1]].agg(['max', 'median', 'mean', 'min'], axis = 0).values.tolist()
				
	def performance(self):
		"""
		生成报告
		"""
		self.lots['ret'] = self.lots['PnL'] / money
		
		outdir = os.path.join(output_path, f"{self.start_backtest.strftime('%Y.%m.%d')}_{self.end}")
		if not os.path.exists(outdir):
			os.makedirs(outdir, exist_ok = True)

		plt.figure(figsize = (16,12))
		plt.plot(self.lots['total asset'])
		plt.ylabel('Yuan')
		plt.title(f"Total Asset Time Series Graph during {self.lots.index[0].strftime('%y%m%d')}-{self.lots.index[-1].strftime('%y%m%d')}")
		plt.savefig(os.path.join(outdir, 
			f"total_asset_{self.hold_number}_{self.period}_{self.hold_period}" + \
				f"_{self.lots.index[0].strftime('%y%m%d')}_{self.lots.index[-1].strftime('%y%m%d')}.png"))
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


def select_params(nums):
	df_ls = []
	des = os.path.join(output_path, 'selected')
	os.makedirs(des, exist_ok = True)
	for date_dir, _, _ in os.walk(output_path):
		if date_dir == output_path or date_dir == des:
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
	res = df_total.groupby(['hold_number', 'period', 'hold_period']).size().reset_index(name = 'counts')
	res.sort_values('counts', ascending = False, inplace = True)
	res = res[res['counts'] == 3]
	res.to_csv(os.path.join(output_path, 'para_summary.csv'), encoding='utf_8_sig', index = False)

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
					f"{','.join([str(i) for i  in args.values()])} between {start}-{end}")
	# 导出数据
	lock.acquire()
	with open(os.path.join(outdir, f'summary_{start}_{end}.csv'), 'a', encoding='utf_8_sig', newline='') as csvfile:
		csv.writer(csvfile).writerow(list(args.values()) + list(c.result.values[0]))
	lock.release()

	if output_excel:
		with pd.ExcelWriter(os.path.join(
				outdir,
				f"GSMS_Arg_{'_'.join([str(i) for i  in args.values()])}_{start}_{end}.xlsx"
				)) as writer_excel:
			pd.DataFrame(args.items()).to_excel(writer_excel, sheet_name = "参数表")
			c.gsms.to_excel(writer_excel, sheet_name = 'gsms')
			c.MoneyRatio.to_excel(writer_excel, sheet_name = 'MoneyRatio0')
			c.lots.to_excel(writer_excel, sheet_name = '仓位')
			c.result.to_excel(writer_excel, sheet_name = '指标')
			c.summary.to_excel(writer_excel, sheet_name = '总结')

def init(l, q):
	global lock
	lock = l
	qh = QueueHandler(q)
	logger = logging.getLogger()
	logger.setLevel(logging.INFO)
	logger.addHandler(qh)

def run(start: str, end: str, arg_mat:list, stock_ls: list = select_ls, output_excel = False):
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

	c = ETF_GSMS(stock_ls, start, end)
	logger.info(f"Load data {start}-{end} finished")

	with open(os.path.join(outdir, f'summary_{start}_{end}.csv'), 'a', encoding='utf_8_sig', newline='') as csvfile:
		writer_csv = csv.writer(csvfile)
		writer_csv.writerow([ 'hold_number', 'period', 'hold_period', \
			'累计收益率', 'Sharpe', '年化收益', '胜率', '盈亏比', '最大日收益率', '最大日亏损率', '最大回撤', 'MAR', '累计手续费'])

	l = Lock()
	pool = Pool(maxtasksperchild = 400, initializer  = init, initargs = (l, q))
	for row in arg_mat:
		args  = {}
		args['hold_number'] = int(row[0])
		args['period'] = int(row[1])
		args['hold_period'] = int(row[2])
		pool.apply_async(run_parameters, args = (c, args, outdir, output_excel))
	pool.close()
	pool.join()
	ql.stop()



if __name__ == '__main__':
	freeze_support() # prevent raising run-time error from running the frozen executable

	# 参数
	hold_number = [1,2,3,4,5] # 股票持仓数
	period = [20,25,30,35,40] # 计算GSMS的周期
	hold_period = [20,25,30,35,40,45,50,55,60] #持有周期

	arg_mat = list(itertools.product(hold_number, period, hold_period))

	start_train = ['2016.01.01', '2017.01.01', '2018.01.01']
	end_train = ['2018.01.01', '2019.01.01', '2020.01.01']

	
	# ---------------------------------------跑Train的参数------------------------------------------------------------------
	
	# for start, end in zip(start_train, end_train):
	# 	run(start, end, arg_mat, select_ls, False)

	#---------------------------------------生成train里共同的优质参数组-------------------------------------------

	# select_params(70)

	# ---------------------------------------test------------------------------------------------------------------
	# 
	# params = pd.read_csv(os.path.join(output_path, 'para_summary.csv'))
	# run('2020.01.01', '2020.07.01', params.values, select_ls, False)

	# ---------------------------------------删去test集里不好的参数组-------------------------------------------------------
	# 
	# 
   #  test_result = pd.read_csv(os.path.join(output_path,'2020.01.01_2020.07.01', 'summary_2020.01.01_2020.07.01.csv'))
   #  final_params = test_result[test_result['MAR'] > 2].iloc[:,:2]
   #  if len(test_result) > 0:
			# print(f"Reserve Ratio: {len(final_params)}/{len(test_result)} = {len(final_params)/len(test_result)}")
   #  final_params.to_csv(os.path.join(output_path, 'final_params.csv'), index = False)

	# ---------------------------------------生成全周期报告--------------------------------------------------------------
	#
	
	params = pd.read_csv(os.path.join(output_path,'final_params.csv'))
	run('2016.01.01', '2020.07.01', params.values, select_ls, True)
	
	
	# ---------------------------------------DEBUG----------------------------------------------------------------
	# run('2018.01.01', '2020.01.01', [[2, 20, 20]], select_ls, True)
