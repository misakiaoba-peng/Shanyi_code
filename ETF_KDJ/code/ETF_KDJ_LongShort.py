#-*-coding:utf-8-*-
# @title: ETF 无量尺多空策略
# @author: Brian Shan
# @date: 2020.07.08
# 注： 交易时间为9点30 - 11点30， 1点 - 3点
# ToDo: 目前是从dolphindb_data里取一分钟的数据重新组合成15分钟
#       原因是15分钟的数据目前有问题，等待技术部门更新


import os
import numpy as np
np.seterr(divide = 'ignore', invalid='ignore')
import pandas as pd
from datetime import datetime, timedelta
from matplotlib import pyplot as plt
from typing import Set
from typing import List

from Dolphindb_Data import GetData
from Generate_Min_Data import generate_any_minute_data
from Constant import trade_time, future_multiplier, margin_percent, margin_multiplier, output_path, log_path

import time
from multiprocessing import Pool, freeze_support
import logging
import logging.handlers


# logging setup
logger = logging.getLogger('ETF_KDJ_LongShort')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
fh = logging.handlers.RotatingFileHandler(os.path.join(log_path, 'log.txt'), maxBytes=10240, backupCount=5)
fh.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)


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
            logger: logger class
        """
        self.count: int = 0
        self.ETF_sym = ETF_sym
        self.future_sym = future_sym
        self.num_ETF = len(ETF_sym)
        self.num_future = len(future_sym)
        assert self.num_ETF > 0
        self.start = start
        self.end = end
        self.cycle: int = cycle

        self.money= 100_000_000
        self.get = GetData()
        self.close_df = None

        self.load_data()
        

    def load_data(self):
        # # read from dolphin_db
        # for sym in self.ETF_sym:
        #     data=self.get.Stock_candle(sym, self.start, self.end, 1) 
        #     if len(data) > 0:
        #         self.etf_handler(generate_any_minute_data(data, self.cycle))
        
        # read from csv from tinysoft
        data = pd.read_csv(os.path.dirname(os.path.realpath(__file__)) + "\\data_15min.csv",
                index_col = 0)
        data.index = pd.to_datetime(data.index)
        data.columns = [str(i) for i in data.columns]
        data = data.loc[(pd.to_datetime(self.start) <= data.index) & (data.index <= pd.to_datetime(self.end) + timedelta(days = 1))]
        self.close_df = data[self.ETF_sym]
        
        self.close_buy_df = self.close_df.copy() # close price for buying ETF and future
        self.close_sell_df = self.close_df.copy() # close price for selling ETF and future
        for prod in self.future_sym:
            switch_df = self.get.Future_hist_Mswitch(prod, self.start, self.end)
            
            dateStart = self.start
            dateStart_buy = self.close_df.index[0]
            dateStart_sell = self.close_df.index[0]
            if len(switch_df) > 0: # 主力合约换过
                for row in switch_df.values:
                    premain = row[4]
                    pre_tradingday = row[9]
                    data = self.get.Future_hist_candle(premain, dateStart, pre_tradingday.strftime("%Y.%m.%d"), 1)
                    if len(data) > 0:
                        dateStart_buy, dateStart_sell = self.future_handler(generate_any_minute_data(data, self.cycle), dateStart_buy, dateStart_sell)
                    dateStart = (pre_tradingday - timedelta(days = 3)).strftime("%Y.%m.%d")
                
                data = self.get.Future_hist_candle(
                        switch_df['main'].iloc[-1], 
                        switch_df['pre_tradingday'].iloc[-1].strftime("%Y.%m.%d"), 
                        self.end, 1
                    )
                if len(data) > 0:
                    dateStart_buy, dateStart_sell = self.future_handler(generate_any_minute_data(data, self.cycle), dateStart_buy, dateStart_sell)
            else: # 主力合约没有换过
                data = self.get.Future_hist_Mcandle(prod, self.start, self.end)
                if len(data) > 0:
                    dateStart_buy, dateStart_sell = self.future_handler(generate_any_minute_data(data, self.cycle), dateStart_buy, dateStart_sell)
        for prod in self.future_sym:
            idx_nan = np.where(self.close_sell_df[prod].isnull())[0]
            if len(idx_nan) > 0 and len(idx_nan) != idx_nan[-1] + 1:
                logger.warning(f"there are missing close price in future: {prod} between {self.start}-{self.end}")
        
        self.close_buy_df.fillna(method = 'ffill', inplace = True)
        self.close_sell_df.fillna(method = 'ffill', inplace = True)

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

    def etf_handler(self, data: pd.DataFrame):
        """
        处理单个ETF数据
        """
        sym = data['symbol'].iloc[0]
        data.set_index('date', inplace = True)
        data = data.rename(columns = {'close': sym})[[sym]]
        for g, g_df in data.groupby([data.index.hour, data.index.minute]):
            if g not in trade_time:
                data.drop(g_df.index, inplace = True)
        if self.close_df is None:
            self.close_df = data.copy()
        else:
            self.close_df = self.close_df.join(data, how = "outer", sort = True)  
        idx_nan = np.where(self.close_df[sym].isnull())[0]
        if len(idx_nan) > 0 and len(idx_nan) != idx_nan[-1] + 1:
            logger.warning(f"there are missing close price in stock: {sym} between {self.start}-{self.end}")

    def future_handler(self, data: pd.DataFrame, dateStart_buy: datetime, dateStart_sell:datetime):
        """
        处理单个期货品种数据
        """
        prod = data['product'].iloc[0]
        data.set_index('date', inplace = True)
        data = data.rename(columns = {'close': prod})[[prod]]
        # 去掉非ETF交易时间的数据
        for g, g_df in data.groupby([data.index.hour, data.index.minute]):
            if g not in trade_time:
                data.drop(g_df.index, inplace = True)
        # attach data to the close price for buy and the close price for sell 
        self.close_buy_df.loc[dateStart_buy:data.index[-2], prod] = data.loc[dateStart_buy:data.index[-2], prod]
        self.close_sell_df.loc[dateStart_sell:data.index[-1], prod] = data.loc[dateStart_sell:data.index[-1], prod]

        dateStart_buy = data.index[-1]
        idx = self.close_sell_df.index.get_loc(data.index[-1])
        if idx + 1 < len(self.close_sell_df):
            idx += 1
        dateStart_sell = self.close_sell_df.index[idx]
        return dateStart_buy, dateStart_sell

    def strategy(self, **args):
        """
        策略，生成每天的money ratio

        """
        self.StochLen1 = args['StochLen1']
        self.StochLen2 = args['StochLen2']
        self.SmoothingLen1 = args['SmoothingLen1']
        self.SmoothingLen2 = args['SmoothingLen2']
        self.weight = args['weight']
        
        def KDJ(data, StochLen, SmoothingLen):
            var0 = data.rolling(StochLen).min()
            var1 = data.rolling(StochLen).max()
            var2 = data - var0
            var3 = var1 - var0
            FastKCustom = var2/var3 * 100
            idx = np.where(var3 <= 0)
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
        self.avg_indicator = self.indicator.mean(axis = 1) # 各品种强弱指标均值
        self.std_indicator = self.indicator.std(axis = 1, ddof = 0) # 各品种强弱指标标准差
        self.indUp = self.avg_indicator + 1.5 * self.std_indicator # 指标上轨
        self.indDn = self.avg_indicator - 1.5 * self.std_indicator # 指标下轨
        self.indicator.mask(self.indicator.gt(self.indUp, axis = 0), self.indUp, axis = 0, inplace = True)
        self.indicator.mask(self.indicator.lt(self.indDn, axis = 0), self.indDn, axis = 0, inplace = True)
        self.ReIndicator = self.indicator.divide(self.avg_indicator, axis = 0) - 1
        self.ReIndicator[self.ReIndicator[self.ETF_sym] < 0] = 0 # 不空ETF
        self.MoneyRatio0 = self.ReIndicator.divide(self.ReIndicator.abs().sum(axis = 1), axis = 0)
        # 修改极值
        self.MoneyRatio0[self.MoneyRatio0 > 0.1] = 0.1
        self.MoneyRatio0[self.MoneyRatio0 < -0.1] = -0.1

    def generate_lots(self):
        """
        生成每天的仓位
        """
        self.lots = pd.DataFrame(
            index = self.close_buy_df.index, 
            columns = self.close_buy_df.columns
            )
        self.lots['cash'] = np.nan
        self.lots['total asset'] = self.money
        init = False
        lots_hold = None
        cash = None
        future_buy_price = None
        index_MoneyRatio = self.MoneyRatio0.index

        MoneyRatio0_mat = self.MoneyRatio0.values
        close_buy_mat = self.close_buy_df.values
        close_sell_mat = self.close_sell_df.values
        for idx in range(len(MoneyRatio0_mat)):
            row = MoneyRatio0_mat[idx]
            if np.isnan(row).all():
                continue
            i = index_MoneyRatio[idx]
            if not init:
                self.lots.loc[i, self.ETF_sym] = (row[:self.num_ETF] * self.money) / close_buy_mat[idx,:self.num_ETF] // 100 * 100
                cash = self.lots.at[i, 'total asset'] - np.nansum(self.lots.loc[i,self.ETF_sym].values * close_buy_mat[idx, :self.num_ETF])
                for j in range(self.num_future):
                    prod = self.future_sym[j]
                    cost_perhand = margin_percent[prod] * margin_multiplier * future_multiplier[prod] * \
                                close_buy_mat[idx,self.num_ETF + j] 
                    self.lots.loc[i, prod] = (row[self.num_ETF + j] * self.money) // cost_perhand
                init = True
            else:
                total_asset = np.nansum(lots_hold[:self.num_ETF] * close_sell_mat[idx, :self.num_ETF]) + cash
                for j in range(self.num_future):
                    # 期货的利润
                    prod_idx = self.num_ETF + j
                    if not np.isnan(lots_hold[prod_idx]):
                        total_asset += lots_hold[prod_idx] * (close_sell_mat[idx, prod_idx] - future_buy_price[j]) * future_multiplier[self.future_sym[j]]

                self.lots.loc[i, self.ETF_sym] = (row[:self.num_ETF] * total_asset)/close_buy_mat[idx,:self.num_ETF] // 100 * 100
                cash = total_asset - np.nansum(self.lots.loc[i, self.ETF_sym].values * close_buy_mat[idx,:self.num_ETF])
                self.lots.loc[i, 'total asset'] = total_asset
                
                for j in range(self.num_future):
                    prod = self.future_sym[j]
                    cost_perhand = margin_percent[prod] * margin_multiplier * future_multiplier[prod] * close_buy_mat[idx, self.num_ETF + j] 
                    self.lots.loc[i, prod] = (row[self.num_ETF+j] * self.money) // cost_perhand
                
            lots_hold = self.lots.loc[i, self.close_buy_df.columns].values
            future_buy_price = close_buy_mat[idx, -self.num_future:]
            self.lots.loc[i, 'cash'] = cash

    def performance(self):
        """
        生成报告
        """
        source = self.lots[['total asset']].resample('1D').last()
        source.dropna(inplace = True)
        
        plt.figure(figsize = (16,12))
        plt.plot(source['total asset'])
        plt.ylabel('Yuan')
        plt.title(f"Total Asset Time Series Graph during {source.index[0].strftime('%y%m%d')}-{source.index[-1].strftime('%y%m%d')}")
        
        outdir = os.path.join(output_path, f"{self.start}_{self.end}")
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        plt.savefig(os.path.join(outdir, 
            f"total_asset_{source.index[0].strftime('%y%m%d')}_{source.index[-1].strftime('%y%m%d')}" + \
                f"_{self.StochLen1}_{self.StochLen2}_{self.SmoothingLen1}_{self.SmoothingLen2}_{self.weight}.png"))
        plt.close()

        source['ret'] = source['total asset'].pct_change(1)
        source['ret'].iloc[0] = source['total asset'].iloc[0] / self.money - 1
        source['nav'] = (1 + source.ret.values).cumprod()
        profit_loss_ratio = -source[source['ret']>0]['ret'].mean()/source[source['ret']<0]['ret'].mean()  # 盈亏比（日）
        # 最大日收益
        daily_max = source['ret'].max()
        # 最大日亏损率
        daily_min = source['ret'].min()
        rety = source['nav'][-1] ** (252 / source.shape[0]) - 1  # 年化收益率
        sharp = source['ret'].mean() / source['ret'].std() * np.sqrt(252)
        MDD = max(1 - source['nav'] / source['nav'].cummax())
        if MDD != 0:
            MAR = rety / MDD
        else:
            MAR = 0
        result = {
            '累计收益率': source['nav'][-1] - 1, 
            'Sharpe': sharp, 
            '年化收益': rety,
            '盈亏比': profit_loss_ratio,
            '最大日收益率': daily_max, 
            '最大日亏损率': daily_min,
            '最大回撤': MDD, 
            'MAR': MAR}
        self.result = pd.DataFrame.from_dict(result, orient='index').T
        self.source = source
        # self.nav_permonth = source['nav'].resample('1M').last() / source[
        #     'nav'].resample('1M').first() - 1


def run(start: str, end: str, ETF_ls: list, future_ls:list, 
            StochLen:list, SmoothingLen:list, weight:list, cycle:"int > 0"):
    """
    主程序
    """
    summary_df = pd.DataFrame(index = [], 
        columns = ['StochLen1', 'StochLen2', 'SmoothingLen1', 'SmoothingLen2', 'weight', \
            '累计收益率', 'Sharpe', '年化收益', '盈亏比', '最大日收益率', '最大日亏损率', '最大回撤', 'MAR'])

    outdir = os.path.join(output_path, f"{start}_{end}")
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    c = ETF_KDJ_LongShort(ETF_ls, future_ls, start, end, cycle)
    logger.info(f"Load data {start}-{end} finished")
    for i in range(len(StochLen)-1):
        for j in range(i+1, len(StochLen)):
            for k in SmoothingLen:
                for l in SmoothingLen:
                    for m in weight:
                        args  = {}
                        args['StochLen1'] = StochLen[i]
                        args['StochLen2'] = StochLen[j]
                        args['SmoothingLen1'] = k
                        args['SmoothingLen2'] = l
                        args['weight'] = m

                        logger.info(f"start Args:{','.join([str(i) for i  in args.values()])} between {start}-{end}")
                        
                        start_time = time.time()
                        c.Backtest(**args)
                        logger.info(f"Time used: {time.time() - start_time}s for args:" +
                            f"{','.join([str(i) for i  in args.values()])} between {start}-{end}")

                        # 导出数据
                        writer = pd.ExcelWriter(os.path.join(
                            outdir,
                            f"KDJ_Arg_{c.start}_{c.end}_{'_'.join([str(i) for i  in args.values()])}.xlsx"
                            ))
                        pd.DataFrame(args.items()).to_excel(writer, sheet_name = "参数表")
                        # c.close_df.to_excel(writer, sheet_name = "15分钟级收盘价")
                        # c.var0_s.to_excel(writer, sheet_name = "短期var0")
                        # c.var1_s.to_excel(writer, sheet_name = "短期var1")
                        # c.var2_s.to_excel(writer, sheet_name = "短期var2")
                        # c.var3_s.to_excel(writer, sheet_name = "短期var3")
                        # c.FastKCustom_s.to_excel(writer, sheet_name = "短期FastKCustom")
                        # c.Kvalue_s.to_excel(writer, sheet_name = "短期K值")
                        # c.Dvalue_s.to_excel(writer, sheet_name = "短期D值")
                        # c.Jvalue_s.to_excel(writer, sheet_name = "短期J值")
                        # c.var0_l.to_excel(writer, sheet_name = "长期var0")
                        # c.var1_l.to_excel(writer, sheet_name = "长期var1")
                        # c.var2_l.to_excel(writer, sheet_name = "长期var2")
                        # c.var3_l.to_excel(writer, sheet_name = "长期var3")
                        # c.FastKCustom_l.to_excel(writer, sheet_name = "长期FastKCustom")
                        # c.Kvalue_l.to_excel(writer, sheet_name = "长期K值")
                        # c.Dvalue_l.to_excel(writer, sheet_name = "长期D值")
                        # c.Jvalue_l.to_excel(writer, sheet_name = "长期J值")
                        # c.close_buy_df.to_excel(writer, sheet_name = "买入价")
                        # c.close_sell_df.to_excel(writer, sheet_name = "卖出价")
                        c.indicator.to_excel(writer, sheet_name = "Indicator")
                        # c.avg_indicator.to_excel(writer, sheet_name = "AvgIndicator")
                        # c.ReIndicator.to_excel(writer, sheet_name = "ReIndicator")
                        c.MoneyRatio0.to_excel(writer, sheet_name = 'MoneyRatio0')
                        c.lots.to_excel(writer, sheet_name = '仓位')
                        c.source.to_excel(writer, sheet_name = '利润表')
                        c.result.to_excel(writer, sheet_name = '总结')
                        # c.nav_permonth.to_excel(writer, sheet_name = '月度净值')
                        writer.save()

                        summary_df.loc[len(summary_df)] = list(args.values()) + list(c.result.values[0])

    summary_df.to_csv(os.path.join(our_dir, f'summary_{start}_{end}.csv'), encoding='utf_8_sig', index = False)


if __name__ == '__main__':
    freeze_support() # prevent raising run-time error from running the frozen executable

    whole_list = ['510300','510330','510050','159919','510310','159949','510500',
        '159915','512500', '159968','515800','512990','512380','512160','512090',
        '159995','512760','515050','159801','512480','512290','159992','512170',
        '512010','159938','515000','515750','159807','515860','159987','515030',
        '515700','159806','512880','512000','512800','512900','159993','159928',
        '512690','515650','159996','510150','512660','512710','515210','512400',
        '515220','159966','159905','159967','510880','515180','515680','515900',
        '159976','515600','159978','511010','511260','159972','510900','159920',
        '513050','513090','513500','518880','159934','159937','518800','159980'
        ]
    select_list = ['510050', '159995', '512090', '515050', '512290', '515000', 
        '515700', '512800', '159928', '512660', '512400', '510880', '159976', 
        '511010', '510900', '518880']
    
    future_list = ['IH', 'IF', 'IC', 'TF', "T"]

    # 参数
    StochLen = [5, 9, 18, 25, 34, 46, 72, 89]
    SmoothingLen = [3, 8, 13, 18]
    weight = [0.2, 0.4, 0.6, 0.8]

    start_train = ['2016.01.01', '2017.01.01', '2018.01.01']
    end_train = ['2018.01.01', '2019.01.01', '2020.01.01']
    
    pool = Pool()
    results = []
    for start, end in zip(start_train, end_train):
        results.append(pool.apply_async(run, 
            args = (start, end, select_list, future_list, StochLen, SmoothingLen, weight, 15)))
    pool.close()
    pool.join

    for i in results:
        i.get()

