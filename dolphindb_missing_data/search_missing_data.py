#-*-coding:utf-8-*-
# @title: 查找dolpindb里缺失数据的相关信息
# @author: Brian Shan
# @date: 2020.07.16

import os
import numpy as np
import pandas as pd
from Dolphindb_Data import GetData

def filter_trading_time(data: pd.DataFrame):
    """
    去除非交易时间的数据点, 是Inplace的修改

    Args:
        data: pd.DataFrame, 所需清理的数据，index为datetime.datetime

    Return:
        None
    """
    for g, g_df in data.groupby([data.index.hour, data.index.minute]):
        if not (g[0] == 10 or 13 <= g[0] <= 14 or \
                (g[0] == 9 and g[1] >= 31) or \
                (g[0] == 11 and g[1] <= 30) or \
                (g[0] == 15 and g[1] == 0)):
            data.drop(g_df.index, inplace = True)

def SearchMissingData(template_stock: str, stock_ls:list, start:str, end:str):
    """
    通过搜索其他证券和模板证券的分钟级数据的差异来寻找缺失值。

    Args:
        template_stock: str， 模板股票代码，e.g. '510300'
        stock_list: list of str, 需要查找缺失值的股票列表
        start: str, 开始日期，格式为'yyyy.mm.dd' e.g. '2020.01.06'
        end: str，结束日期， 格式为'yyyy.mm.dd' e.g. '2020.07.06', 结束日期必须比开始日期晚 
    
    Return:
        pd.DataFrame, 列名为股票代码，数据为缺失的日期
    """
    loader = GetData()
    result = {}
    close_df = loader.Stock_candle(template_stock, start, end, 1)[['date','close']]
    close_df = close_df.rename(columns = {'close': template_stock})
    close_df.set_index('date', inplace = True)
    filter_trading_time(close_df)

    for s in stock_ls:
        data = loader.Stock_candle(s, start, end, 1)[['date','close']]
        data.set_index('date', inplace = True)
        data = data.rename(columns = {'close': s})
        close_df = close_df.join(data, how = 'left', sort = True)

        idx_nan = np.where(close_df[s].isnull())[0]
        if len(idx_nan) > 0 and len(idx_nan) != idx_nan[-1] + 1:
            idx_start = 0
            for i in range(len(idx_nan)):
                if i != idx_nan[i]:
                    idx_start = i
                    break
            missing_time = idx_nan[idx_start:]
            missing_df = close_df.iloc[missing_time].groupby(pd.Grouper(freq='D')).filter(lambda x: len(x) > 10)
            missing_time = list(missing_df.groupby([missing_df.index.date]).groups.keys())
            if len(missing_time) > 0:
                result[s] = missing_time

    res_df = pd.DataFrame(dict([(k,pd.Series(v)) for k, v in result.items()]))
    return res_df

if __name__ == '__main__': 
    whole_list = ['510330','510050','159919','510310','159949','510500',
        '159915','512500', '159968','515800','512990','512380','512160','512090',
        '159995','512760','515050','159801','512480','512290','159992','512170',
        '512010','159938','515000','515750','159807','515860','159987','515030',
        '515700','159806','512880','512000','512800','512900','159993','159928',
        '512690','515650','159996','510150','512660','512710','515210','512400',
        '515220','159966','159905','159967','510880','515180','515680','515900',
        '159976','515600','159978','511010','511260','159972','510900','159920',
        '513050','513090','513500','518880','159934','159937','518800','159980'
        ]
    start = '2016.01.01'
    end = '2020.07.10'
    template_stock = '510300'
    res_df = SearchMissingData(template_stock, whole_list, start, end)
    res_df.to_csv('StockMissingData.csv', index=False)
