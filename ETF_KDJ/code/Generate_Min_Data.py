#-*-coding:utf-8-*-
# @title: 生成任意分钟线的价格数据，缺失值会以前值填充
# @author: Brian Shan
# @date: 2020.07.10

import pandas as pd
from datetime import time, date, datetime

def generate_any_minute_data(data: pd.DataFrame, cycle:'int > 0') -> pd.DataFrame:
    """
    生成任意分种线的加量数据,缺失的会以前值填充

    
    Args: 
        data: pd.DataFrame, columns为date, open, high, low, close 等等
        cycle: int， 需要的分钟数， 240分钟就是全天数据，480分钟就是2天的数据。
    Return:
        pd.DataFrame
    """
    
    trading_day = data["date"].map(lambda t: t.date()).unique()
    def generate_time_per_date(date):
        res = pd.date_range(datetime.combine(date, time(9,31)), datetime.combine(date, time(11,30)), freq = 'T')
        res = res.union(pd.date_range(datetime.combine(date, time(13,1)), datetime.combine(date, time(15,00)),freq = 'T'))
        return res
    template_index = generate_time_per_date(trading_day[0]).union_many([generate_time_per_date(i) for i in trading_day[1:]])
    res = pd.DataFrame(index = template_index).join(data.set_index('date'), how = 'left')
    res = res[['open', 'high', 'low', 'close']].fillna(method = 'ffill').rename_axis('date').reset_index(drop = False)
    res = res.groupby(res.index // cycle).agg({
                'date': 'last',
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
            })
    if 'product' in data.columns:
        res['product'] = data['product'].iloc[0]
    elif 'symbol' in data.columns:
        res['symbol'] = data['symbol'].iloc[0]
        
    return res

    # 不work的版本。。。
    # res = data[['date', 'open', 'high', 'low', 'close']].groupby(
    #     pd.Grouper(freq = f'{cycle}T',closed = 'right', label = 'right',origin = 'start',key = 'date')
    #         ).agg({
    #             'open': 'first',
    #             'high': 'max',
    #             'low': 'min',
    #             'close': 'last',
    #         })
    # res.dropna(thresh = 3,inplace = True)
    
    # if 'product' in data.columns:
    #     res['product'] = data['product'].iloc[0]
    # elif 'symbol' in data.columns:
    #     res['symbol'] = data['symbol'].iloc[0]
        
    # return res.reset_index(drop = False)