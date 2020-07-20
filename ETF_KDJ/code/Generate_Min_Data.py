#-*-coding:utf-8-*-
# @title: 生成任意分钟线的加量
# @author: Brian Shan
# @date: 2020.07.10

import pandas as pd

def generate_any_minute_data(data: pd.DataFrame, cycle:'int > 0') -> pd.DataFrame:
    """
    生成任意分种线的加量数据

    
    Args: 
        data: pd.DataFrame, columns为date, open, high, low, close 等等
        cycle: int， 需要的分钟数
    Return:
        pd.DataFrame
    """

    # # 需要turnover的版本, turnover 在2018.02.01前后出现格式的变换
    # res = pd.DataFrame()
    # 
    # temp_df = data[data['date'] < datetime(2018,2,1)].resample(
    #             f'{cycle}T',closed = 'right', label = 'right', on = 'date'
    #         ).agg({
    #             'open': 'first',
    #             'high': 'max',
    #             'low': 'min',
    #             'close': 'last',
    #             'volume': 'sum',
    #             'turnover': 'sum'
    #         })
    # temp_df.dropna(thresh = 3,inplace = True)
    # res = res.append(temp_df)
    # temp_df = data[data['date'] >= datetime(2018,2,1)].resample(
    #             f'{cycle}T', closed = 'right', label = 'right', on = 'date'
    #         ).agg({
    #             'open': 'first',
    #             'high': 'max',
    #             'low': 'min',
    #             'close': 'last',
    #             'volume': 'sum',
    #             'turnover': 'last'
    #         })
    # temp_df.dropna(thresh = 3,inplace = True)
    # res = res.append(temp_df)
    # res['symbol'] = data['symbol'].iloc[0]
    # return res.reset_index(drop = False)

    # 不需要turnover的版本：
    res = data[['date', 'open', 'high', 'low', 'close']].resample(
                f'{cycle}T',closed = 'right', label = 'right', on = 'date'
            ).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
            })
    res.dropna(thresh = 3,inplace = True)
    
    if 'product' in data.columns:
        res['product'] = data['product'].iloc[0]
    elif 'symbol' in data.columns:
        res['symbol'] = data['symbol'].iloc[0]
        
    return res.reset_index(drop = False)