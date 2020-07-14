#-*-coding:utf-8-*-
# @title: ETF风险平价
# @author: Brian Shan
# @date: 2020.07.08
# 注： 交易时间为9点30 - 11点30， 1点 - 3点
# ToDo: 目前是从dolphindb_data里取一分钟的数据重新组合成15分钟
#       原因是15分钟的数据目前有问题，等待技术部门更新

# %%
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from matplotlib import pyplot as plt
from typing import Set
from typing import List

from Dolphindb_Data import GetData
from Generate_Min_Data import generate_any_minute_data
from Constant import trade_time, future_multiplier, margin_percent, margin_multiplier, output_path


class ETF_RiskParity(object):
    