# 内容
训练集为: [2016.01.01 - 2018.01.01, 2017.01.01 - 2019.01.01, 2018.01.01 - 2020.01.01]

测试集为:[2020.01.01 - 2020.07.01]

ETF涵盖：16只，每一类一只流动性最好的

从每个训练集挑选150组参数，挑选的条件为：

1.  参数最大回撤属于前25%
2.  剩余的根据MAR值选最大的

450个参数joint出来的结果被用于测试集的筛选，删选比例为26/29

# 文件:
* KDJ_Arg_{#参数}_{#日期区间}.xlsx： 存储每组参数每次交易的资产分配，仓位，利润表，指标，总结等数据
* para_summary.csv: 存储450个参数共同的
* final_summary.csv: 存储被测试集删去后的参数组
* tota_asset_{#参数}_{#日期区间}.png: 每个参数资产净值的时间序列图