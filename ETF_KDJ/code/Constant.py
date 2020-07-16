#-*-coding:utf-8-*-
# @title: 回测期货的参数
# @author: Brian Shan
# @date: 2020.07.10

import os 

trade_time = (
    (9,45), 
    (10, 0), (10, 15), (10,30), (10,45), 
    (11, 0), (11, 15), (11, 30), 
    (13, 15), (13,30), (13, 45),
    (14, 0), (14,15), (14,30), (14,45),
    (15 ,0)
    )

future_multiplier = {
    "IH": 300, 
    "IF": 300, 
    "IC": 200, 
    "TF": 10000, 
    "T": 10000, 
}


margin_percent = {
    "IH": 0.1, 
    "IF": 0.1, 
    "IC": 0.12, 
    "TF": 0.012, 
    "T": 0.02, 
}

margin_multiplier = 3

log_path = os.path.dirname(os.path.realpath(__file__)) + "\\..\\log"
output_path = os.path.dirname(os.path.realpath(__file__)) + "\\..\\result"