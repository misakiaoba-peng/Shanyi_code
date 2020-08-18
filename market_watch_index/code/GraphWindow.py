#-*-coding:utf-8-*-
# @title: 市场监控指数
# @author: Brian Shan
# @date: 2020.08.17

import pyqtgraph as pg
from typing import *
from PyQt5.QtWidgets import  *
from PyQt5.QtCore import *
from pyqtgraph import PlotWidget, plot
from Constant import future_name_dict


class GraphWindow(QMainWindow):
    def __init__(self, main, selected_ls):
        super().__init__()
        self.main = main
        self.selected_ls = selected_ls
        self.line_ref_mat = []

        self.setGeometry(0, 0, 1200, 800)
        self.showMaximized()
        self.setWindowTitle("多股同列")

        widget = QWidget(self)

        horizontalLayout = QHBoxLayout(widget)
        horizontalLayout.setContentsMargins(0, 0, 0, 0)
        
        pen = pg.mkPen(color=(255, 0, 0))
        
        
        for x in selected_ls:
            groupBox = QGroupBox(self)
            verticalLayout = QVBoxLayout(groupBox)
            new_ls = []
            if x == 0:
                groupBox.setTitle("全商品")
                spacerItem = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
                verticalLayout.addItem(spacerItem)

                spacerItem2 = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
                verticalLayout.addItem(spacerItem2)

                spacerItem3 = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
                verticalLayout.addItem(spacerItem3)

                graph = pg.PlotWidget()
                graph.setTitle("全商品波动率")
                graph.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Expanding)
                verticalLayout.addWidget(graph)
                
                graph.setBackground('w')
                data = main.vol_whl
                line = graph.plot(data, pen = pen)
                new_ls.append(None)
                new_ls.append(None)
                new_ls.append(line)
            else:
                symbol = main.total_ls[x-1]
                groupBox.setTitle(f"{future_name_dict[symbol]}({symbol})")

                graph = PlotWidget()
                graph.setTitle("夏普比率")
                graph.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Expanding)
                verticalLayout.addWidget(graph)
                graph.setBackground('w')
                data = main.sharpe_dict[symbol]
                line = graph.plot(data, pen = pen)
                new_ls.append(line)

                graph2 = PlotWidget()
                graph2.setTitle("动量效率")
                graph2.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Expanding)
                verticalLayout.addWidget(graph2)
                graph2.setBackground('w')
                data = main.momentum_dict[symbol]
                line2 = graph2.plot(data, pen = pen)
                new_ls.append(line2)

                graph3 = PlotWidget()
                graph3.setTitle("波动率")
                graph3.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Expanding)
                verticalLayout.addWidget(graph3)
                graph3.setBackground('w')
                data = main.vol_dict[symbol]
                line3 = graph3.plot(data, pen = pen)
                new_ls.append(line3)

            horizontalLayout.addWidget(groupBox)
            self.line_ref_mat.append(new_ls)

        widget.setLayout(horizontalLayout)
        self.setCentralWidget(widget)

    def updateGraph(self, product_idx, which_graph: str):
        if which_graph == 'sharpe':
            self.line_ref_mat[product_idx][0].setData(self.main.sharpe_dict[self.main.total_ls[product_idx-1]])
        elif which_graph == 'momentum': 
            self.line_ref_mat[product_idx][1].setData(self.main.momentum_dict[self.main.total_ls[product_idx-1]])
        elif which_graph == 'vol':
            self.line_ref_mat[product_idx][2].setData(self.main.vol_dict[self.main.total_ls[product_idx-1]])
        elif which_graph == 'vol_whl': 
            self.line_ref_mat[0][2].setData(self.main.vol_whl)
                
