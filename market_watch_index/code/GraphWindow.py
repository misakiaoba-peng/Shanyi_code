#-*-coding:utf-8-*-
# @title: 图像窗口
# @author: Brian Shan
# @date: 2020.08.17

import pyqtgraph as pg
from typing import *
from PyQt5.QtWidgets import  *
from PyQt5.QtCore import *
from pyqtgraph import PlotWidget, plot
from Constant import future_name_dict


class GraphWindow(QWidget):
    def __init__(self, main: QObject, selected_ls: tuple):
        super().__init__()
        self.main = main # 主程序，market_main_index
        self.selected_ls = selected_ls # 挑选需要画图的集合
        self.line_ref_mat = [] # 画图线的矩阵用于更新每张图的数据

        self.setGeometry(0, 0, 1200, 800)
        self.setWindowTitle("多股同列")

        horizontalLayout = QHBoxLayout(self)
        horizontalLayout.setContentsMargins(0, 0, 0, 0)
        
        pen = pg.mkPen(color=(255, 0, 0))
        self.order_dict = {} # 存证券的顺序
        idx = 0
        for x in selected_ls:
            self.order_dict[x] = idx
            idx += 1

            groupBox = QGroupBox(self)
            verticalLayout = QVBoxLayout(groupBox)
            new_ls = []
            if x == 0:
                groupBox.setTitle("全商品")

                graph = pg.PlotWidget()
                graph.showGrid(x=True, y=True)
                graph.setTitle("全商品波动率")
                graph.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Expanding)
                verticalLayout.addWidget(graph)
                
                spacerItem = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
                verticalLayout.addItem(spacerItem)

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
                graph.showGrid(x=True, y=True)
                graph.setTitle("夏普比率")
                graph.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Expanding)
                verticalLayout.addWidget(graph)
                graph.setBackground('w')
                data = main.sharpe_dict[symbol]
                line = graph.plot(data, pen = pen)
                new_ls.append(line)

                graph2 = PlotWidget()
                graph2.showGrid(x=True, y=True)
                graph2.setTitle("动量效率")
                graph2.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Expanding)
                verticalLayout.addWidget(graph2)
                graph2.setBackground('w')
                data = main.momentum_dict[symbol]
                line2 = graph2.plot(data, pen = pen)
                new_ls.append(line2)

                graph3 = PlotWidget()
                graph3.showGrid(x=True, y=True)
                graph3.setTitle("波动率")
                graph3.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Expanding)
                verticalLayout.addWidget(graph3)
                graph3.setBackground('w')
                data = main.vol_dict[symbol]
                line3 = graph3.plot(data, pen = pen)
                new_ls.append(line3)

            horizontalLayout.addWidget(groupBox)
            self.line_ref_mat.append(new_ls)

        self.setLayout(horizontalLayout)
        

    def updateGraph(self, product_idx:int, which_graph: str) -> None:
        """
        更新所选的图

        Args：
            product_idx: int, 要更新图的证券的标号
            which_graph：str, 图的种类，目前支持：sharpe, momentum, vol, vol_whl
        """
        if which_graph == 'sharpe':
            self.line_ref_mat[self.order_dict[product_idx]][0].setData(self.main.sharpe_dict[self.main.total_ls[product_idx-1]])
        elif which_graph == 'momentum': 
            self.line_ref_mat[self.order_dict[product_idx]][1].setData(self.main.momentum_dict[self.main.total_ls[product_idx-1]])
        elif which_graph == 'vol':
            self.line_ref_mat[self.order_dict[product_idx]][2].setData(self.main.vol_dict[self.main.total_ls[product_idx-1]])
        elif which_graph == 'vol_whl': 
            self.line_ref_mat[0][2].setData(self.main.vol_whl)
                
