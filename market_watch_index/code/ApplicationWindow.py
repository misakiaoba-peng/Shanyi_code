#-*-coding:utf-8-*-
# @title: 主窗口
# @author: Brian Shan
# @date: 2020.08.13

from typing import *
import sys
import os
from PyQt5.QtWidgets import  *
from PyQt5.QtCore import *
import numpy as np
from GraphWindow import GraphWindow


class ApplicationWindow(QMainWindow):
	'''
	主窗口

	'''
	def __init__(self, length, width, verticle_label, horizontal_label, main):
		super().__init__()
		self.setGeometry(0, 0, 1200, 800)
		self.setWindowTitle("市场观测指数")

		self.main = main
		self.selected_ls = set()
		self.graphWindow = None
		self.graphWindow_ls = []

		weidget = QWidget(self)

		horizontalLayout = QHBoxLayout(weidget)
		self.table = QTableWidget(length,width)
		self.table.setHorizontalHeaderLabels(horizontal_label)
		self.table.setVerticalHeaderLabels(verticle_label)
		self.table.itemClicked.connect(self.handleItemClicked)
		self.table.setEditTriggers(QTableWidget.NoEditTriggers)
		for i in range(length):
			item = QTableWidgetItem("")
			item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
			item.setCheckState(Qt.Unchecked)
			self.table.setItem(i, 3, item)

		header = self.table.horizontalHeader()
		header.setSectionResizeMode(QHeaderView.ResizeToContents)
		horizontalLayout.addWidget(self.table)

		verticalLayout = QVBoxLayout(self)
		button = QPushButton("多股同列")
		sizePolicy = QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
		sizePolicy.setHorizontalStretch(0)
		sizePolicy.setVerticalStretch(0)
		sizePolicy.setHeightForWidth(button.sizePolicy().hasHeightForWidth())
		button.setSizePolicy(sizePolicy)
		button.clicked.connect(self.handleButtonClicked)

		verticalLayout.addWidget(button)
		spacerItem = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
		verticalLayout.addItem(spacerItem)

		self.label = QLabel()
		verticalLayout.addWidget(self.label)

		horizontalLayout.addLayout(verticalLayout)

		self.setCentralWidget(weidget)
		self.showMaximized()

		# 生成线程池
		self.threadpool = QThreadPool()
		
	def handleItemClicked(self, item):
		"""
		表格被点击后生成的event
		"""
		if item.column() == 3:
			if item.checkState() == Qt.Checked:
				self.selected_ls.add(item.row())
			elif item.checkState() == Qt.Unchecked and item.row() in self.selected_ls:
				self.selected_ls.remove(item.row())

	def handleButtonClicked(self):
		"""
		按键被点击后生成的event
		"""
		if len(self.selected_ls) == 0 or len(self.selected_ls) > 8:
			msg = QMessageBox()
			msg.setWindowTitle('错误')
			msg.setText("请至少勾选1个, 至多勾选8个产品 :)!")
			msg.exec_()
		else:
			self.graphWindow = GraphWindow(self.main, self.selected_ls)
			self.graphWindow.showMaximized()
			
	def updateGraphWindow(self, product_idx, which_graph):
		"""
        更新所选的图

        Args：
            product_idx: int, 要更新图的证券的标号
            which_graph：str, 图的种类，目前支持：sharpe, momentum, vol, vol_whl
        """
		self.graphWindow.updateGraph(product_idx, which_graph)
			


			
		

	