#-*-coding:utf-8-*-
# @title: 市场监控指数
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
	The PyQt5 main window.

	'''
	def __init__(self, length, width, verticle_label, horizontal_label, main):
		super().__init__()
		self.setGeometry(0, 0, 1200, 800)
		self.showMaximized()
		self.setWindowTitle("市场观测指数")

		self.main = main
		self.selected_ls = set()
		self.graphWindow = None

		weidget = QWidget(self)

		horizontalLayout = QHBoxLayout(weidget)
		self.table = QTableWidget(length,width)
		self.table.setHorizontalHeaderLabels(horizontal_label)
		self.table.setVerticalHeaderLabels(verticle_label)
		self.table.itemClicked.connect(self.handleItemClicked)
		self.table.setEditTriggers(QTableWidget.NoEditTriggers)
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

		horizontalLayout.addLayout(verticalLayout)

		self.setCentralWidget(weidget)

		self.threadpool = QThreadPool()
		
	
	def handleItemClicked(self, item):
		if item.checkState() == Qt.Checked:
			self.selected_ls.add(item.row())
		elif item.checkState() == Qt.Unchecked and item.row() in self.selected_ls:
			self.selected_ls.remove(item.row())

	def handleButtonClicked(self):
		if len(self.selected_ls) == 0 or len(self.selected_ls) > 8:
			msg = QMessageBox()
			msg.setWindowTitle('错误')
			msg.setText("请至少勾选1个, 至多勾选8个产品 :)!")
			msg.exec_()
		else:
			self.graphWindow = GraphWindow(self.main, self.selected_ls)


	def updateGraphWindow(self, product_idx, which_graph):
		self.graphWindow.updateGraph(product_idx, which_graph)
			


			
		

	