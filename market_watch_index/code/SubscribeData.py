from threading import Event
import dolphindb as ddb
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import traceback, sys
import logging

class SubscribeData(QRunnable):
    def __init__(self, main):
        super().__init__()
        self.main = main
    
    @pyqtSlot()
    def run(self):
        try:         
            s=ddb.session()
            s.enableStreaming(30001)
            s.subscribe("10.0.40.33", 8505, self.main.handler,"Future_stream","subscription",-1,True)
            # s.subscribe('10.0.60.56', 8503,array.handler,"rtq_stock_stream_quick","action_x",-1,True)
            Event().wait()
        except Event as e:
            logging.error(e)
            