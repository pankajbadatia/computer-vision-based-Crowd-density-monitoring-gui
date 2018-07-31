# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import sqlite3
import sys
import os.path
import platform
import serialportcontext
from PyQt5.QtCore import Qt, QTimer,QThread,pyqtSignal
from PyQt5.QtGui import QPalette, QColor, QImage, QPixmap 
from PyQt5.QtWidgets import QMainWindow, QWidget, QFrame, QSlider, QHBoxLayout, QPushButton, QVBoxLayout, QAction, QFileDialog, QApplication, QLabel,QGroupBox , QInputDialog,QTableWidget,QTableWidgetItem , QMessageBox
import cv2
import glob
from PyQt5 import QtCore, QtGui, QtWidgets


from PyQt5.QtWidgets import QGridLayout, QComboBox
from PyQt5.QtCore import QSize, QRect  
import time
##########################################################################
data = [0]
data1 =[0]
data2 = [0]
data3 = [0]
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.console
import numpy as np



##########################################################################

import imageio
import os

import seaborn as sns
#import matplotlib.pyplot as plt
import scipy.misc

########################################################################
 #th.changePixmap.connect(label.setPixmap,label1.setPixmap,label2.setPixmap,label3.setPixmap)
class Second(QtGui.QMainWindow):
    def __init__(self, parent=None):
        super(Second, self).__init__(parent)
        
        
class Thread(QThread):
    changePixmap = pyqtSignal(QPixmap)
    changePixmap1 = pyqtSignal(QPixmap)
    changePixmap2 = pyqtSignal(QPixmap)
    changePixmap3 = pyqtSignal(QPixmap)
    
    showdata = pyqtSignal(str)
    showdata1 = pyqtSignal(str)
    showdata2 = pyqtSignal(str)
    showdata3 = pyqtSignal(str)
    datacollect = pyqtSignal(int,int,int,int,name='datacollect')
    def __init__(self, parent=None):
        QThread.__init__(self, parent=parent)

    def run(self):

        cap = cv2.VideoCapture(r'E:/crowdcount/videos/104207/overlay.mp4')
        cap1 = cv2.VideoCapture(r'E:/crowdcount/videos/200608/overlay.mp4')
        cap2 = cv2.VideoCapture(r'E:/crowdcount/videos/200702/overlay.mp4')
        cap3 = cv2.VideoCapture(r'E:/crowdcount/videos/500717/overlay.mp4')

        density = np.load('data1.npy',mmap_mode='r')
        density1 = np.load('data2.npy',mmap_mode='r')
        density2 = np.load('data3.npy',mmap_mode='r')
        density3 = np.load('data4.npy',mmap_mode='r')

        
        count1=0
        while True:
            ret, frame = cap.read()
            ret, frame1 = cap1.read()
            ret, frame2 = cap2.read()
            ret, frame3 = cap3.read()
            
            time.sleep(3)

            if ret is False:
                break
            if count1 == 2999:
                break
            count1 = count1 +1
            
            densecount= int(density[count1]-20)
            densecount1= int(density1[count1])
            densecount2= int(density2[count1])
            densecount3= int(density3[count1])
            
            denser = str(densecount)
            denser1 = str(densecount1)
            denser2 = str(densecount2)
            denser3 = str(densecount3)
            
           # print('count =',count1)
            
            self.showdata.emit(denser)
            self.showdata1.emit(denser1)
            self.showdata2.emit(denser2)
            self.showdata3.emit(denser3)
            
            self.datacollect.emit(densecount,densecount1,densecount2,densecount3)

            rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            convertToQtFormat = QImage(rgbImage.data, rgbImage.shape[1], rgbImage.shape[0], QImage.Format_RGB888)
            convertToQtFormat = QPixmap.fromImage(convertToQtFormat)
            p = convertToQtFormat.scaled(210, 210)
            
            rgbImage1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
            convertToQtFormat1 = QImage(rgbImage1.data, rgbImage1.shape[1], rgbImage1.shape[0], QImage.Format_RGB888)
            convertToQtFormat1 = QPixmap.fromImage(convertToQtFormat1)
            q = convertToQtFormat1.scaled(210, 210)            
            
            rgbImage2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
            convertToQtFormat2 = QImage(rgbImage2.data, rgbImage2.shape[1], rgbImage2.shape[0], QImage.Format_RGB888)
            convertToQtFormat2 = QPixmap.fromImage(convertToQtFormat2)
            r = convertToQtFormat2.scaled(210, 210)            
            
            rgbImage3 = cv2.cvtColor(frame3, cv2.COLOR_BGR2RGB)
            convertToQtFormat3 = QImage(rgbImage3.data, rgbImage3.shape[1], rgbImage3.shape[0], QImage.Format_RGB888)
            convertToQtFormat3 = QPixmap.fromImage(convertToQtFormat3)
            s = convertToQtFormat3.scaled(210, 210)            
            
            
            self.changePixmap.emit(p)
            self.changePixmap1.emit(q)
            self.changePixmap2.emit(r)
            self.changePixmap3.emit(s)
            
        count1=0

        cap.release()
        cap1.release()
        cap2.release()
        cap3.release()
        
        cv2.destroyAllWindows()



class App(QWidget):
    _receive_signal = QtCore.pyqtSignal(str)
    _auto_send_signal = QtCore.pyqtSignal()

    global alist,newdata
    alist = ['']
    newdata = ['']
    def __init__(self):
        super().__init__()
        self.title = 'Crowd Monitoring' 
        self.initUI()
        

    def initUI(self):
        self.setWindowTitle(self.title)
        #self.setGeometry(self.left, self.top, self.width, self.height)
        
        self.resize(1370, 720)

        
        

        #image = cv2.imread(r'C:\Users\Pankaj\Downloads\pank do not throw\WorldExpo\test_video')
        # Combo BOX
        self.groupBox = QGroupBox(self)
        self.groupBox.setGeometry(QtCore.QRect(1150, 20, 200, 180))
        font = QtGui.QFont()
        font.setPointSize(8)
        self.groupBox.setFont(font)
        self.groupBox.setObjectName("groupBox")
######################################## PORT #######################
        self.comboBoxPort = QtWidgets.QComboBox(self.groupBox)
        self.comboBoxPort.setGeometry(QtCore.QRect(100, 25, 75, 21))
        self.comboBoxPort.setObjectName("comboBoxPort")
#####################################label_2 = COM ##################################################################
        self.label_2 = QtWidgets.QLabel(self.groupBox)
        self.label_2.setGeometry(QtCore.QRect(2, 25, 80, 21))
        font = QtGui.QFont()
        font.setPointSize(8)
        self.label_2.setFont(font)
        self.label_2.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_2.setObjectName("label_2")
######################################label = Baud rate ##################################################################
        self.label = QtWidgets.QLabel(self.groupBox)
        self.label.setGeometry(QtCore.QRect(2, 50, 80, 21))
        font = QtGui.QFont()
        font.setPointSize(8)
        self.label.setFont(font)
        self.label.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label.setObjectName("label")
####################################################################
        self.comboBoxBaud = QtWidgets.QComboBox(self.groupBox)
        self.comboBoxBaud.setGeometry(QtCore.QRect(100, 50, 75, 21))
        self.comboBoxBaud.setObjectName("comboBoxBaud")
#################################### label_3 = Parity Bits ######################################################
        self.label_3 = QtWidgets.QLabel(self.groupBox)
        self.label_3.setGeometry(QtCore.QRect(2, 75, 80, 21))
        font = QtGui.QFont()
        font.setPointSize(8)
        self.label_3.setFont(font)
        self.label_3.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_3.setObjectName("label_3")
#########################################################################
        self.comboBoxCheckSum = QtWidgets.QComboBox(self.groupBox)
        self.comboBoxCheckSum.setGeometry(QtCore.QRect(100, 75, 75, 21))
        self.comboBoxCheckSum.setObjectName("comboBoxCheckSum")
##################################### label_4 = Data Bits ##########################################################        
        self.label_4 = QtWidgets.QLabel(self.groupBox)
        self.label_4.setGeometry(QtCore.QRect(2, 100, 80, 21))
        font = QtGui.QFont()
        font.setPointSize(8)
        self.label_4.setFont(font)
        self.label_4.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_4.setObjectName("label_4")
######################################################################
        self.comboBoxBits = QtWidgets.QComboBox(self.groupBox)
        self.comboBoxBits.setGeometry(QtCore.QRect(100, 100, 75, 21))
        self.comboBoxBits.setObjectName("comboBoxBits")
###################################### label_5 = Stop Bits ###########################################################
        self.label_5 = QtWidgets.QLabel(self.groupBox)  
        self.label_5.setGeometry(QtCore.QRect(2, 125, 80, 21))        
        font = QtGui.QFont()
        font.setPointSize(8)
        self.label_5.setFont(font)
        self.label_5.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_5.setObjectName("label_5")
##########################################################################
        self.comboBoxStopBits = QtWidgets.QComboBox(self.groupBox)
        self.comboBoxStopBits.setGeometry(QtCore.QRect(100, 125, 75, 21))
        self.comboBoxStopBits.setObjectName("comboBoxStopBits")
######################################### Open Button ############################################################
        self.pushButtonOpenSerial = QtWidgets.QPushButton(self.groupBox)
        self.pushButtonOpenSerial.setGeometry(QtCore.QRect(50, 150, 125, 21))
        self.pushButtonOpenSerial.setObjectName("pushButtonOpenSerial")
########################################  Send Button  for sending ####################################################
        self.pushButtonSendData = QtWidgets.QPushButton(self)
        self.pushButtonSendData.setGeometry(QtCore.QRect(1300, 450, 50, 75))
        self.pushButtonSendData.setAutoDefault(True)
        self.pushButtonSendData.setObjectName("pushButtonSendData")
########################################  clear Button  for sending ####################################################
        self.pushButtonclearSendData = QtWidgets.QPushButton(self)
        self.pushButtonclearSendData.setGeometry(QtCore.QRect(1300, 375, 50, 75))
        self.pushButtonclearSendData.setObjectName("pushButtonclearSendData")
        self.pushButtonclearSendData.clicked.connect(self.clearsenddata)    
####################################### text  box for receiving  ############################################################
        self.textEditReceived = QtWidgets.QTextEdit(self)
        self.textEditReceived.setGeometry(QtCore.QRect(1100, 210, 250, 150))
###################################################################################################################
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.textEditReceived.sizePolicy().hasHeightForWidth())
        self.textEditReceived.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.textEditReceived.setFont(font)
        self.textEditReceived.setReadOnly(True)
        self.textEditReceived.setObjectName("textEditReceived")
        ################################### Sending text box ###################################
        self.textEditSent = QtWidgets.QTextEdit(self)
        self.textEditSent.setGeometry(QtCore.QRect(1100, 375, 200, 150))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.textEditSent.setFont(font)
        self.textEditSent.setObjectName("textEditSent")

#########################################################################################################################       
        self.groupBox.setTitle("COM Setting")
        self.label_2.setText("COM")
        self.label.setText("Baud rate")
        self.label_3.setText("Parity Bit")
        self.label_4.setText("Data Bit")
        self.label_5.setText("Stop Bit")
        self.pushButtonOpenSerial.setText("Open")
        self.pushButtonSendData.setText("Send")
        self.pushButtonclearSendData.setText("Clear")

        self.textEditSent.setHtml("<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:14pt; font-weight:400; font-style:normal;\">\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; font-family:\'PMingLiU\';\"><br /></p></body></html>")
        #groupBox = QGroupBox("Best Food")
 ######################################### controlling post creating group box ########################################################
        self.groupBox1 = QGroupBox(self)
        self.groupBox1.setGeometry(QtCore.QRect(1150,535, 190, 80))
        font = QtGui.QFont()
        font.setPointSize(8)
        self.groupBox1.setFont(font)
        self.groupBox1.setObjectName("groupBox1")
        self.groupBox1.setTitle("Manual Post control")

####################################### post 1 ############################################################
        self.post1 = QtWidgets.QPushButton(self.groupBox1)
        self.post1.setGeometry(QtCore.QRect(10, 20, 80, 21))
        self.post1.setObjectName("post1")
        self.post1.setText("Post 1")
        self.post1.setCheckable(True)
        
        #self.post1.toggle()
        self.post1.clicked.connect(lambda:self.whichbtn(self.post1))
        self.post1.clicked.connect(self.btnstate1)

        
        
#################################### post 2 ###########################        
        self.post2 = QtWidgets.QPushButton(self.groupBox1)
        self.post2.setGeometry(QtCore.QRect(100, 20, 80, 21))
        self.post2.setObjectName("post2")
        self.post2.setText("Post 2")
        
        self.post2.setCheckable(True)
        #self.post2.toggle()
        self.post2.clicked.connect(lambda:self.whichbtn(self.post2))
        self.post2.clicked.connect(self.btnstate2)
        
        

#################################### post 3 ###########################        
        self.post3 = QtWidgets.QPushButton(self.groupBox1)
        self.post3.setGeometry(QtCore.QRect(10, 45, 80, 21))
        self.post3.setObjectName("post3")
        self.post3.setText("Post 3")
        
        self.post3.setCheckable(True)
        #self.post3.toggle()
        self.post3.clicked.connect(lambda:self.whichbtn(self.post3))
        self.post3.clicked.connect(self.btnstate3)

#################################### post 4 ###########################        
        self.post4 = QtWidgets.QPushButton(self.groupBox1)
        self.post4.setGeometry(QtCore.QRect(100, 45, 80, 21))
        self.post4.setObjectName("post4")
        self.post4.setText("Post 4")
        
        self.post4.setCheckable(True)
        #self.post4.toggle()
        self.post4.clicked.connect(lambda:self.whichbtn(self.post4))
        self.post4.clicked.connect(self.btnstate4)

######################################### group box for database of rfid ########################################################
        self.groupBox3 = QGroupBox(self)
        self.groupBox3.setGeometry(QtCore.QRect(1150,615,190, 80))
        font = QtGui.QFont()
        font.setPointSize(8)
        self.groupBox3.setFont(font)
        self.groupBox3.setObjectName("groupBox1")
        self.groupBox3.setTitle("RFID Database") 
#################################### rfid database ##########################        
        self.rfid = QtWidgets.QPushButton(self.groupBox3)
        self.rfid.setGeometry(QtCore.QRect(10, 20,170, 21))
        self.rfid.setObjectName("rfidtable")
        self.rfid.setText("Open Database")
        self.rfid.clicked.connect(self.showDialog1)     
############################# Rfid Register#########################        
        self.rfiddata = QtWidgets.QPushButton(self.groupBox3)
        self.rfiddata.setGeometry(QtCore.QRect(10, 45,170, 21))
        self.rfiddata.setObjectName("rfidtable")
        self.rfiddata.setText("Register")
        self.rfiddata.clicked.connect(self.showDialog2)    


      
        
############################### Group Box and update of serial com and baud rate setting ###############
        print(platform.system())
        
        if platform.system() == "Windows":
            ports = list()
            for i in range(8):
                ports.append("COM%d" %((i+1)))    
            self.comboBoxPort.addItems(ports)
            print(ports)
            
        if platform.system() == "Linux":
            ports = glob.glob('/dev/tty[A-Za-z]*')
            print(ports)
            self.comboBoxPort.addItems(ports)

        
        bauds = ["50","75","134","110","150","200","300","600","1200","2400","4800","9600","14400","19200","38400","56000","57600",
            "115200"]
        self.comboBoxBaud.addItems(bauds)
        self.comboBoxBaud.setCurrentIndex(len(bauds) - 1)
        
        checks = ["None","Odd","Even","Zero","One"]
        self.comboBoxCheckSum.addItems(checks)
        self.comboBoxCheckSum.setCurrentIndex(len(checks) - 1)
        
        bits = ["4 Bits", "5 Bits","6 Bits", "7 Bits", "8 Bits"]
        self.comboBoxBits.addItems(bits)
        self.comboBoxBits.setCurrentIndex(len(bits) - 1)
        
        stopbits = ["1 Bit","1.5 Bits","2 Bits"];
        self.comboBoxStopBits.addItems(stopbits)
        self.comboBoxStopBits.setCurrentIndex(0)
        
        
        #self._auto_send_signal.connect(self.__auto_send_update__)
        
        port = self.comboBoxPort.currentText()
        baud = int("%s" % self.comboBoxBaud.currentText(), 10)
        self._serial_context_ = serialportcontext.SerialPortContext(port = port,baud = baud)

#        self.lineEditReceivedCounts.setText("0")
#        self.lineEditSentCounts.setText("0")
        self.pushButtonOpenSerial.clicked.connect(self.__open_serial_port__)
       # self.pushButtonClearRecvArea.clicked.connect(self.__clear_recv_area__)
        self.pushButtonSendData.clicked.connect(self.__send_data__)
        self._receive_signal.connect(self.__display_recv_data__)
       # self.pushButtonOpenRecvFile.clicked.connect(self.__save_recv_file__)
        
        
############################# Setting Label for image  #######################################       
        imglabel = QLabel(self)
        imglabel.setGeometry(QtCore.QRect(0, 0, 1100, 700))

        pixmape = QPixmap('crossroads.jpg')
        scaleimage = pixmape.scaledToWidth(1090)
        imglabel.setPixmap(scaleimage)
#
#        
        label = QLabel(self)
        label.move(450, 0)
        label.resize(210, 210)
        label.setStyleSheet("background-color:blue;")
        
        label1 = QLabel(self)        
        label1.move(830, 260)
        label1.resize(210, 210)
        label1.setStyleSheet("background-color:red;")


        label2 = QLabel(self)        
        label2.move(85, 260)
        label2.resize(210, 210)
        label2.setStyleSheet("background-color:white;")

        label3 = QLabel(self)        
        label3.move(450, 500)
        label3.resize(210, 210)
        label3.setStyleSheet("background-color:yellow;")

        self.label21 = QLabel(self)
        self.label21.move(400, 220)
        self.label21.resize(300, 25)
        self.label21.setStyleSheet("background-color:blue;")#525,245
        
        self.label22 = QLabel(self)        
        self.label22.move(700, 220)
        self.label22.resize(25, 300)
        self.label22.setStyleSheet("background-color:red;")

        self.label23 = QLabel(self)        
        self.label23.move(400, 495)
        self.label23.resize(300, 25)
        self.label23.setStyleSheet("background-color:white;")

        self.label24 = QLabel(self)        
        self.label24.move(375, 220)
        self.label24.resize(25, 300)
        self.label24.setStyleSheet("background-color:yellow;")
        
        
        
        labelt = QLabel(self)
        labelt.move(525, 170)
        labelt.resize(50,50)
        labelt.setStyleSheet("background-color:blue;""font: bold 20pt 'Arial'")#525,245
        labelt.setAlignment(QtCore.Qt.AlignCenter)  
        
        labelt21 = QLabel(self)        
        labelt21.move(725, 350)
        labelt21.resize(50,50)
        labelt21.setStyleSheet("background-color:red;""font: bold 20pt 'Arial'")
        labelt21.setAlignment(QtCore.Qt.AlignCenter)  

        labelt22 = QLabel(self)        
        labelt22.move(325, 350)
        labelt22.resize(50,50)
        labelt22.setStyleSheet("background-color:white;""font: bold 20pt 'Arial'")        
        labelt22.setAlignment(QtCore.Qt.AlignCenter)  
        
        
        labelt23 = QLabel(self)        
        labelt23.move(525, 520)
        labelt23.resize(50, 50)
        labelt23.setStyleSheet("background-color:yellow;""font: bold 20pt 'Arial'")#525,445
        labelt23.setAlignment(QtCore.Qt.AlignCenter)  

#############################################################################################3


        
        self.groupBox8 = QGroupBox(self)
        self.groupBox8.setGeometry(QtCore.QRect(388,232, 325, 276))

        font = QtGui.QFont()
        font.setPointSize(8)

        self.groupBox8.setFont(font)
        self.groupBox8.setObjectName("groupBox8")
        
       
        self.groupBox8.setStyleSheet("background-color:transparent; border: 2px solid transparent;")

            
        self.horizontalLayout =  QtGui.QHBoxLayout(self.groupBox8)
        self.horizontalLayout.setGeometry(QtCore.QRect(0, 0, 200, 200))
        

        
        self.win = pg.GraphicsWindow()
        #self.win.resize(200,200)
        self.horizontalLayout.addWidget(self.win)
#        pg.setConfigOption('background', 'w')
#        pg.setConfigOption('foreground', 'k')
        
        self.p6 = self.win.addPlot(title="Crowd Count Plot")
        self.curve = self.p6.plot(pen='b')
        self.curve1 = self.p6.plot(pen='r')
        self.curve2 = self.p6.plot(pen='w')
        self.curve3 = self.p6.plot(pen='y')


   
        
                


#################################################################################################### 
        th = Thread(self)
        th.changePixmap.connect(label.setPixmap)
        th.changePixmap1.connect(label1.setPixmap)
        th.changePixmap2.connect(label2.setPixmap)
        th.changePixmap3.connect(label3.setPixmap)
                
        th.showdata.connect(labelt.setText)
        th.showdata1.connect(labelt21.setText)
        th.showdata2.connect(labelt22.setText)
        th.showdata3.connect(labelt23.setText)
        
        th.datacollect.connect(self.collecteddensity)
        #th.datacollect.connect(self.plotgraph)
        
        th.start()
        
         
################################### Crowd count maths ########################################  



    def on_pushButton_clicked(self):
            self.dialog.show()

    def getmax(self,densitylist):
       # print('density list =',densitylist)
        maximumdensity= max(densitylist)
       # print('maximum density list =', maximumdensity)
        maxposition = np.argmax(densitylist)
       # print('position of maximum',maxposition)
        
        if maxposition == 0:
           # print('inside 0')
        
            self.post1.setChecked(True)
            self.btnstate1()
            time.sleep(1)
            
            self.post2.setChecked(False)
            self.btnstate2()
            time.sleep(1)
            
            self.post3.setChecked(False)
            self.btnstate3()
        
    
            
        if maxposition == 1:
           # print('inside 1')

        
            self.post1.setChecked(False)
            self.btnstate1()
            time.sleep(1)
            self.post2.setChecked(True)
            self.btnstate2()
            time.sleep(1)
            self.post3.setChecked(False)
            self.btnstate3()
        
    
            
        if maxposition == 2:
           # print('inside 2')
        
            self.post1.setChecked(False)
            self.btnstate1()
            time.sleep(1)

            self.post2.setChecked(False)
            self.btnstate2()
            time.sleep(1)

            self.post3.setChecked(True)
            self.btnstate3()
            

            
    def allowed(self):
       # print('############### open all gates ##############')
        
        self.post1.setChecked(False)
        self.btnstate1()
        time.sleep(1)

        self.post2.setChecked(False)
        self.btnstate2()
        time.sleep(1)

        self.post3.setChecked(False)
        self.btnstate3()
        
        #self.post1.toggle()
#        self.post1.setCheckable(True)
        
#        self.post1.clicked.connect(lambda:self.whichbtn(self.post1))
#        self.post1.clicked.connect(self.btnstate1)
        
        
        
        
    def control(self,densitylist):
        self.getmax(densitylist)
        

    def collecteddensity(self,d0,d1,d2,d3):
       # print('d0 =', d0,'d1 =',d1,'d2 =',d2,'d3 =',d3)
        ################################################
        #d0= d0-22
        global curve,data,data1,data2,data3
        
        data.append(int(d0))
        data1.append(int(d1))
        data2.append(int(d2))
        data3.append(int(d3))
        
        self.curve.setData(data)
        self.curve1.setData(data1)
        self.curve2.setData(data2)
        self.curve3.setData(data3)
        
        x0=1              
        x1=1        
        x2=1
        
        referdensity = 60
        
        overalldensity = (d0*x0)+(d1*x1)+(d3*x2)
        #print('##################### overalldensity  #####################',overalldensity)
        #print('###############d2= ',d2)
        currentdensity = (referdensity - d2)
        #print('current space available =',currentdensity)
        densitylist= [d0,d1,d3]
        
        if currentdensity >= 0:
           if currentdensity == 0:
               #print("Zero")
               self.allowed()
               
           else:
               #print("Positive number")
               self.allowed()
        else:
           #print("Negative number")
           
           self.control(densitylist)
           
        
################################### Dialog Box for whole project #########################        

    def showDialog1(self):
        Dialog = QtWidgets.QDialog()
        Dialog.setObjectName("Dialog")
        Dialog.resize(427, 418)
        self.tableWidget = QtWidgets.QTableWidget(Dialog)
        self.tableWidget.setGeometry(QtCore.QRect(50, 20, 331, 321))
        self.tableWidget.setRowCount(10)
        self.tableWidget.setColumnCount(3)
        self.tableWidget.setObjectName("tableWidget")
        self.pushButton7 = QtWidgets.QPushButton(Dialog)
        self.pushButton7.setGeometry(QtCore.QRect(170, 370, 75, 23))
        self.pushButton7.setObjectName("pushButton")
        QtCore.QMetaObject.connectSlotsByName(Dialog)
        Dialog.setWindowTitle("Load Rfid Database")
        self.pushButton7.setText("Load")
        self.pushButton7.clicked.connect(self.loadData)
        Dialog.exec_()
        
        
        
    def showDialog2(self):
        Dialog1 = QtWidgets.QDialog()
        Dialog1.setObjectName("Dialog1")
        Dialog1.resize(500, 250)
        self.label10 = QtWidgets.QLabel(Dialog1)
        self.label10.setGeometry(QtCore.QRect(40, 40, 47, 13))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label10.setFont(font)
        self.label10.setObjectName("label10")
        self.label11 = QtWidgets.QLabel(Dialog1)
        self.label11.setGeometry(QtCore.QRect(40, 100, 131, 13))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label11.setFont(font)
        self.label11.setObjectName("label11")
        self.label12 = QtWidgets.QLabel(Dialog1)
        self.label12.setGeometry(QtCore.QRect(40, 150, 131, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label12.setFont(font)
        self.label12.setObjectName("label12")
        self.registerbutton = QtWidgets.QPushButton(Dialog1)
        self.registerbutton.setGeometry(QtCore.QRect(200, 210, 100, 25))
        self.registerbutton.setObjectName("registerbutton")
        self.lineEdit10 = QtWidgets.QLineEdit(Dialog1)
        self.lineEdit10.setGeometry(QtCore.QRect(180, 30, 250, 30))
        self.lineEdit10.setObjectName("lineEdit10")
        self.lineEdit11 = QtWidgets.QLineEdit(Dialog1)
        self.lineEdit11.setGeometry(QtCore.QRect(180, 90, 250, 30))
        self.lineEdit11.setObjectName("lineEdit11")
        self.lineEdit12 = QtWidgets.QLineEdit(Dialog1)
        self.lineEdit12.setGeometry(QtCore.QRect(180, 150, 250, 30))
        self.lineEdit12.setObjectName("lineEdit12")
        QtCore.QMetaObject.connectSlotsByName(Dialog1)
        Dialog1.setWindowTitle("Registration Form")
        self.label10.setText("Name")
        self.label11.setText("Mobile number")
        self.label12.setText("RFID Tag number")
        self.registerbutton.setText("Register")
        self.registerbutton.clicked.connect(self.insertData)
        Dialog1.exec_()



    def clearsenddata(self):
        self.textEditSent.clear()
        
####################################  Mysql Database function for RFID #####################################        
    def loadData(self):
        connection = sqlite3.connect('rfid.db')
        query = "SELECT * FROM USERS"
        result = connection.execute(query)
        self.tableWidget.setRowCount(0)
        for row_number , row_data in enumerate(result):
            self.tableWidget.insertRow(row_number)
            for column_number, data in enumerate(row_data):
                self.tableWidget.setItem(row_number, column_number,QtWidgets.QTableWidgetItem(str(data)))
        connection.close()


    def insertData(self):
        name = self.lineEdit10.text()
        mobilenumber = self.lineEdit11.text()
        rfidnumber = self.lineEdit12.text()

        connection  = sqlite3.connect("rfid.db")
        connection.execute("INSERT INTO USERS VALUES(?,?,?)",(name,mobilenumber,rfidnumber))
        connection.commit()
        connection.close()
        QMessageBox.about(self, "Registration", "Registration Successful")
        self.lineEdit10.clear()
        self.lineEdit11.clear()
        self.lineEdit12.clear()

######################## RFID Authentication and access to post ################################

    def parsingctrl(self):
       
        totaldata = ''.join(alist)
        print('totaldata =',totaldata)
        commandnumber = totaldata
        self.post1.setCheckable(True)
        if commandnumber.startswith("*a*"):
            print("Rfid from Post A")
            adata = commandnumber.strip("*a*#")
            print('Rfid number =',adata)
          
            checkaccess = self.checkdatabase(adata)
            print('access result =',checkaccess)
            if checkaccess == True:
                print('self.post1.isChecked =',self.post1.isChecked())
                self.post1.toggle()
                self.btnstate1()
                
                



        elif commandnumber.startswith("*b*"):
            print("Rfid from Post B")
            adata = commandnumber.strip("*b*#")
            print('Rfid number =',adata)
            self.post2.setCheckable(True)
            checkaccess = self.checkdatabase(adata)
            print('access result =',checkaccess)
            if checkaccess == True:
                print('self.post1.isChecked =',self.post1.isChecked())
                self.post2.toggle()
                self.btnstate2()
                
                    


        elif commandnumber.startswith("*c*"):
            print("Rfid from Post C")
            adata = commandnumber.strip("*c*#")
            print('Rfid number =',adata)
            self.post3.setCheckable(True)
            checkaccess = self.checkdatabase(adata)
            print('access result =',checkaccess)
            if checkaccess == True:
                print('self.post1.isChecked =',self.post1.isChecked())
                self.post3.toggle()
                self.btnstate3()
                
                    

                                

        elif commandnumber.startswith("*d*"):
            print("Rfid from Post D")
            adata = commandnumber.strip("*d*#")
            print('Rfid number =',adata)
            self.post4.setCheckable(True)
            checkaccess = self.checkdatabase(adata)
            print('access result =',checkaccess)
            if checkaccess == True:
                print('self.post1.isChecked =',self.post1.isChecked())
                self.post4.toggle()
                self.btnstate4()
                
                    

            

    def checkdatabase(self,rfidnumber):
        print("Inside check database rfid =",rfidnumber)
        connection = sqlite3.connect("rfid.db")
        result = connection.execute("SELECT * FROM USERS WHERE rfidnumber = ?",[str(rfidnumber)])
        if(len(result.fetchall()) > 0):
            print("User Found allowing to access the post! ")
            #QMessageBox.about(self, "Warning", "User Found allowing to access the post!")
            access = True
        else:
            print("User Not Found !")
            #QMessageBox.about(self, "Warning", "Invalid Rfid number")
            access = False
        connection.close()
        return access
        
        
#################################### Serial communication function #########################   
        
    def __display_recv_data__(self,data):
        #for l in range(len(data)):
        #   hexstr = "%02X " % ord(str(data[l]))
        #  self.textEditReceived.insertPlainText(hexstr)
        self.textEditReceived.insertPlainText(data)
        #print("gogog",len(data))

        
        global alist
        for character in data:
            if character != " " :
                if character != '\n':
                    if character !=  '\r':
                        alist.append(data)
                        
            if character == '\n' :
                print('alist =', alist)
                newdata = alist
                self.parsingctrl()
                print('newdata =',newdata)
                alist.clear()
                print('alist =', alist)
  
                break
        

                

        for l in range(len(data)):
            #self.textEditReceived.insertPlainText(data[l])  
            sb = self.textEditReceived.verticalScrollBar()
            sb.setValue(sb.maximum())
           # print("test recive", data[l])






            
    def __data_received__(self,data):
        print('recv:%s' % data)
        self._receive_signal.emit(data)

        
        
    def __open_serial_port__(self):
        print("I am here")
        if  self._serial_context_.isRunning():
            print("lets see")
            self._serial_context_.close()
            self.pushButtonOpenSerial.setText(u'open')
            print("open")
        else:
            try:
                
                #currentIndex() will get the number
                portss = self.comboBoxPort.currentText()
                port = self.comboBoxPort.currentText()
                print("the", portss)
                baud = int("%s" % self.comboBoxBaud.currentText(),10)
                self._serial_context_ = serialportcontext.SerialPortContext(port = port,baud = baud)
                #print(self._serial_context_ )
                self._serial_context_ .recall()
                self._serial_context_.registerReceivedCallback(self.__data_received__)
                
                print("4")
                self._serial_context_.open()
                print("5")
                self.pushButtonOpenSerial.setText(u'close')
            except Exception as e:
                print("error")
    
    def __send_data__(self):
        data = str(self.textEditSent.toPlainText()+'\n')
        #print("i m data", data)
        if self._serial_context_.isRunning():
            if len(data) > 0:
                self._serial_context_.send(data, 0)
                print(data)
                
################################### Button state ###########################################               
    def whichbtn(self,b):
      print ("clicked button is "+b.text())

    def btnstate1(self):
      if self.post1.isChecked():
         #print ("button pressed post 1")
         self.post1.setStyleSheet('background-color: red') 
         self.label21.setStyleSheet('background-color: red')
         self.label21.setAlignment(QtCore.Qt.AlignCenter)
         font = QtGui.QFont()
         font.setPointSize(18)
         self.label21.setText("Stop") 
         self.sendpost1()
      else:
          self.post1.setStyleSheet('background-color: green')
          self.label21.setStyleSheet('background-color: green')
          self.label21.setAlignment(QtCore.Qt.AlignCenter)
          font = QtGui.QFont()
          font.setPointSize(18)
          self.label21.setText("Go")
#          self.groupBox21.setStyleSheet('background-color: green')       
          self.sendpost10()
          #print ("button release - post 1")
          
          
    def btnstate2(self):
        if self.post2.isChecked():
            #print("button pressed post2")
            self.post2.setStyleSheet('background-color: red')     
            self.label22.setStyleSheet('background-color: red')
            self.label22.setAlignment(QtCore.Qt.AlignCenter)
            font = QtGui.QFont()
            font.setPointSize(18)
            self.label22.setText("Stop")
            self.sendpost2()
            
        else:
            self.post2.setStyleSheet('background-color: green')
            self.label22.setStyleSheet('background-color: green')
            self.label22.setAlignment(QtCore.Qt.AlignCenter)
            font = QtGui.QFont()
            font.setPointSize(18)
            self.label22.setText("Go") 
            self.sendpost20()
            #print ("button release - post 2")
            
            
    def btnstate3(self):
        if self.post3.isChecked():
            #print("button pressed post3")
            self.post3.setStyleSheet('background-color: red')   
            self.label23.setStyleSheet('background-color: red')
            self.label23.setAlignment(QtCore.Qt.AlignCenter)
            font = QtGui.QFont()
            font.setPointSize(18)
            self.label23.setText("Stop")
            self.sendpost3()
        else:
            self.post3.setStyleSheet('background-color: green')
            self.label23.setStyleSheet('background-color: green')
            self.label23.setAlignment(QtCore.Qt.AlignCenter)
            font = QtGui.QFont()
            font.setPointSize(18)
            self.label23.setText("Go") 
            self.sendpost30()
            #print ("button release - post 3")


    def btnstate4(self):
        if self.post4.isChecked():
            #print("button pressed post4")
            self.post4.setStyleSheet('background-color: red') 
            self.label24.setStyleSheet('background-color: red')
            self.label24.setAlignment(QtCore.Qt.AlignCenter)
            font = QtGui.QFont()
            font.setPointSize(18)
            self.label24.setText("Stop")
            self.sendpost4()
        else:
            self.post4.setStyleSheet('background-color: green')
            self.label24.setStyleSheet('background-color: green')
            self.label24.setAlignment(QtCore.Qt.AlignCenter)
            font = QtGui.QFont()
            font.setPointSize(18)
            self.label24.setText("Go") 
            self.sendpost40()
            #print ("button release - post 4")
        
        
        

############################ Post control ###########################################                
    def sendpost1(self):
        data = str('*2*closep1#'+'\n')
        self.textEditSent.insertPlainText(data)
        print (data)
        if self._serial_context_.isRunning():
            if len(data) > 0:
                self._serial_context_.send(data, 0)


    def sendpost10(self):
        data = str('*1*openp1#'+'\n')
        self.textEditSent.insertPlainText(data)
        print (data)
        if self._serial_context_.isRunning():
            if len(data) > 0:
                self._serial_context_.send(data, 0)

    
    def sendpost2(self):
        data = str('*4*closep2#'+'\n')
        self.textEditSent.insertPlainText(data)
        print (data)
        if self._serial_context_.isRunning():
            if len(data) > 0:
                self._serial_context_.send(data, 0)    
                

    def sendpost20(self):
        data = str('*3*openp2#'+'\n')
        self.textEditSent.insertPlainText(data)
        print (data)
        if self._serial_context_.isRunning():
            if len(data) > 0:
                self._serial_context_.send(data, 0)  
        
    def sendpost3(self):
        data = str('*6*closep3#'+'\n')
        self.textEditSent.insertPlainText(data)
        print (data)
        if self._serial_context_.isRunning():
            if len(data) > 0:
                self._serial_context_.send(data, 0)     
                
                
        
    def sendpost30(self):
        data = str('*5*openp3#'+'\n')
        self.textEditSent.insertPlainText(data)
        print (data)
        if self._serial_context_.isRunning():
            if len(data) > 0:
                self._serial_context_.send(data, 0) 
                

    def sendpost4(self):
        data = str('*8*closep4#'+'\n')
        self.textEditSent.insertPlainText(data)
        print (data)
        if self._serial_context_.isRunning():
            if len(data) > 0:
                self._serial_context_.send(data, 0) 
                
    def sendpost40(self):
        data = str('*7*openp4#'+'\n')
        self.textEditSent.insertPlainText(data)
        print (data)
        if self._serial_context_.isRunning():
            if len(data) > 0:
                self._serial_context_.send(data, 0) 
    


 
                
                
                
                
                
                
if __name__ == "__main__":
    app = QApplication(sys.argv)
    player = App()

    
    player.show()


    sys.exit(app.exec_())


    
    






