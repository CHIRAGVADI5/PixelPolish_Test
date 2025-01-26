# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'PixelPolishTest.ui'
##
## Created by: Qt User Interface Compiler version 6.8.1
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QFrame, QGridLayout, QLabel,
    QMainWindow, QPushButton, QSizePolicy, QVBoxLayout,
    QWidget)
import Resource.Icons_rc

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(800, 600)
        font = QFont()
        font.setFamilies([u"Segoe UI"])
        font.setPointSize(10)
        MainWindow.setFont(font)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.gridLayout = QGridLayout(self.centralwidget)
        self.gridLayout.setSpacing(0)
        self.gridLayout.setObjectName(u"gridLayout")
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.frame = QFrame(self.centralwidget)
        self.frame.setObjectName(u"frame")
        self.frame.setFrameShape(QFrame.NoFrame)
        self.frame.setFrameShadow(QFrame.Raised)
        self.verticalLayout = QVBoxLayout(self.frame)
        self.verticalLayout.setSpacing(0)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.frame_2 = QFrame(self.frame)
        self.frame_2.setObjectName(u"frame_2")
        self.frame_2.setMaximumSize(QSize(16777215, 40))
        self.frame_2.setFrameShape(QFrame.NoFrame)
        self.frame_2.setFrameShadow(QFrame.Raised)
        self.verticalLayout_2 = QVBoxLayout(self.frame_2)
        self.verticalLayout_2.setSpacing(0)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.frame_4 = QFrame(self.frame_2)
        self.frame_4.setObjectName(u"frame_4")
        self.frame_4.setFrameShape(QFrame.NoFrame)
        self.frame_4.setFrameShadow(QFrame.Raised)
        self.verticalLayout_3 = QVBoxLayout(self.frame_4)
        self.verticalLayout_3.setSpacing(0)
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.verticalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.pushButton = QPushButton(self.frame_4)
        self.pushButton.setObjectName(u"pushButton")
        self.pushButton.setStyleSheet(u"/* Default state settings */\n"
"QPushButton {\n"
"    background-color: white; /* Default background */\n"
"    border: 0px solid transparent; /* No border */\n"
"    border-radius: 7px; /* Rounded corners */\n"
"    color: #333; /* Default text color */\n"
"    font-size: 14px; /* Text size */\n"
"    text-align: center; /* Center alignment */\n"
"    box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1); /* Drop shadow effect */\n"
"}\n"
"\n"
"/* Hover state (when not checked) */\n"
"QPushButton:hover:!checked {\n"
"    background-color: lightgray; /* Hover background color */\n"
"    border-radius: 7px; /* Maintain rounded corners */\n"
"}\n"
"\n"
"/* Checked state (applied immediately without hovering) */\n"
"QPushButton:checked {\n"
"    background-color: #27E2F5; /* Light blue background for checked state */\n"
"    border-radius: 7px; /* Maintain rounded corners */\n"
"    color: white; /* White text for checked state */\n"
"}\n"
"\n"
"/* Pressed state (button pressed and held) */\n"
"QPushButton:pressed {\n"
" "
                        "   background-color: #b2ebf2; /* Darker blue for pressed state */\n"
"}\n"
"\n"
"/* Drop shadow effect to simulate Border.Effect and ContentPresenter.Effect in WPF */\n"
"QPushButton {\n"
"    box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1); /* Outer shadow for the button */\n"
"}\n"
"")
        icon = QIcon()
        icon.addFile(u":/Icons/Icons/open.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.pushButton.setIcon(icon)
        self.pushButton.setIconSize(QSize(30, 30))

        self.verticalLayout_3.addWidget(self.pushButton)


        self.verticalLayout_2.addWidget(self.frame_4, 0, Qt.AlignLeft)


        self.verticalLayout.addWidget(self.frame_2)

        self.frame_3 = QFrame(self.frame)
        self.frame_3.setObjectName(u"frame_3")
        self.frame_3.setFrameShape(QFrame.StyledPanel)
        self.frame_3.setFrameShadow(QFrame.Raised)
        self.verticalLayout_4 = QVBoxLayout(self.frame_3)
        self.verticalLayout_4.setSpacing(0)
        self.verticalLayout_4.setObjectName(u"verticalLayout_4")
        self.verticalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.label = QLabel(self.frame_3)
        self.label.setObjectName(u"label")

        self.verticalLayout_4.addWidget(self.label)


        self.verticalLayout.addWidget(self.frame_3)


        self.gridLayout.addWidget(self.frame, 0, 0, 1, 1)

        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"PixelPolishTest", None))
        self.pushButton.setText("")
        self.label.setText("")
    # retranslateUi

