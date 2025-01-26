from PySide6.QtWidgets import QMainWindow
from UI.PixelPolishTest import Ui_MainWindow
class MainView(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)