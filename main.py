"""
main.py — Point d'entrée.
"""
import sys
from PyQt5.QtWidgets import QApplication
from gui.app_logic import AppLogic as MainWindow
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())
