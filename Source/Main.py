import sys
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer
from Controller.main_controller import MainController


def main():
    app = QApplication(sys.argv)
    # Initialize and run MainController
    controller = MainController()
    controller.run()


    # Handle application close gracefully
    def handle_exit():
        app.quit()  # Exit application

    # Connect application exit event
    app.aboutToQuit.connect(handle_exit)

    # Start the application event loop
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
