import sys
import time
from PyQt5 import QtCore
from PyQt5 import QtGui
from PyQt5 import QtWidgets


class projector(QtGui.QOpenGLWindow):
    def __init__(self, screen=0):
        super().__init__(flags=(QtCore.Qt.Window |
                                QtCore.Qt.WindowStaysOnTopHint |
                                QtCore.Qt.FramelessWindowHint |
                                QtCore.Qt.WindowOverridesSystemGestures |
                                QtCore.Qt.MaximizeUsingFullscreenGeometryHint |
                                QtCore.Qt.WindowFullscreenButtonHint))
        self.title = 'MEGA scanner PyQT5 projector'
        self.setTitle(self.title)
        self.setCursor(QtCore.Qt.BlankCursor)
        self.screens = QtGui.QGuiApplication.screens()
        self.change_screen(screen)

    def change_screen(self, screen):
        self.selected_screen = self.screens[screen]
        self.setScreen(self.selected_screen)
        self.width = self.selected_screen.size().width()
        self.height = self.selected_screen.size().height()
        print("(W,H) ==", (self.width, self.height))
        self.setGeometry(self.selected_screen.geometry())
        self.showFullScreen()
        self.update()

    def initializeGL(self):  # called automatically with GL context current
        gl = self.gl = self.context().versionFunctions()
        gl.initializeOpenGLFunctions()
        gl.glClearColor(1.0, 1.0, 1.0, 1.0)

    def paintGL(self):  # called automatically with GL context current
        self.gl.glClear(self.gl.GL_COLOR_BUFFER_BIT)

    def update(self):
        super().update()
        QtGui.QGuiApplication.processEvents()

    def black(self):
        self.gl.glClearColor(0.0, 0.0, 0.0, 0.0)
        self.update()

    def white(self):
        self.gl.glClearColor(1.0, 1.0, 1.0, 1.0)
        self.update()


if __name__ == '__main__':
    app = QtGui.QGuiApplication(sys.argv)
    proj = projector(1)

    timer = QtCore.QTimer()
    timer.timeout.connect(lambda: None)
    timer.start(100)

    for i in range(40):
        QtWidgets.QApplication.processEvents()
        time.sleep(0.1)

    time.sleep(5)
    proj.black()

    for i in range(40):
        QtWidgets.QApplication.processEvents()
        time.sleep(0.1)

    # app.exec_()
