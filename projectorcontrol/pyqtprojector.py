import sys
import time
from PyQt5 import QtCore
from PyQt5 import QtGui
from PyQt5 import QtWidgets
import numpy as np


class projector:
    def __init__(self, screen=0, qt_app=None):
        self.app = qt_app
        self.app_owner = qt_app is None
        if self.app_owner:  # i.e. projector will control a Qt application
            self.app = QtGui.QGuiApplication(sys.argv)
        self.initialize()
        self.set_screen(screen)

    def initialize(self):
        # QT window
        flags = (
            QtCore.Qt.WindowTransparentForInput |
            QtCore.Qt.BypassGraphicsProxyWidget |
            QtCore.Qt.WindowOverridesSystemGestures |
            QtCore.Qt.WindowStaysOnTopHint |
            QtCore.Qt.MaximizeUsingFullscreenGeometryHint |
            QtCore.Qt.FramelessWindowHint |
            QtCore.Qt.Window
        )
        self.window = QtGui.QWindow()
        self.window.setFlags(flags)
        self.window.setSurfaceType(QtGui.QWindow.OpenGLSurface)
        self.window.setCursor(QtCore.Qt.BlankCursor)
        self.window.show()

        # OpenGL context (through QT)
        self.context = QtGui.QOpenGLContext(self.window)
        self.context.setFormat(self.window.requestedFormat())
        self.context.create()

        # OpenGL version functions (through QT)
        self.context.makeCurrent(self.window)
        self.gl = self.context.versionFunctions()

        # OpenGL shader program
        self.program = QtGui.QOpenGLShaderProgram()
        self.program.addShaderFromSourceCode(QtGui.QOpenGLShader.Vertex, """
            attribute mediump vec2 vertex;
            attribute mediump vec2 vertex_uv;
            varying mediump vec2 uv;
            void main() {
               gl_Position = vec4(vertex, 0, 1);
               uv = vertex_uv;
            }""")
        self.program.addShaderFromSourceCode(QtGui.QOpenGLShader.Fragment, """
            uniform sampler2D texture;
            varying mediump vec2 uv;
            void main() {
                gl_FragColor = texture2D(texture, uv.st);
                // gl_FragColor = vec4(0.5, 0.3, 0.2, 1.0);
            }""")
        self.program.link()

        # OpenGL vertex buffers
        self.program.bind()
        self.vao = QtGui.QOpenGLVertexArrayObject()
        self.vao.create()
        self.vao.bind()
        vertices = [(-1.0, -1.0), (-1.0, 3.0), (3.0, -1.0)]
        uv = [(0.0, 1.0), (0.0, -1.0), (2.0, 1.0)]
        self.program.enableAttributeArray("vertex")
        self.program.setAttributeArray("vertex", vertices)
        self.program.enableAttributeArray("vertex_uv")
        self.program.setAttributeArray("vertex_uv", uv)
        self.program.setUniformValue('texture', 0)
        self.gl.glActiveTexture(self.gl.GL_TEXTURE0)
        self.program.release()
        self.vao.release()

    def clear(self, r=0.0, g=0.0, b=0.0):
        self.context.makeCurrent(self.window)
        self.gl.glClearColor(r, g, b, 1.0)
        self.gl.glClear(self.gl.GL_COLOR_BUFFER_BIT)
        self.context.swapBuffers(self.window)

    def set_screen(self, screen):
        self.screen = self.app.screens()[screen]
        self.window.setWindowStates(QtCore.Qt.WindowNoState)
        self.window.setScreen(self.screen)
        self.window.setGeometry(self.screen.geometry())
        self.window.setWindowStates(QtCore.Qt.WindowFullScreen)
        self.pixel_ratio = self.window.devicePixelRatio()
        self.width = self.window.width() * self.pixel_ratio
        self.height = self.window.height() * self.pixel_ratio
        self.clear()

    def update(self):
        self.app.processEvents()

    def wait(self, sec=1):
        end_time = QtCore.QTime.currentTime().addSecs(sec)
        while QtCore.QTime.currentTime() < end_time:
            self.update()

    def draw(self, image):
        return self.draw_sequence(np.broadcast_to(image, (1,) + image.shape))

    def draw_sequence(self, images, delay=0, callbackfn=None):
        self.context.makeCurrent(self.window)
        self.program.bind()

        if not isinstance(images, np.ndarray):
            raise RuntimeError("Only numpy arrays are supported for images!")

        # prepare image
        N, H, W, C = images.shape
        if C == 1:
            images = np.tile(images, (1, 1, 1, 3))
        images = [QtGui.QImage(im.tobytes(), W, H, QtGui.QImage.Format_RGB888)
                  for im in images]

        # adjust texture coordinates
        tex_W = self.width / W
        tex_H = self.height / H
        uv = [(0.0, tex_H), (0.0, -tex_H), (2 * tex_W, tex_H)]
        self.vao.bind()
        self.program.setAttributeArray("vertex_uv", uv)

        # set texture
        textures = [QtGui.QOpenGLTexture(im) for im in images]

        # draw
        for i, tex in enumerate(textures):
            self.gl.glClearColor(0.5, 0.3, 0.2, 1.0)
            self.gl.glClear(self.gl.GL_COLOR_BUFFER_BIT)

            self.vao.bind()
            tex.bind()
            self.gl.glDrawArrays(self.gl.GL_TRIANGLES, 0, 3)
            self.context.swapBuffers(self.window)

            if callbackfn is not None:
                callbackfn(i, N)
            self.wait(delay)


if __name__ == '__main__':
    from cv2 import imread
    proj = projector(1)
    proj.clear()
    proj.wait(1)  # allow time for the system to respond

    img = imread('test.png')[:, :, ::-1]

    def callbackfn(i, N):
        proj.wait(1)
        proj.clear(i / N, 0, 0)

    proj.draw_sequence(np.broadcast_to(img, (5,) + img.shape), 1, callbackfn)

    proj.clear()
    proj.wait(1)  # delayed shutdown
