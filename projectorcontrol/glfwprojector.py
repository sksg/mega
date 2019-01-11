import sys
import time
import numpy as np
import glfw
import moderngl
import atexit


class projector:
    def __init__(self, screen=0, verbose=False, gl_version=(3, 3)):
        self.verbose = verbose
        self.initialize(gl_version)
        self.set_screen(screen)

    def close(self):
        glfw.destroy_window(self.window)

    def initialize(self, gl_version):
        if not glfw.init():  # multiple of these is okay
            raise RuntimeError('GLFW could not be initialized.')
        atexit.register(glfw.terminate)  # multiple of these is also okay

        # Open window
        if gl_version is not None:
            glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, gl_version[0])
            glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, gl_version[1])
            if gl_version > (3, 2):
                glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
            if gl_version >= (3, 0):
                glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)
        glfw.window_hint(glfw.VISIBLE, True)
        glfw.window_hint(glfw.AUTO_ICONIFY, False)
        self.window = glfw.create_window(800, 600, "Projector", None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError('GLFW could not create window.')
        glfw.make_context_current(self.window)

        # OpenGL context (through QT)
        self.context = moderngl.create_context()

        if self.verbose:
            print("OpenGL vendor:", self.context.info['GL_VENDOR'])
            print("renderer:", self.context.info['GL_RENDERER'])
            print("version:", self.context.info['GL_VERSION'])

        # OpenGL shader program
        self.program = self.context.program(
            vertex_shader="""
            #version 330
            in vec2 pos; // dummy
            out vec2 uv;
            void main(void) {
               vec2 calc_pos = pos;
               calc_pos.x = float(gl_VertexID / 2) * 4.0f - 1.0f;
               calc_pos.y = float(gl_VertexID % 2) * 4.0f - 1.0f;
               gl_Position = vec4(calc_pos, 0, 1);
               uv.x = float(gl_VertexID / 2) * 2.0f;
               uv.y = 1.0f - float(gl_VertexID % 2) * 2.0f;
            }""",
            fragment_shader="""
            #version 330
            uniform sampler2D Texture;
            // uniform vec2 scale;
            in vec2 uv;
            out vec4 color;
            void main(void) {
                color = texture(Texture, uv.st);
                // color = texture(Texture, uv.s * scale.s, uv.t * scale.t);
            }""")

        # OpenGL vertex buffers
        x = np.array([[[0, 0], [0, 0], [0, 0]]])
        vbo = self.context.buffer(x.astype('f4').tobytes())
        self.vao = self.context.simple_vertex_array(self.program, vbo, 'pos')

    def clear(self, r=0.0, g=0.0, b=0.0):
        glfw.make_context_current(self.window)
        self.context.clear(r, g, b)
        glfw.swap_buffers(self.window)

    def set_screen(self, screen):
        self.screen = glfw.get_monitors()[screen]
        vidmode = glfw.get_video_mode(self.screen)
        self.width = vidmode[0].width
        self.height = vidmode[0].height
        rect = 0, 0, self.width, self.height
        glfw.set_window_monitor(self.window, self.screen,
                                *rect, glfw.DONT_CARE)
        if self.verbose:
            print('Window: {} x {}'.format(self.width, self.height))

    def update(self):
        glfw.poll_events()

    def wait(self, sec=1):
        timeout = time.time() + sec
        while time.time() < timeout:
            self.update()
        self.update()

    def draw(self, image):
        return self.draw_sequence(np.broadcast_to(image, (1,) + image.shape))

    def draw_sequence(self, images, delay=0, callbackfn=None):
        if not isinstance(images, np.ndarray):
            raise RuntimeError("Only numpy arrays are supported for images!")

        # prepare images
        N, H, W, C = images.shape
        if C == 1:
            images = np.tile(images, (1, 1, 1, 3))

        # adjust texture coordinates
        if int(self.width) != W or int(self.height) != H:
            print("Screen size: {} x {}".format(self.width, self.height))
            print("Images size: {} x {}".format(W, H))
            raise RuntimeError("For now, only fullscreen images are supported")
        # tex_W = self.width / W
        # tex_H = self.height / H
        # self.uv = [(0.0, tex_H), (0.0, -tex_H), (2 * tex_W, tex_H)]
        # self.vao.bind()
        # self.program.setAttributeArray("vertex_uv", self.uv)

        # set texture
        textures = [self.context.texture((W, H), 3, im.tobytes())
                    for im in images]
        for tex in textures:
            tex.build_mipmaps()

        # draw
        for i, tex in enumerate(textures):

            glfw.make_context_current(self.window)
            tex.use()
            self.vao.render()
            glfw.swap_buffers(self.window)

            if callbackfn is not None:
                callbackfn(i, N)
            self.wait(delay)


if __name__ == '__main__':
    import cv2, os, re
    proj = projector(1, verbose=True)
    proj.clear()
    proj.wait(1)  # allow time for the system to respond

    path = 'test_images'  # this directory should contain .png or .jpg images

    files = [os.path.join(path, f) for f in os.listdir(path)
             if re.match(r'.*\.(png|jpg)', f)]
    if files != []:
        image0 = cv2.imread(files[0])
        images = np.empty((len(files), *image0.shape), dtype=image0.dtype)
        images[0] = image0
        for i, path in enumerate(files[1:], 1):
            images[i] = cv2.imread(path)

    def continue_query(i, N):
        query = input("Enter to continue, or q[uit]... ")
        if len(query) > 0 and query.lower()[0] == "q":
            exit()

    proj.draw_sequence(images, delay=0, callbackfn=continue_query)

    proj.clear()
    proj.wait(1)  # delayed shutdown
