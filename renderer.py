import moderngl_window as mglw
from moderngl_window.timers.clock import Timer
import time, numpy
import shaders



class Renderer:
    def __init__(self, render_rate):
        self.scene = None
    
        self.render_rate = render_rate
        self.render_dt_ns = int(1e9 / render_rate)
        self.last_render = time.time_ns()
        self.last_duration = 1
        self.interp_param = None

        # Creates the window instance and activates its context
        self.window = mglw.create_window_from_settings()

        self.program = self.window.ctx.program(
            vertex_shader=shaders.items_vertex_shader_code,
            geometry_shader=shaders.items_geo_shader_code,
            fragment_shader=shaders.items_fragment_shader_code
        )

        self.vaos = [None, None]

    def set_scene(self, scene):
        self.scene = scene
        self.scene.ctx = self.window.ctx
        self.scene.build_buffers()
        self.scene.load_compute_shader()

        self.interp_param = self.window.ctx.buffer(numpy.array([0]).astype("f4"))

        self.vaos[0] = self.window.ctx.vertex_array(
            self.program, [
                (self.scene.vbos[0], '4f4 4x4 4f4', 'in_vert_curr', 'in_col'),
                (self.scene.vbos[1], '4f4 8x4', 'in_vert_prev'),
                (self.interp_param, 'f4 /r', 'in_interp'),
            ]
        )
        self.vaos[1] = self.window.ctx.vertex_array(
            self.program, [
                (self.scene.vbos[1], '4f4 4x4 4f4', 'in_vert_curr', 'in_col'),
                (self.scene.vbos[0], '4f4 8x4', 'in_vert_prev'),
                (self.interp_param, 'f4 /r', 'in_interp'),
            ]
        )

    def run(self):
        while not self.window.is_closing:
            # Update if needed
            time_since_last_update = time.time_ns() - self.scene.last_update
            if self.scene.update_dt_ns < (time_since_last_update - self.scene.last_duration):
                self.scene.update()

            # Render if needed
            time_since_last_render = time.time_ns() - self.last_render
            if self.render_dt_ns < (time_since_last_render - self.last_duration):
                time_begin_render = time.time_ns()

                self.window.clear()
                self.render()
                self.window.swap_buffers()

                self.last_render = time.time_ns()
                self.last_duration = self.last_render - time_begin_render

    def render(self):
        interp_param = (time.time_ns() - self.scene.last_update) / self.scene.update_dt_ns
        self.interp_param.write(numpy.array([interp_param]).astype("f4"))
        self.vaos[self.scene.curr_vbo].render(mode=self.window.ctx.POINTS)