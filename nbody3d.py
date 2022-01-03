import math
import random
import numpy as np
import moderngl, moderngl_window as mglw
import shaders

class ComputeParticleBase(mglw.WindowConfig):
    gl_version = 4, 3  # Required opengl version
    window_size = 800, 800  # Initial window size
    aspect_ratio = 1.0  # Force viewport aspect ratio (regardless of window size)
    vsync = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Create the two buffers the compute shader will write and read from
        #initial_state = np.genfromtxt("init.csv", skip_header=1, delimiter=",")[:,1:].astype("f4")
        #initial_state = initial_state[:, [2,3,4,1,5,6,7,0,8,9,10,11]] # Re-order columns to match layout in shader
        initial_state = np.random.uniform(-0.9, 0.9, (4096,12)).astype("f4")
        initial_state[:,3] = 0.02 # Radius
        initial_state[:,7] = 100 # Mass
        initial_state[:,8:11] = np.random.uniform(0.5, 1.0, (4096,3)).astype("f4") # Color
        initial_state[:,11] = 1 # Alpha
        initial_state[:,4:6] = 0 # Zero velocity
        #print(initial_state)
        self.n_bodies = initial_state.shape[0]

        self.buf_particles = self.ctx.buffer(initial_state.flatten())
        self.buf_acc = self.ctx.buffer(reserve=self.n_bodies*self.n_bodies*4*4) # Reserve 3 4-byte floats for each combination of n_bodies

        # Program for drawing the particles / items
        self.program = self.ctx.program(
            vertex_shader=shaders.items_vertex_shader_code,
            geometry_shader=shaders.items_geo_shader_code,
            fragment_shader=shaders.items_fragment_shader_code
        )

        ### Load compute shaders
        # Acceleration shader
        particle_acc_compute_parsed = shaders.particle_acc_compute.replace("%COMPUTE_SIZE_X%", "32")
        particle_acc_compute_parsed = particle_acc_compute_parsed.replace("%COMPUTE_SIZE_Y%", "32")
        particle_acc_compute_parsed = particle_acc_compute_parsed.replace("%N_BODIES%", str(self.n_bodies))
        self.particle_acc_compute = self.ctx.compute_shader(particle_acc_compute_parsed)
        # Particle update shader
        particle_update_compute_parsed = shaders.particle_update_compute.replace("%COMPUTE_SIZE%", "1024")
        particle_update_compute_parsed = particle_update_compute_parsed.replace("%N_BODIES%", str(self.n_bodies))
        self.particle_update_compute = self.ctx.compute_shader(particle_update_compute_parsed)

        # Prepare vertex arrays to drawing particles using the compute shader buffers are input
        # We use 4x4 (padding format) to skip the velocity data (not needed for drawing the particles)
        self.vao_particles = self.ctx.vertex_array(
            self.program, [(self.buf_particles, '4f 4x4 4f', 'in_vert', 'in_col')],
        )


    def render(self, time, frame_time):
        if frame_time > 0.0001: print(f"Frame time: {frame_time*1000} ms")

        # Bind buffers
        self.buf_particles.bind_to_storage_buffer(0)
        self.buf_acc.bind_to_storage_buffer(1)

        # Calculate all accelerations
        n_workgroups_acc = np.ceil(self.n_bodies / 32).astype(int)
        self.particle_acc_compute.run(group_x=n_workgroups_acc)

        # Calculate the next position of the particles with compute shader
        n_workgroups_update = np.ceil(self.n_bodies / 1024).astype(int)
        self.particle_update_compute['dt'] = frame_time * 0.01
        self.particle_update_compute.run(group_x=n_workgroups_update)

        # Batch draw the particles
        self.vao_particles.render(mode=self.ctx.POINTS)


if __name__ == "__main__":
    ComputeParticleBase.run()