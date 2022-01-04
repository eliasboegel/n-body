import numpy as np
import moderngl_window as mglw
import shaders


class ComputeParticleBase(mglw.WindowConfig):
    gl_version = 4, 3  # Required opengl version
    window_size = 800, 800  # Initial window size
    aspect_ratio = 1.0  # Force viewport aspect ratio (regardless of window size)
    vsync = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.n_bodies = 4096
        m = 0.0001
        m_sun = np.power(10,7)  # Mass of sun, roughly 10^6 times bigger than earth
        # Create the two buffers the compute shader will write and read from
        #array = np.genfromtxt("init.csv", skip_header=1, delimiter=",")[:,1:].astype("f4")
        #initial_state = initial_state[:, [2,3,4,1,5,6,7,0,8,9,10,11]] # Re-order columns to match layout in shader
        initial_state = np.random.uniform(-0.45, 0.45, (self.n_bodies,12)).astype("f4")
        # initial_state = np.ones((self.n_bodies, 12)).astype("f4")
        initial_state[:,2] = 0
        initial_state[:,3] = 0.005 # Radius
        initial_state[:,7] = m # Mass
        initial_state[:,8:11] = np.random.uniform(0.5, 1.0, (self.n_bodies,3)).astype("f4") # Color
        initial_state[:,11] = 1 # Alpha

        ## Primitively generating 'Mars' and 'Earth'
        initial_state[1:3, 1] = 0       # Setting y coord of both mars n earth to be 0
        initial_state[1, 0] = 0.5       # x coord of earth, 0.5 in game distance equals 1 AU
        initial_state[2, 0] = 0.75      # x coord of mars, 1.5 AU
        initial_state[1, 7] = 10        # Mass of earth
        initial_state[2, 7] = 1         # Mass of mars, is roughly 10 times smaller than earth
        initial_state[1:3, 3] = 0.02    # Making radius of earth and mars bigger so its more visible
        initial_state[1:3, 8:11] = np.array([1,1,1])

        ## Generating circular orbits for all the bodies
        pos = initial_state[:,0:3]
        r = np.sqrt(np.sum(pos**2,axis=1))
        v = np.sqrt((m_sun) / r)*np.random.uniform(0.9, 1.1, (self.n_bodies)).astype("f4") # velocity with small randomness
        vx = -v*pos[:,1]/r
        vy = v*pos[:,0]/r
        vz = v*pos[:,2]/r
        initial_state[:,4:7] = np.array([vx, vy, vz]).transpose()/1

        ## Creating the Sun
        initial_state[0, :3] = 0  # Pos
        initial_state[0, 3] = 0.045  # Radius
        initial_state[0,4:7] = 0 # Velocity
        initial_state[0, 7] = m_sun  # Mass
        initial_state[0, 8:11] = np.array([1,0,0])  # Color

        self.buf_particles = self.ctx.buffer(initial_state.flatten())
        self.buf_acc = self.ctx.buffer(reserve=self.n_bodies*self.n_bodies*4*4) # Reserve 3 4-byte floats for each combination of n_bodies

        # Program for drawing the particles / items
        self.program = self.ctx.program(
            vertex_shader=shaders.items_vertex_shader_code,
            geometry_shader=shaders.items_geo_shader_code,
            fragment_shader=shaders.items_fragment_shader_code
        )

        # Particle update shader
        particle_update_compute_parsed = shaders.particle_update_compute.replace("%COMPUTE_SIZE%", "1024")
        particle_update_compute_parsed = particle_update_compute_parsed.replace("%N_BODIES%", str(self.n_bodies))
        self.particle_update_compute = self.ctx.compute_shader(particle_update_compute_parsed)

        # Prepare vertex arrays to drawing particles using the compute shader buffers are input
        # We use 4x4 (padding format) to skip the velocity data (not needed for drawing the particles)
        self.vao_particles = self.ctx.vertex_array(
            self.program, [(self.buf_particles, '4f 4x4 4f', 'in_vert', 'in_col')],
        )

        ## Frame time stuff
        self.frame_times = [0,0]

        ## Keyboard stuff
        self.paused = False


    def render(self, time, frame_time):
        # if frame_time > 0.0001: print(f"Frame time: {frame_time * 1000} ms")
        self.frame_times[0] += frame_time
        self.frame_times[1] += 1

        if self.frame_times[1] >= 200:
            print(f'Frame time = {round(self.frame_times[0] / self.frame_times[1] * 1000, 2)}ms')
            self.frame_times[0], self.frame_times[1] = 0, 0

        dt = 0.0002
        # Bind buffers
        self.buf_particles.bind_to_storage_buffer(0)

        # Calculate the next position of the particles with compute shader
        n_workgroups_update = np.ceil(self.n_bodies / 1024).astype(int)
        self.particle_update_compute['dt'] = frame_time * dt * self.paused
        self.particle_update_compute.run(group_x=n_workgroups_update)

        # Batch draw the particles
        self.vao_particles.render(mode=self.ctx.POINTS)

    def key_event(self, key, action, modifiers):
        # Key presses
        if action == self.wnd.keys.ACTION_PRESS:
            if key == self.wnd.keys.SPACE:
                ## Pausing
                if self.paused == False:
                    self.paused = True
                else:
                    self.paused = False
                print("SPACE key was pressed")

            # Using modifiers (shift and ctrl)

            if key == self.wnd.keys.ENTER:
                ## Retrieving data from buffers into a numpy array
                print("Enter was pressed")
                raw = self.buf_particles.read(size=-1)
                arr = np.frombuffer(raw, dtype="f4")
                print((arr))

            if key == self.wnd.keys.Z and modifiers.ctrl:
                print("ctrl + Z was pressed")

        # Key releases
        elif action == self.wnd.keys.ACTION_RELEASE:
            if key == self.wnd.keys.SPACE:
                print("SPACE key was released")

if __name__ == "__main__":
    ComputeParticleBase.run()