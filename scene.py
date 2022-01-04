import numpy as np
import time
import shaders

class Scene:
    def __init__(self, dt, update_rate):
        self.ctx = None
        self.data = None
        self.curr_vbo = None
        self.vbos = [None, None]
        self.n_bodies = 0
        self.dt = dt
        self.update_dt_ns = int(1e9 / update_rate)
        self.last_update = time.time_ns()
        self.last_duration = 0

    def set_data(self, data):
        # Format:
        # 2D array, 1 row for 1 particle, columns are fields in this order:
        # 3 Values: X, Y, Z position
        # Radius
        # 3 Values: X, Y, Z velocity
        # Mass
        # 4 Values: R, G, B, A color
        self.n_bodies = data.shape[0]
        self.data = data
        self.build_buffers()

    def build_buffers(self):
        if not self.ctx == None:
            self.vbos[0] = self.ctx.buffer(self.data.flatten())
            self.vbos[1] = self.ctx.buffer(self.data.flatten())
            self.curr_vbo = 0

    def load_compute_shader(self):
        # Particle update shader
        particle_update_compute_parsed = shaders.particle_update_compute.replace("%COMPUTE_SIZE%", "1024")
        particle_update_compute_parsed = particle_update_compute_parsed.replace("%N_BODIES%", str(self.n_bodies))
        self.particle_update_compute = self.ctx.compute_shader(particle_update_compute_parsed)

    def update(self):
        time_begin = time.time_ns()

        # Update 
        self.ctx.copy_buffer(self.vbos[self.curr_vbo], self.vbos[not self.curr_vbo])
        self.vbos[self.curr_vbo].bind_to_storage_buffer(0)

        # Calculate the next position of the particles with compute shader
        n_workgroups_update = np.ceil(self.n_bodies / 1024).astype(int)
        self.particle_update_compute['dt'] = self.dt
        self.particle_update_compute.run(group_x=n_workgroups_update)

        # Switch to other vbo, flips between 0 and 1
        self.curr_vbo = not self.curr_vbo


        self.last_update = time.time_ns()
        self.last_duration = self.last_update - time_begin
        

class SolarSystem(Scene):
    def __init__(self, dt, update_rate):
        super().__init__(dt, update_rate)

        self.n_bodies = 2048

        m = 0.00000001
        m_sun = np.power(10,7)  # Mass of sun, roughly 10^6 times bigger than earth
        # Create the two buffers the compute shader will write and read from
        #array = np.genfromtxt("init.csv", skip_header=1, delimiter=",")[:,1:].astype("f4")
        #initial_state = initial_state[:, [2,3,4,1,5,6,7,0,8,9,10,11]] # Re-order columns to match layout in shader
        initial_state = np.random.uniform(-0.95, 0.95, (self.n_bodies,12)).astype("f4")
        # initial_state = np.ones((self.n_bodies, 12)).astype("f4")
        initial_state[:,2] = 0
        initial_state[:,3] = 0.005 # Radius
        initial_state[:,7] = m # Mass
        initial_state[:,8:11] = np.random.uniform(0.5, 1.0, (self.n_bodies,3)).astype("f4") # Color
        initial_state[:,11] = 1 # Alpha

        ## Primitively generating Mars and Earth
        initial_state[1:3, 1] = 0   # Setting y coord of both mars n earth to be 0
        initial_state[1, 0] = 0.5   # x coord of earth, 0.5 in game distance equals 1 AU
        initial_state[2, 0] = 0.75  # x coord of mars, 1.5 AU
        initial_state[1, 7] = 10    # Mass of earth
        initial_state[2, 7] = 1     # Mass of mars, is roughly 10 times smaller than earth
        initial_state[1:3, 3] = 0.02# Making radius of earth and mars bigger so its more visible
        initial_state[1:3, 8:11] = np.array([1,1,1])

        ## Generating circular orbits for all the bodies
        pos = initial_state[:,0:3]
        r = np.sqrt(np.sum(pos**2,axis=1))
        v = np.sqrt((m_sun) / r)
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

        self.set_data(initial_state)