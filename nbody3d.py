from moderngl_window.conf import settings
import renderer, scene

class nBody:
    def __init__(self):

        settings.WINDOW['gl_version'] = (4, 3)
        settings.WINDOW['vsync'] = False
        settings.WINDOW['size'] = (800, 800)
        settings.WINDOW['aspect_ratio'] = settings.WINDOW['size'][0] / settings.WINDOW['size'][1]

        sim_rate = 2 # Particle updates per second
        render_rate = 60 # Rendering FPS

        self.renderer = renderer.Renderer(render_rate)
        self.scene = scene.SolarSystem(1e-6, sim_rate)
        self.renderer.set_scene(self.scene)
        self.renderer.run()

asd = nBody()