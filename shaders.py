items_fragment_shader_code = """
    #version 430
    in vec2 uv;
    in vec4 color;
    out vec4 out_color;
    void main()
    {
        // Calculate the length from the center of the "quad"
        // using texture coordinates discarding fragments
        // further away than 0.5 creating a circle.
        vec2 uv_norm = (uv.xy - vec2(0.5)) * 2; // ranges from -1 to 1 on the quad, (0,0) at the center
        if (length(vec2(0.5, 0.5) - uv.xy) > 0.5)
        {
            discard;
        }
        out_color = color * (1-length(uv_norm));
    }
"""

items_vertex_shader_code = """
    #version 430
    in vec4 in_vert;
    in vec4 in_col;
    out vec4 v_color;
    void main()
    {
        gl_Position = in_vert; // x, y, 0, radius
        v_color = in_col;
    }
"""

# Geometry shader turning the points into triangle strips.
# This can also be done with point sprites.
items_geo_shader_code = """
    #version 330
    layout(points) in;
    layout(triangle_strip, max_vertices=4) out;
    in vec4 v_color[];
    out vec2 uv;
    out vec4 color;
    void main() {
        float radius = gl_in[0].gl_Position.w;
        vec2 pos = gl_in[0].gl_Position.xy;
        // Emit the triangle strip creating a "quad"
        // Lower left
        gl_Position = vec4(pos + vec2(-radius, -radius), 0, 1);
        color = v_color[0];
        uv = vec2(0, 0);
        EmitVertex();
        // upper left
        gl_Position = vec4(pos + vec2(-radius, radius), 0, 1);
        color = v_color[0];
        uv = vec2(0, 1);
        EmitVertex();
        // lower right
        gl_Position = vec4(pos + vec2(radius, -radius), 0, 1);
        color = v_color[0];
        uv = vec2(1, 0);
        EmitVertex();
        // upper right
        gl_Position = vec4(pos + vec2(radius, radius), 0, 1);
        color = v_color[0];
        uv = vec2(1, 1);
        EmitVertex();
        EndPrimitive();
    }
"""

particle_update_compute = """
    #version 430
    #define GROUP_SIZE %COMPUTE_SIZE%
    #define N_BODIES %N_BODIES%
    layout(local_size_x=GROUP_SIZE) in;
    // All values are vec4s because of block alignment rules (keep it simple).
    // We could also declare all values as floats to make it tightly packed.
    // See : https://www.khronos.org/opengl/wiki/Interface_Block_(GLSL)#Memory_layout

    uniform float dt;

    struct Particle
    {
        // Order of these declarations is important
        vec3 pos; // x, y, z (Position)
        float radius;
        vec3 vel; // x, y, z (Velocity)
        float mass;
        vec4 col; // r, g, b, a (Color)
    };
    layout(std430, binding=0) buffer particles_ssbo
    {
        Particle particles[];
    } ParticleSSBO;
    layout(std430, binding=1) buffer acc_ssbo
    {
        vec3 acc[N_BODIES][N_BODIES];
    } AccSSBO;

    void main()
    {
        int id = int(gl_GlobalInvocationID);

        Particle particle = ParticleSSBO.particles[id];

        // Acceleration calculation
        vec3 acc = vec3(0.0f);
        for (int i = 0; i < N_BODIES; ++i) {
            if (i != id) {
                Particle other_particle = ParticleSSBO.particles[i];
                vec3 pos_diff = other_particle.pos - particle.pos;
                float d = length(pos_diff);
                float acc_val = other_particle.mass / (d * d); // Is already acceleration as it has been divided by mass of current body
                acc += normalize(pos_diff) * acc_val;
            }
        }

        // Symplectic Euler integration
        particle.vel += acc * dt;
        particle.pos += particle.vel * dt;
        
        // Write particle
        ParticleSSBO.particles[id] = particle;
    }
"""