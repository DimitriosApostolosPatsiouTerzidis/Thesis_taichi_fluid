import taichi as ti
ti.init(arch=ti.gpu)

vec3 = ti.math.vec3 #initializing taichi vec3 type
ivec3 = ti.math.ivec3 #initializing taichi vec3 type


density = 1000.0
stifness = 8e3
gravity = -9.81
restitution_coef = 0.09
dt = 0.00004
substeps = 15

#particle space dimensions
x_dim, y_dim, z_dim = 30, 30, 30
NUM_PARTICLES = x_dim * y_dim * z_dim
print(f"Number of particles: {NUM_PARTICLES}")
GRID_SIZE = 32
PARTICLE_RADIUS = 0.008
PADDING = PARTICLE_RADIUS/2

color_mode = 0 #0->default || 1->rng_color