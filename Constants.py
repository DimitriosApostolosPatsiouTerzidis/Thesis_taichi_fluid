import taichi as ti
#ti.init(arch=ti.gpu)

vec3 = ti.math.vec3 #initializing taichi vec3 type
ivec3 = ti.math.ivec3 #initializing taichi vec3 type


density = 1000.0    #to calculate mass
stifness = 8e3      #constant affecting collision forces
gravity = -9.81
restitution_coef = 0.09 #constant affecting collision forces
dt = 0.00004
substeps = 20

#computational domain boundaries
x_max, y_max, z_max = 2.0, 2.0, 0.8

#particle space dimensions
x_dim, y_dim, z_dim = 30, 60, 30
NUM_PARTICLES = x_dim * y_dim * z_dim
print(f"Number of particles: {NUM_PARTICLES}")
GRID_SIZE = 64
PARTICLE_RADIUS = 0.006
PADDING = PARTICLE_RADIUS/2
#max_vel: ti.f32
#max_vel = 0.0

color_mode = 0   #0->default || 1->rng_color || 2-> velocity_color