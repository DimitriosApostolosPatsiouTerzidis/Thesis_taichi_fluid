import taichi as ti
#ti.init(arch=ti.gpu)

vec3 = ti.math.vec3 #initializing taichi vec3 type
ivec3 = ti.math.ivec3 #initializing taichi vec3 type


density = 1000.0    #to calculate mass
stifness = 8e3      #constant affecting collision forces
gravity = -9.81
restitution_coef = 0.09 #constant affecting collision forces
dt = 0.00004
substeps = 30

x_max, y_max, z_max = 1.0, 1.0 , 1.0

#particle space dimensions
x_dim, y_dim, z_dim = 30, 30, 30
NUM_PARTICLES = x_dim * y_dim * z_dim
print(f"Number of particles: {NUM_PARTICLES}")
GRID_SIZE = 64

'''
#GRID_SIZE_VEC = ivec3(50, 100, 50)
CELL_SIZE = PARTICLE_RADIUS * 3
#computational domain boundaries
x_max = CELL_SIZE * GRID_SIZE_VEC.x
y_max = CELL_SIZE * GRID_SIZE_VEC.y
z_max = CELL_SIZE * GRID_SIZE_VEC.z
'''

PARTICLE_RADIUS = 0.004
PADDING = PARTICLE_RADIUS/2
#max_vel: ti.f32
#max_vel = 0.0

color_mode = 0     #0->default || 1->rng_color || 2->velocity_color || 3->density_color
assert color_mode in [0, 1, 2, 3], (" Invalid color mode\n\t"
                                    "0 - Static Color Mode\n\t"
                                    "1 - Random Color Mode\n\t"
                                    "2 - Velocity Color Mode\n\t"
                                    "3 - Density Color Mode")
