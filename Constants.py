import taichi as ti
import math
#ti.init(arch=ti.gpu)

vec3 = ti.math.vec3 #initializing taichi vec3 type
ivec3 = ti.math.ivec3 #initializing taichi vec3 type

PARTICLE_RADIUS = 0.01
PADDING = PARTICLE_RADIUS/2


#p_V = (4 / 3)  * math.pi * (PARTICLE_RADIUS ** 3)  #particle volume

gravity = -9.81
dt = 0.00006
substeps = 30

#fluid parameters
density = 1000.0    #to calculate mass
p_V = 0.8 * (2*PARTICLE_RADIUS) ** 3    #particle volume
stifness = 8e4      #constant affecting collision forces
p_stifness = 50000     #constant affecting pressure forces
restitution_coef = 0.6 #constant affecting collision forces
dens_cor_coef = 1.1 #density correction coefficient for near boundary particles
viscosity = 0.01   #viscosity coefficient
exponent = 7.0
surface_tension = 0.01









#computational domain boundaries
x_max, y_max, z_max = 1.9, 1.9 , 2.6

#particle space dimensions
x_dim, y_dim, z_dim = 40, 60, 40
NUM_PARTICLES = x_dim * y_dim * z_dim
print(f"Number of particles: {NUM_PARTICLES}")
GRID_SIZE = 50


'''
#GRID_SIZE_VEC = ivec3(50, 100, 50)
CELL_SIZE = PARTICLE_RADIUS * 3
#computational domain boundaries
x_max = CELL_SIZE * GRID_SIZE_VEC.x
y_max = CELL_SIZE * GRID_SIZE_VEC.y
z_max = CELL_SIZE * GRID_SIZE_VEC.z
'''

#max_vel: ti.f32
#max_vel = 0.0

color_mode = 0     #0->default || 1->rng_color || 2->velocity_color || 3->density_color
assert color_mode in [0, 1, 2, 3], (" Invalid color mode\n\t"
                                    "0 - Static Color Mode\n\t"
                                    "1 - Random Color Mode\n\t"
                                    "2 - Velocity Color Mode\n\t"
                                    "3 - Density Color Mode")
