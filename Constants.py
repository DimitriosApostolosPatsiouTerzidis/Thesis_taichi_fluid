import taichi as ti
import math

vec3 = ti.math.vec3 #initializing taichi vec3 type
ivec3 = ti.math.ivec3 #initializing taichi vec3 type



#simulation parameters
gravity = -9.81
#gravity = 0.
dt = 0.00008  #time step to be 0.00008
substeps = 30



#fluid parameters
PARTICLE_RADIUS = 0.010 #to be 0.01
PADDING = PARTICLE_RADIUS * 0.0
density = 1000.0   #to calculate mass
p_V = 0.8 * (2*PARTICLE_RADIUS) ** 3    #particle volume
#p_V = 4/3 * math.pi * (PARTICLE_RADIUS) ** 3    #particle volume
#stifness = 8e4      #constant affecting collision forces
p_stifness = 50000     #constant affecting pressure forces
p_stifness_near = p_stifness * 1.0  #constant affecting near pressure forces
#restitution_coef = 0.6 #constant affecting collision forces
dens_cor_coef = 1.0 #density correction coefficient for near boundary particles TO BE 1.1
viscosity = 0.01    #viscosity coefficient TO BE 0.01
exponent = 7.0
#surface_tension = 0.001 ######## 0.01

#computational domain boundaries
x_max, y_max, z_max = 1.5, 1.5 , 1.5

#particle space dimaensions
#x_dim, y_dim, z_dim = 75, 40, 20

#x_dim, y_dim, z_dim = 20, 25, 20   #10K particles
#x_dim, y_dim, z_dim = 25, 25, 20   #25K particles
x_dim, y_dim, z_dim = 50, 50, 20   #50K particles
#x_dim, y_dim, z_dim = 50, 50, 40   #100K particles
#x_dim, y_dim, z_dim = 58, 60, 58   #200K particles

NUM_PARTICLES = x_dim * y_dim * z_dim
GRID_SIZE = 40 #40


color_mode = 4       #0->default || 1->rng_color || 2->velocity_color || 3->density_color || 4->pressure_color


'''assert render_mode in [0, 1], (" Invalid render mode\n\t"
                                    "0 - Particle Render Mode\n\t"
                                    "1 - Voxel Render Mode\n\t")'''
assert color_mode in [0, 1, 2, 3, 4], (" Invalid color mode\n\t"
                                    "0 - Static Color Mode\n\t"
                                    "1 - Random Color Mode\n\t"
                                    "2 - Velocity Color Mode\n\t"
                                    "3 - Density Color Mode\n\t"
                                    "4 - Pressure Color Mode\n\t")
