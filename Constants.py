import taichi as ti

vec3 = ti.math.vec3 #initializing taichi vec3 type
ivec3 = ti.math.ivec3 #initializing taichi vec3 type




#simulation parameters
gravity = -9.81
dt = 0.00008
substeps = 30

#fluid parameters
PARTICLE_RADIUS = 0.01
PADDING = PARTICLE_RADIUS/2
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
x_max, y_max, z_max = 1.9, 1.9 , 1.9

#particle space dimensions
x_dim, y_dim, z_dim = 50, 50, 20
NUM_PARTICLES = x_dim * y_dim * z_dim
GRID_SIZE = 40

color_mode = 0     #0->default || 1->rng_color || 2->velocity_color || 3->density_color
assert color_mode in [0, 1, 2, 3], (" Invalid color mode\n\t"
                                    "0 - Static Color Mode\n\t"
                                    "1 - Random Color Mode\n\t"
                                    "2 - Velocity Color Mode\n\t"
                                    "3 - Density Color Mode")
