import taichi as ti
import math
import numpy as np
import time
from Spatial_Grid import SpatialGrid
from Constants import *


ti.init(arch=ti.gpu)
START_POS = ti.Vector.field(3, dtype=ti.f32, shape=1)
START_POS[0].xyz = 0.1, 0.3, 0.1



x_max, y_max, z_max = 1.0, 1.0, 1.0
boundaries = ti.Vector.field(3, dtype=ti.f32, shape = 8)
boundaries[0] = ti.Vector([0.0, 0.0, 0.0])
boundaries[1] = ti.Vector([0.0, y_max, 0.0])
boundaries[2] = ti.Vector([x_max, 0.0, 0.0])
boundaries[3] = ti.Vector([x_max, y_max, 0.0])

boundaries[4] = ti.Vector([0.0, 0.0, z_max])
boundaries[5] = ti.Vector([0.0, y_max, z_max])
boundaries[6] = ti.Vector([x_max, 0.0, z_max])
boundaries[7] = ti.Vector([x_max, y_max, z_max])

box_lines_indices = ti.field(int, shape=(2 * 12))
for i, val in enumerate([0, 1, 0, 2, 1, 3, 2, 3, 4, 5, 4, 6, 5, 7, 6, 7, 0, 4, 1, 5, 2, 6, 3, 7]):
        box_lines_indices[i] = val






@ti.dataclass
class fluidPar:     #particle struct
    id:  ti.i32     #particle id
    f: vec3         #force
    p: vec3         #position
    v: vec3         #velocity
    a: vec3         #acceleration
    color: vec3     #color value
    m: ti.f32       #mass calculated based on given density
    knn: ti.i32     #number of parrticles in neighborhood
    #rig: ti.i32    #if 1 represents rigid body voxel

pf = fluidPar.field(shape = (NUM_PARTICLES,))
colors = vec3.field(shape = (NUM_PARTICLES,)) #due to taichi gui restrictions particle colors cannot be included in the struct




@ti.kernel
def rand_color(cfield: ti.template()):
    for i in range(NUM_PARTICLES):
        #Generate random colors
        r = ti.random(dtype= ti.f32)
        g = ti.random(dtype= ti.f32)
        b = ti.random(dtype= ti.f32)
        #Assign colors to field
        cfield[i].xyz = r, g, b

@ti.kernel
def static_color(cfield: ti.template()):
    # default blue values
    r = 0.1
    g = 0.3
    b = 0.8
    for i in range(NUM_PARTICLES):
        #Assign colors to field
        cfield[i].xyz = r, g, b

@ti.kernel
#initialization of particles based given starting position and dimensions
#if reset = 1 clear all forces and velocities
def init_particles_pos(pfield : ti.template(), start: ti.template(), reset: ti.i32):
    #TODO: replace with one loop in range(NUM_PARTICLES), for optimal parallel
    for i in range(x_dim):
        for j in range(y_dim):
            for k in range(z_dim):
                index = i * y_dim * z_dim + j * z_dim + k
                pfield[index].p.x = start[0].x + i * ((PARTICLE_RADIUS * 2) + PADDING) + (ti.random() * (PARTICLE_RADIUS/4))
                pfield[index].p.y = start[0].y + j * ((PARTICLE_RADIUS * 2) + PADDING) + (ti.random() * (PARTICLE_RADIUS/4))
                pfield[index].p.z = start[0].z + k * ((PARTICLE_RADIUS * 2) + PADDING) + (ti.random() * (PARTICLE_RADIUS/4))
                pfield[index].m = (4/3) * density * math.pi * PARTICLE_RADIUS ** 3
                pfield[index].id = index
                if reset == 1:
                    pfield[index].v = vec3(0.0, 0.0, 0.0)
                    pfield[index].a = vec3(0.0, 0.0, 0.0)
                    pfield[index].f = vec3(0.0, 0.0, 0.0)

    #print(f"mass: {pf[0].mass}")



init_particles_pos(pf, START_POS, 0)

#set color mode
if color_mode == 0:
    static_color(colors)
elif color_mode == 1:
    rand_color(colors)
else:
    print("Invalid color mode\n\t0 - Static Color\n\t1 - Random Color")
    exit(-1)




@ti.func
def update(fp):
    a = fp.f / fp.m # a = F/m
    fp.v += (fp.a + a) * dt / 2.0 #verlet integration
    fp.p += fp.v * dt + 0.5 * a * dt ** 2
    fp.a = a
    return fp

@ti.func
def apply_bc(fp):
    velocity_damping = 0.5
    #friction = 0.99
    x = fp.p.x
    y = fp.p.y
    z = fp.p.z

    if y - PARTICLE_RADIUS < 0:
        fp.p.y = PARTICLE_RADIUS
        fp.v.y *= -velocity_damping
        #fp.v.xz *= friction
    elif y + PARTICLE_RADIUS > y_max:
        fp.p.y = 1.0 - PARTICLE_RADIUS
        fp.v.y *= -velocity_damping

    if z - PARTICLE_RADIUS < 0:
        fp.p.z = PARTICLE_RADIUS
        fp.v.z *= -velocity_damping
    elif z + PARTICLE_RADIUS > z_max:
        fp.p.z = 1.0 - PARTICLE_RADIUS
        fp.v.z *= -velocity_damping

    if x - PARTICLE_RADIUS < 0:
        fp.p.x = PARTICLE_RADIUS
        fp.v.x *= -velocity_damping
    elif x + PARTICLE_RADIUS > x_max:
        fp.p.x = 1.0 - PARTICLE_RADIUS
        fp.v.x *= -velocity_damping
    return fp








grid = SpatialGrid(GRID_SIZE)
grid.update_grid(pf)



@ti.func
def update_particle(fp):
    #tempfp = apply_gravity(fp)
    #tempfp = grid.collision_detection(pfield)
    tempfp = update(fp)
    tempfp = apply_bc(tempfp)
    return tempfp

@ti.kernel
def step(pfield: ti.template()):
    #grid.collision_detection(pfield)
    for i in pfield:
        pfield[i] = update_particle(pfield[i])

def save_frames(frame, start, end):
    if frame > start and frame < end:
        if frame % 5 == 0:
            window.save_image(f"save_data/frame{frame}.jpg")


window = ti.ui.Window("Test for Drawing 3d-particles", (1920, 1080))
canvas = window.get_canvas()
canvas.set_background_color((.1, .1, .11))
scene = window.get_scene()
camera = ti.ui.Camera()
camera.position(1.5, 0.4, 1.5)
camera.up(0,1,0)
camera.lookat(0, 0.1, 0)
frame = 0

while window.running:
    #print(f"P_Gravity: {gravity}")
    #print(f"particle_0 position: {pf[2000].p.x}, {pf[2000].p.y}, {pf[2000].p.z}")
    grid.update_grid(pf)
    for s in range(substeps):
        grid.collision_detection(pf)
        step(pf)
        #grid.update_grid(pf)

    camera.track_user_inputs(window, movement_speed=0.02, hold_key=ti.ui.LMB)
    scene.set_camera(camera)
    scene.ambient_light((0.8, 0.8, 0.8))
    scene.point_light(pos=(0.5, 2.5, 6.5), color=(1, 1, 1))
    scene.particles(pf.p, per_vertex_color = colors, radius = PARTICLE_RADIUS * 0.7)
    scene.lines(boundaries, indices=box_lines_indices, color=(0.99, 0.68, 0.28), width=1.0)
    canvas.scene(scene)
    frame += 1
    #print(f"Frame: {frame}")
    #save_frames(frame, 400, 1400)

    window.show()

    if window.is_pressed(ti.ui.SPACE):
        init_particles_pos(pf, START_POS, 0)
    elif window.is_pressed('n'):
        #print(f"Average Neighbors: {grid.average_n(pf)}")
        print(f"Average Particles per cell: {grid.average_par_count()}")

