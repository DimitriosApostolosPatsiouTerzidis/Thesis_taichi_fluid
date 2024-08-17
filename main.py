import taichi as ti
import math
import numpy as np
import time

from taichi.examples.features.io.export_mesh import alpha

from Spatial_Grid import SpatialGrid
from Constants import *


ti.init(arch=ti.cuda)
START_POS = ti.Vector.field(3, dtype=ti.f32, shape=1)
START_POS[0].xyz = 0.1, 0.6, 0.1



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
colors.fill(0.01)




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

@ti.func
def get_max_vel(pfield: ti.template()):
    max_vel = 0.0
    for i in range(NUM_PARTICLES):
        lin_vel = ti.math.length(pfield[i].v)
        if lin_vel > max_vel:
            max_vel = lin_vel
    return max_vel

@ti.kernel
def velocity_color(cfield: ti.template(), pfield: ti.template()):
    #Intense flickering possibly because only collisions apply and velocity changes are suddent
    #max_vel = get_max_vel(pfield)
    alpha = 0.1 #smoothing variable
    max_vel = 4.0
    r = 0.0
    g = 0.0
    b = 0.0
    for i in range(NUM_PARTICLES):
        linear_vel = ti.math.length(pfield[i].v)
        rel_vel = min((linear_vel / max_vel) , 1.0)
        # if i == 1000:
        #     print(f"rel_vel: {rel_vel} || linear_vel: {linear_vel} || max_vel: {max_vel}")
        if rel_vel <= 0.15:
            r = 0.1
            g = min(rel_vel / 0.15, 0.3)
            b = 0.8
        elif rel_vel <= 0.4:
            r = 0.0
            g = 1.0
            b = 1.0 - (rel_vel - 0.35) / 0.35
        elif rel_vel <= 0.7:
            r = (rel_vel - 0.45) / 0.2
            g = 1.0
            b = 0.0
        elif rel_vel <= 1.0:
            r = 1.0
            g = 1 - (rel_vel - 0.85) / 0.15
            b = 0.0
        prev_r, prev_g, prev_b = cfield[i].xyz
        r = (alpha * r) + (1- alpha) * prev_r
        g = (alpha * g) + (1- alpha) * prev_g
        b = (alpha * b) + (1- alpha) * prev_b
        #g = g + (cfield[i].y * 0.2)
        #b = b + (cfield[i].z * 0.2)
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
        fp.p.y = y_max - PARTICLE_RADIUS
        fp.v.y *= -velocity_damping

    if z - PARTICLE_RADIUS < 0:
        fp.p.z = PARTICLE_RADIUS
        fp.v.z *= -velocity_damping
    elif z + PARTICLE_RADIUS > z_max:
        fp.p.z = z_max - PARTICLE_RADIUS
        fp.v.z *= -velocity_damping

    if x - PARTICLE_RADIUS < 0:
        fp.p.x = PARTICLE_RADIUS
        fp.v.x *= -velocity_damping
    elif x + PARTICLE_RADIUS > x_max:
        fp.p.x = x_max - PARTICLE_RADIUS
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
        #if ti.math.length(pfield[i].v) > max_vel:
        #    max_vel = pfield[i].v

def save_frames(frame, start, end):
    if frame > start and frame < end:
        if frame % 5 == 0:
            window.save_image(f"save_data/frame{frame}.jpg")

#set color mode
if color_mode == 0:
    static_color(colors)
elif color_mode == 1:
    rand_color(colors)






window = ti.ui.Window("3d-particles", res = (1920, 1080))
canvas = window.get_canvas()
gui = window.get_gui()
canvas.set_background_color((.1, .1, .11))
scene = window.get_scene()
camera = ti.ui.Camera()
camera.position(1.5, 0.4, 1.5)
camera.up(0,1,0)
camera.lookat(0, 0.1, 0)
frame = 0




while window.running:



    # with gui.sub_window("Sub Window", x=500, y=500, width=500, height=500):
    #     gui.text("text")

    # gui.begin("Sub Window", x=0.0, y=0.4, width=0.5, height=0.5)
    # x_max = gui.slider_float("x_bound", x_max, 0.7, 2.0)
    # y_max = gui.slider_float("y_bound", y_max, 0.7, 2.0)
    # z_max = gui.slider_float("z_bound", z_max, 0.7, 2.0)
    # PARTICLE_RADIUS = gui.slider_float("Radius", PARTICLE_RADIUS, 0.002, 0.009)
    if color_mode < 2:
        pass
    elif color_mode == 2:
        velocity_color(colors, pf)
    else:
        print(" Invalid color mode\n\t0 - Static Color Mode\n\t1 - Random Color Mode\n\t Velocity Color Mode")
        exit(-1)


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

