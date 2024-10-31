from time import perf_counter

import taichi as ti
import math
from particles import *
from colors import *
import numpy as np
import time
from Spatial_Grid import SpatialGrid
from Constants import *
import timeit
import psutil
import os

ti.init(arch=ti.cuda)
#ti.init(arch=ti.cpu)
#ti.init(arch=ti.cpu, cpu_max_num_threads=1)


START_POS = ti.Vector.field(3, dtype=ti.f32, shape=1)
#START_POS[0].xyz = 0.01, 0.2, 0.5 #middle 60K

START_POS[0].xyz = 0.01, 0.01, 0.01

# For demonstration purposes, these are placeholder sizes
#edge_table = ti.field(dtype=ti.i32, shape=(256))  # Example size, adjust as per your actual table size
#tri_table = ti.field(dtype=ti.i32, shape=(256, 16))  # Adjust this based on your actual triangle table


#initialize_tables(edge_table, tri_table)  # Call to initialize the tables



#initialize computational domain boundarie lines for visualization
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

#initialize particle field
pf = Fluidpar.field(shape = (NUM_PARTICLES,))
colors = vec3.field(shape = (NUM_PARTICLES,)) #due to taichi gui restrictions particle colors cannot be included in the struct
colors.fill(0.01)

#initialize particle positions
init_particles_pos(pf, START_POS, 0)
print(f"\tNumber of particles: {NUM_PARTICLES}")
print(f"\tParticle mass: {pf[0].m}")

#initialize spatial grid
grid = SpatialGrid(GRID_SIZE)
grid.update_grid(pf)

#save frames
def save_frames(frame, start, end):
    if frame > start and frame < end:
        if frame % 5 == 0:
            window.save_image(f"save_data/frame{frame}.jpg")

#set color mode
if color_mode == 0:
    static_color(colors)
elif color_mode == 1:
    rand_color(colors)



#initialize window
window = ti.ui.Window("3d-particles", res = (1920, 1080), fps_limit=240)
canvas = window.get_canvas()
gui = window.get_gui()
canvas.set_background_color((.1, .1, .11))
scene = window.get_scene()
camera = ti.ui.Camera()
camera.position(3.1, 1.8, 2.3)
camera.up(0,1,0)
camera.lookat(0, 0.0, 0)
frame = 0
pause = 1
base_rad = 0.0
for i in range(3):
    if base_rad < grid.cell_size[i]:
        base_rad = grid.cell_size[i]

fixed_radius = base_rad * 1.
color_flag = 0

def get_memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / (1024 * 1024)  # Convert from bytes to MB


#scalar_field = ti.field(dtype=ti.f32, shape=(32,32,32))

while window.running:

    #start_time = time.perf_counter()
    if pause == 1:
        #grid.update_grid(pf)
        for s in range(substeps):
            grid.update_grid(pf)
            #grid.collision_detection(pf)
            grid.calculate_forces(pf, fixed_radius)
            #grid.calculate_dens(pf, fixed_radius)
            #grid.calculate_pressure(pf)
            step(pf)
            #grid.update_grid(pf)
        if color_mode == 2:
            velocity_color(colors, pf)
        elif color_mode == 3:
            density_color(colors, pf)
        elif color_mode == 4:
            pressure_color(colors, pf)



    camera.track_user_inputs(window, movement_speed=0.02, hold_key=ti.ui.LMB)
    scene.set_camera(camera)
    scene.ambient_light((0.8, 0.8, 0.8))
    scene.point_light(pos=(0.5, 2.5, 6.5), color=(1, 1, 1))
    scene.particles(pf.p, per_vertex_color=colors, radius=PARTICLE_RADIUS * 0.7)
    scene.lines(boundaries, indices=box_lines_indices, color=(0.99, 0.68, 0.28), width=1.0)
    canvas.scene(scene)
    frame += 1
    #print(f"Frame: {frame}")
    #save_frames(frame, 400, 1400)

    #current_memory = get_memory_usage()
    #print(f"Memory usage: {current_memory} MB")

    window.show()

    #end_time = time.perf_counter()
    #elapsed_time = end_time - start_time
    #print(f"Elapsed time: {elapsed_time}")

    if window.is_pressed("r"):
        init_particles_pos(pf, START_POS, 1)
    elif window.is_pressed('n'):
        #print(f"Average Neighbors: {grid.average_n(pf)}")
        print(f"Average Fixed Radius Nearest Neighbors: {grid.average_frnn(pf)}")
    elif window.is_pressed('p'):
        print(f"Average Density: {grid.average_density(pf)}")
    elif window.is_pressed('o'):
        print(f"Average Particles per cell: {grid.average_par_count()}")
    elif window.is_pressed('f'):
        print(f"Pressure[0]: {pf[0].pressure}")
    elif window.is_pressed(ti.ui.SPACE):
        pause *= -1

