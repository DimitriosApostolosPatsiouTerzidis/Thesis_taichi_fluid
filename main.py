import taichi as ti
import math
from particles import *
from colors import *
import numpy as np
import time
from taichi.examples.features.io.export_mesh import alpha
from Spatial_Grid import SpatialGrid
from Constants import *


ti.init(arch=ti.cuda)
START_POS = ti.Vector.field(3, dtype=ti.f32, shape=1)
START_POS[0].xyz = 0.1, 0.6, 0.1


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
pf = fluidPar.field(shape = (NUM_PARTICLES,))
colors = vec3.field(shape = (NUM_PARTICLES,)) #due to taichi gui restrictions particle colors cannot be included in the struct
colors.fill(0.01)

#initialize particle positions
init_particles_pos(pf, START_POS, 0)

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
window = ti.ui.Window("3d-particles", res = (1920, 1080))
canvas = window.get_canvas()
gui = window.get_gui()
canvas.set_background_color((.1, .1, .11))
scene = window.get_scene()
camera = ti.ui.Camera()
camera.position(1.5, 0.4, 1.5)
camera.up(0,1,0)
camera.lookat(0, 0.1, 0)
#frame = 0
pause = 1


while window.running:

    #print(grid.cell_size[0])
    #c_size = float(grid.cell_size[0])
    #print(c_size)
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
    elif color_mode == 3:
        density_color(colors, pf)

    if pause == 1:
        grid.update_grid(pf)
        for s in range(substeps):
            #grid.collision_detection(pf)
            grid.calculate_dens(pf, grid.cell_size[0])
            step(pf)
            #grid.update_grid(pf)

    camera.track_user_inputs(window, movement_speed=0.02, hold_key=ti.ui.LMB)
    scene.set_camera(camera)
    scene.ambient_light((0.8, 0.8, 0.8))
    scene.point_light(pos=(0.5, 2.5, 6.5), color=(1, 1, 1))
    scene.particles(pf.p, per_vertex_color = colors, radius = PARTICLE_RADIUS * 0.7)
    scene.lines(boundaries, indices=box_lines_indices, color=(0.99, 0.68, 0.28), width=1.0)
    canvas.scene(scene)
    #frame += 1
    #print(f"Frame: {frame}")
    #save_frames(frame, 400, 1400)

    window.show()

    if window.is_pressed("r"):
        init_particles_pos(pf, START_POS, 0)
    elif window.is_pressed('n'):
        #print(f"Average Neighbors: {grid.average_n(pf)}")
        print(f"Average Particles per cell: {grid.average_par_count()}")
    elif window.is_pressed('p'):
        print(f"Average Density: {grid.average_density(pf)}")
    elif window.is_pressed(ti.ui.SPACE):
        pause *= -1

