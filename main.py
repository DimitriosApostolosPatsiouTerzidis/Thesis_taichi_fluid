import taichi as ti
import math
import numpy as np
import time

ti.init(arch=ti.gpu)
vec3 = ti.math.vec3 #initializing taichi vec3 type
ivec3 = ti.math.ivec3 #initializing taichi vec3 type


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



density = 1000.0
stifness = 8e3
gravity = -9.81
restitution_coef = 0.09
dt = 0.00004
substeps = 15

#particle space dimensions
x_dim, y_dim, z_dim = 35, 40, 35
NUM_PARTICLES = x_dim * y_dim * z_dim
print(f"Number of particles: {NUM_PARTICLES}")
GRID_SIZE = 64
PARTICLE_RADIUS = 0.004
PADDING = PARTICLE_RADIUS/2
START_POS = ti.Vector.field(3, dtype=ti.f32, shape=1)
START_POS[0].xyz = 0.1, 0.5, 0.1
color_mode = 0 #0->default || 1->rng_color


@ti.dataclass
class fluidPar:     #particle struct
    id:  ti.i32     #particle id
    f: vec3         #force
    p: vec3         #position
    v: vec3         #velocity
    a: vec3         #acceleration
    color: vec3     #color value
    m: ti.f32       #mass calculated based on given density
    #rig: ti.i32     #if 1 represents rigid body voxel

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
def apply_gravity(fp):
    fp.f = vec3(0.0, gravity * fp.m, 0.0)
    return fp


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




@ti.data_oriented
class SpatialGrid:

    def __init__(self, grid_size: ti.i32):
        self.grid_size = grid_size
        self.cell_size = 1.0 / grid_size
        self.par_id = ti.field(dtype=ti.i32, shape=(NUM_PARTICLES,), name="par_id")
        self.head_pointer = ti.field(dtype =ti.i32, shape=(grid_size*grid_size*grid_size,), name="head_pointer")    #cells equal to number of grid cells
        self.tail_pointer = ti.field(dtype =ti.i32, shape=(grid_size*grid_size*grid_size,), name="tail_pointer")
        self.cur_pointer = ti.field(dtype =ti.i32, shape=(grid_size*grid_size*grid_size,), name="cur_pointer")
        self.par_count = ti.field(dtype =ti.i32, shape=(grid_size, grid_size, grid_size), name="par_count")
        self.prefix_sum = ti.field(dtype =ti.i32, shape=(grid_size, grid_size, grid_size), name="prefix_sum")
        self.row_sum = ti.field(dtype =ti.i32, shape=(grid_size, grid_size), name="row_sum")
        self.layer_sum = ti.field(dtype =ti.i32, shape=(grid_size,), name="layer_sum")




        print(f"\tEstimated max particles per cell: {int(ti.floor((self.cell_size / (2 * PARTICLE_RADIUS))) ** 3)}")
        #self.id_in_cell = ti.field(int)
        #self.grid = ti.root.dense(ti.ijk, self.grid_size).dynamic(ti.l, 2 * max_particles_per_cell, chunk_size = 32)
        #self.cell = self.grid.dynamic(ti.l, 2 * max_particles_per_cell, chunk_size = 16)
        #self.grid.place(self.id_in_cell)

        print(f"\tParticle Diameter: {PARTICLE_RADIUS * 2}")
        print(f"Particle Mass: {pf[0].m}")
        print(f"\tGrid Size: {grid_size} x {grid_size} x {grid_size} ")
        print(f"\tNumber of Grid Cells: {grid_size**3}")
        print(f"\tCell size: {self.cell_size} x {self.cell_size} x {self.cell_size}")
        #print(f"\tMax particles per cell: {max_particles_per_cell}")
        assert PARTICLE_RADIUS * 2 < self.cell_size  # cell must be able to contain at least one particle


    @ti.func
    def count_particles(self, pfield):
        self.par_count.fill(0)
        for i in range(NUM_PARTICLES):
            cell_id = int(ti.floor(pfield[i].p / self.cell_size))
            #print(cell_id)
            self.par_count[cell_id] += 1


    @ti.func
    def row_layer_count(self):
        self.row_sum.fill(0)
        self.layer_sum.fill(0)
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                sum = 0
                for k in range(self.grid_size):
                    sum += self.par_count[i,j,k]
                self.row_sum[i,j] += sum
                self.layer_sum[i] += self.row_sum[i,j]
                #print(f"Row[{i}][{j}]: {self.row_sum[i,j]}")
                #print(f"Layer[{i}]: {self.layer_sum[i]}")


    @ti.func
    def calculate_prefix_sum(self): #with this sequence of iterations independancy is achieved along with the ability of parallelization
        #self.cur_pointer.fill(0)
        self.prefix_sum[0, 0, 0] = 0
        ti.loop_config(serialize=True)
        for i in range(1, self.grid_size): #cells[i,0,0] init porefix sum sequentially
            self.prefix_sum[i,0,0] = self.prefix_sum[i-1, 0, 0] + self.layer_sum[i-1]

        ti.loop_config(block_dim = self.grid_size)
        for i in range(self.grid_size):
            for j in range(1, self.grid_size): #cells[i,j,0] sequentially accumulate ro_count to prefix
                self.prefix_sum[i,j,0] = self.prefix_sum[i, j-1, 0] + self.row_sum[i,j-1]


        ti.loop_config(block_dim = self.grid_size)
        #ti.loop_config(serialize=True)
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                for k in range(self.grid_size):
                    if k == 0:
                        self.prefix_sum[i,j,k] += self.par_count[i,j,k]
                    else:
                        self.prefix_sum[i,j,k] = self.prefix_sum[i,j,k-1] + self.par_count[i,j,k]
                    linear_idx = i * self.grid_size * self.grid_size + j * self.grid_size + k
                    self.head_pointer[linear_idx] = self.prefix_sum[i,j,k] - self.par_count[i,j,k] #start of cell pointer
                    self.tail_pointer[linear_idx] = self.prefix_sum[i,j,k] #end of cell pointer
                    self.cur_pointer[linear_idx] = self.head_pointer[linear_idx] #current in-cell pointer
                    #print(f"Linear id: {linear_idx}, head_pointer: {self.head_pointer[linear_idx]}, tail_pointer: {self.tail_pointer[linear_idx]}")
                    #print(f"Prefix Sum[{i}][{j}][{k}]: {self.prefix_sum[i,j,k]}")


    @ti.func
    def populate_par_id(self, pfield: ti.template()):
        #ti.loop_config(serialize=True)
        for i in range(NUM_PARTICLES):
            cell_id = ti.cast((ti.floor(pfield[i].p / self.cell_size)), ti.i32)
            linear_idx = cell_id[0] * self.grid_size * self.grid_size + cell_id[1] * self.grid_size + cell_id[2]
            par_location = ti.atomic_add(self.cur_pointer[linear_idx], 1)
            self.par_id[par_location] = i
            #print(f"loc: {par_location} || i: [{i}] || par_id: {self.par_id[par_location]}")
            #print(f"particle id: [{pfield[i].id}][{pfield[i].p}], Cell id: {cell_id}, Linear cell idx:[{linear_idx}],Current pointer: {self.cur_pointer[linear_idx]}  Par_id: {self.par_id[par_location]}")
        #print("===================================")

    @ti.func
    def resolve_collision(self, pfield: ti.template(), i: ti.i32, j: ti.i32):
        #TODO: implement NN
        #TODO: ensure non compresion
        rel_pos = pfield[j].p - pfield[i].p
        dist = ti.sqrt(rel_pos[0]**2 + rel_pos[1]**2 + rel_pos[2]**2)
        delta = -dist + (2 * PARTICLE_RADIUS) #distance with radius accounted
        #normal = rel_pos / delta
        #print(f"normal:{normal}")
        #print(f"p[{i}].v:{pfield[i].v}, p[{j}].v:{pfield[j].v}")
        if delta > 0: # in contact
            #print(f"collision: par[{i}] - par[{j}]")
            normal = rel_pos / dist
            #print(f"{delta}")
            f1 = normal * delta * stifness
            #Damping force
            M = (pfield[i].m * pfield[j].m) / (pfield[i].m + pfield[j].m)
            K = stifness
            C =  (1. / ti.sqrt(1. + (math.pi / ti.log(restitution_coef)) ** 2)) * ti.sqrt(K * M)
            V = (pfield[j].v - pfield[i].v) * normal
            f2 = C * V * normal
            #print(f"V:{V}")
            #print(f"normal:{normal}")
            pfield[i].f += f2 - f1
            pfield[j].f -= f2 - f1
            #print(f"pf[{i}] force: {pf[i].f} || pf[{j}] force: {pf[j].f}")



    @ti.kernel
    def collision_detection(self, pfield: ti.template()):

        for i in range(NUM_PARTICLES):
            #print(f"Particle position: {i} id:{grid.par_id[i]}")

            pfield[i] = apply_gravity(pfield[i])
            p = pfield[i].p
            cell_size = self.cell_size
            cell_id = ti.cast((ti.floor(p / cell_size)), ti.i32)
            #min & max assure boundary conditions
            x_begin = max(cell_id[0]-1, 0)
            X_end = min(cell_id[0]+2, self.grid_size)

            y_begin = max(cell_id[1]-1, 0)
            Y_end = min(cell_id[1]+2, self.grid_size)

            z_begin = max(cell_id[2]-1, 0)
            Z_end = min(cell_id[2]+2, self.grid_size)

            for cell_i in range(x_begin, X_end):
                for cell_j in range(y_begin, Y_end):
                    for cell_k in range(z_begin, Z_end):
                        linear_idx = cell_i * self.grid_size * self.grid_size + cell_j * self.grid_size + cell_k
                        for p_id in range(self.head_pointer[linear_idx], self.tail_pointer[linear_idx]):
                            j = self.par_id[p_id]
                            #print(f"p_id: [{p_id}] i:[{i}] j:[{j}]")

                            if i < j:   #no overlapping iterations
                                #print(f"p_id: [{p_id}] i:[{i}] j:[{j}]")
                                #print(f"neighbors of {i}: {self.par_id[p_id]}")
                                self.resolve_collision(pfield, i, j)

    @ti.kernel
    def update_grid(self, pfield: ti.template()):

        #print(self.par_count[self.grid_size, 0, self.grid_size])
        self.count_particles(pfield)
        self.row_layer_count()
        self.calculate_prefix_sum()
        self.populate_par_id(pfield)
        #self.collision_detection(pfield)




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


window = ti.ui.Window("Test for Drawing 3d-particles", (1280, 720))
canvas = window.get_canvas()
scene = window.get_scene()
camera = ti.ui.Camera()
camera.position(1.5, 0.4, 1.5)
camera.up(0,1,0)
camera.lookat(0, 0.1, 0)

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
    window.show()
    if window.is_pressed(ti.ui.SPACE):
        init_particles_pos(pf, START_POS, 1)

