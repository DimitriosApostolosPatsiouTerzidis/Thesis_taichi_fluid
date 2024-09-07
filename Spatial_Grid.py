import taichi as ti
import math
from Constants import *
#from main import apply_gravity

ti.init(arch=ti.cuda)




@ti.func
def apply_gravity(fp):
    fp.f = vec3(0.0, gravity * fp.m, 0.0)
    return fp




@ti.data_oriented
class SpatialGrid:
    #TODO: make gridsize self assigned based on computational domain size
    def __init__(self, grid_size: ti.i32):
        self.grid_size = grid_size
        self.cell_size = vec3(x_max / grid_size, y_max / grid_size, z_max / grid_size)
        self.par_id = ti.field(dtype=ti.i32, shape=(NUM_PARTICLES,), name="par_id")
        self.head_pointer = ti.field(dtype =ti.i32, shape=(grid_size*grid_size*grid_size,), name="head_pointer")    #cells equal to number of grid cells
        self.tail_pointer = ti.field(dtype =ti.i32, shape=(grid_size*grid_size*grid_size,), name="tail_pointer")
        self.cur_pointer = ti.field(dtype =ti.i32, shape=(grid_size*grid_size*grid_size,), name="cur_pointer")
        self.par_count = ti.field(dtype =ti.i32, shape=(grid_size, grid_size, grid_size), name="par_count")
        self.prefix_sum = ti.field(dtype =ti.i32, shape=(grid_size, grid_size, grid_size), name="prefix_sum")
        self.row_sum = ti.field(dtype =ti.i32, shape=(grid_size, grid_size), name="row_sum")
        self.layer_sum = ti.field(dtype =ti.i32, shape=(grid_size,), name="layer_sum")




        #print(f"\tEstimated max particles per cell: {int(ti.floor((self.cell_size / (2 * PARTICLE_RADIUS))) ** 3)}")
        #self.id_in_cell = ti.field(int)
        #self.grid = ti.root.dense(ti.ijk, self.grid_size).dynamic(ti.l, 2 * max_particles_per_cell, chunk_size = 32)
        #self.cell = self.grid.dynamic(ti.l, 2 * max_particles_per_cell, chunk_size = 16)
        #self.grid.place(self.id_in_cell)

        print(f"\tParticle Diameter: {PARTICLE_RADIUS * 2}")
        #print(f"Particle Mass: {pf[0].m}")
        print(f"\tGrid Size: {grid_size} x {grid_size} x {grid_size} ")
        print(f"\tNumber of Grid Cells: {grid_size**3}")
        print(f"\tCell size: {self.cell_size}")
        #print(f"\tMax particles per cell: {max_particles_per_cell}")
        assert PARTICLE_RADIUS * 2 < self.cell_size[0]  # cell must be able to contain at least one particle
        assert PARTICLE_RADIUS * 2 < self.cell_size[1]  # cell must be able to contain at least one particle
        assert PARTICLE_RADIUS * 2 < self.cell_size[2]  # cell must be able to contain at least one particle

    #defines the cell id of a particle
    @ti.func
    def get_cell_id(self, pos: ti.template()):
        cell_id = ivec3(0.0, 0.0, 0.0)
        cell_id[0] = int(pos[0] / self.cell_size[0])
        cell_id[1] = int(pos[1] / self.cell_size[1])
        cell_id[2] = int(pos[2] / self.cell_size[2])
        return cell_id

    @ti.func
    def count_particles(self, pfield):
        self.par_count.fill(0)
        for i in range(NUM_PARTICLES):
            cell_id = self.get_cell_id(pfield[i].p)
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

        #ti.loop_config(block_dim = self.grid_size)
        for i in range(self.grid_size):
            for j in range(1, self.grid_size): #cells[i,j,0] sequentially accumulate ro_count to prefix
                self.prefix_sum[i,j,0] = self.prefix_sum[i, j-1, 0] + self.row_sum[i,j-1]


        #ti.loop_config(block_dim = self.grid_size)
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
            cell_id = self.get_cell_id(pfield[i].p)
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
        if delta > 0: # in contact
            normal = rel_pos / dist
            f1 = normal * delta * stifness
            #Damping force
            M = (pfield[i].m * pfield[j].m) / (pfield[i].m + pfield[j].m)
            K = stifness
            C =  (1. / ti.sqrt(1. + (math.pi / ti.log(restitution_coef)) ** 2)) * ti.sqrt(K * M)
            V = (pfield[j].v - pfield[i].v) * normal
            f2 = C * V * normal
            pfield[i].f += f2 - f1
            pfield[j].f -= f2 - f1



    @ti.kernel
    def collision_detection(self, pfield: ti.template()):

        for i in range(NUM_PARTICLES):
            pfield[i].knn = 0
            pfield[i] = apply_gravity(pfield[i])
            #p = pfield[i].p
            #cell_size = self.cell_size
            cell_id = self.get_cell_id(pfield[i].p)
            #min & max assure boundary conditions
            x_begin = max(cell_id[0]-1, 0)
            x_end = min(cell_id[0]+2, self.grid_size)

            y_begin = max(cell_id[1]-1, 0)
            y_end = min(cell_id[1]+2, self.grid_size)

            z_begin = max(cell_id[2]-1, 0)
            z_end = min(cell_id[2]+2, self.grid_size)

            for cell_i in range(x_begin, x_end):
                for cell_j in range(y_begin, y_end):
                    for cell_k in range(z_begin, z_end):
                        linear_idx = cell_i * self.grid_size * self.grid_size + cell_j * self.grid_size + cell_k
                        for p_id in range(self.head_pointer[linear_idx], self.tail_pointer[linear_idx]):
                            j = self.par_id[p_id]
                            if i < j:   #no overlapping iterations
                                self.resolve_collision(pfield, i, j)
                                #pass
                        pfield[i].knn += self.par_count[cell_i, cell_j, cell_k]

    @ti.func
    def smoothing_kernel(self, radius: ti.f32, dist: ti.f32) -> ti.f32:
        scale = 15/ (2 * math.pi * radius**(5))
        value = ti.math.max(0.0, radius - dist)
        value = (value ** 3) * scale
        #print(vol)
        return value

    #search for particles based on radius
    @ti.kernel
    def calculate_dens(self, pfield: ti.template(), radius: ti.f32):
        #print("HELLO")

        for i in range(NUM_PARTICLES):
            cell_id = self.get_cell_id(pfield[i].p)
            pfield[i].knn = 0
            pfield[i].dens = 0.0
            pfield[i] = apply_gravity(pfield[i])
            cell_radius = ti.math.ceil(radius / self.cell_size[0], dtype = ti.i32)

            x_begin = max(cell_id[0] - cell_radius, 0)
            x_end = min(cell_id[0] + cell_radius + 1, self.grid_size)

            y_begin = max(cell_id[1] - cell_radius, 0)
            y_end = min(cell_id[1] + cell_radius + 1, self.grid_size)

            z_begin = max(cell_id[2] - cell_radius, 0)
            z_end = min(cell_id[2] + cell_radius + 1, self.grid_size)

            for cell_i in range(x_begin, x_end):
                for cell_j in range(y_begin, y_end):
                    for cell_k in range(z_begin, z_end):
                        linear_idx = cell_i * self.grid_size * self.grid_size + cell_j * self.grid_size + cell_k
                        for p_id in range(self.head_pointer[linear_idx], self.tail_pointer[linear_idx]):
                            j = self.par_id[p_id]
                            rel_pos = pfield[j].p - pfield[i].p
                            dist = ti.sqrt(rel_pos[0]**2 + rel_pos[1]**2 + rel_pos[2]**2)
                            #print(radius)
                            if i < j:
                                self.resolve_collision(pfield, i, j)

                            if dist < radius:
                                pfield[i].dens += self.smoothing_kernel(radius, dist) * pfield[j].m
                                #pfield[i].dens_near += self.smoothing_kernel(radius/2, dist) * pfield[j].m






    @ti.kernel
    def update_grid(self, pfield: ti.template()):

        #print(self.par_count[self.grid_size, 0, self.grid_size])
        self.count_particles(pfield)
        self.row_layer_count()
        self.calculate_prefix_sum()
        self.populate_par_id(pfield)
        #self.collision_detection(pfield)
    @ti.kernel
    def average_n(self, pfield: ti.template()) -> ti.i32:
        sum = 0
        for i in range(NUM_PARTICLES):
            sum += pfield[i].knn

        return sum / NUM_PARTICLES

    @ti.kernel
    def average_par_count(self) -> ti.f32:
        sum = 0
        active_cells = 0
        for I in ti.grouped(self.par_count):
            if self.par_count[I] > 0:
                active_cells += 1
                sum += self.par_count[I]

        return sum / active_cells

    @ti.kernel
    def average_density(self, pfield: ti.template()) -> ti.f32:
        sum = 0.0
        for i in range(NUM_PARTICLES):
            sum += pfield[i].dens

        return sum / float(NUM_PARTICLES)
