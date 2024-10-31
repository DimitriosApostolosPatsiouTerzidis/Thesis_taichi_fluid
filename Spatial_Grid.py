from math_base import *
import numpy as np
#from marchingcubes import *


@ti.data_oriented
class SpatialGrid:
    #TODO: make gridsize self assigned based on computational domain size
    def __init__(self, grid_size: ti.i32):

        #initialize grid buffers
        self.grid_size = grid_size
        #dimensions of each cell
        self.cell_size = vec3(x_max / grid_size, y_max / grid_size, z_max / grid_size)
        #particle id buffer
        self.par_id = ti.field(dtype=ti.i32, shape=(NUM_PARTICLES,), name="par_id")
        # cells equal to number of grid cells
        self.head_pointer = ti.field(dtype =ti.i32, shape=(grid_size*grid_size*grid_size,), name="head_pointer")
        #tail pointer to keep track of end of cell
        self.tail_pointer = ti.field(dtype=ti.i32, shape=(grid_size * grid_size * grid_size,), name="tail_pointer")
        #current pointer to keep track of current particle in cell
        self.cur_pointer = ti.field(dtype =ti.i32, shape=(grid_size*grid_size*grid_size,), name="cur_pointer")
        #particle count in each cell
        self.par_count = ti.field(dtype =ti.i32, shape=(grid_size, grid_size, grid_size), name="par_count")
        #prefix sum for parallelization
        self.prefix_sum = ti.field(dtype =ti.i32, shape=(grid_size, grid_size, grid_size), name="prefix_sum")
        #row sum for parallelization
        self.row_sum = ti.field(dtype =ti.i32, shape=(grid_size, grid_size), name="row_sum")
        #layer sum for parallelization
        self.layer_sum = ti.field(dtype =ti.i32, shape=(grid_size,), name="layer_sum")

        #print particle and grid information
        print(f"\tParticle Diameter: {PARTICLE_RADIUS * 2}")
        #print(f"Particle Mass: {pf[0].m}")
        print(f"\tGrid Size: {grid_size} x {grid_size} x {grid_size} ")
        print(f"\tNumber of Grid Cells: {grid_size**3}")
        print(f"\tCell size: {self.cell_size}")
        assert PARTICLE_RADIUS * 2 < self.cell_size[0]  # cell must be able to contain at least one particle
        assert PARTICLE_RADIUS * 2 < self.cell_size[1]  # cell must be able to contain at least one particle
        assert PARTICLE_RADIUS * 2 < self.cell_size[2]  # cell must be able to contain at least one particle


    #dreturns the cell id of a particle
    @ti.func
    def get_cell_id(self, pos: ti.template()):
        cell_id = ivec3(0.0, 0.0, 0.0)
        cell_id[0] = int(pos[0] / self.cell_size[0])
        cell_id[1] = int(pos[1] / self.cell_size[1])
        cell_id[2] = int(pos[2] / self.cell_size[2])
        return cell_id

    #count particles in each cell
    @ti.func
    def count_particles(self, pfield):
        self.par_count.fill(0)
        for i in range(NUM_PARTICLES):
            cell_id = self.get_cell_id(pfield[i].p)
            #print(cell_id)
            self.par_count[cell_id] += 1

    #count particles in each row and layer
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


    @ti.func
    def calculate_prefix_sum(self): #with this sequence of iterations independancy is achieved along with the ability of parallelization
        self.prefix_sum[0, 0, 0] = 0

        #needs to be serialized
        ti.loop_config(serialize=True)
        for i in range(1, self.grid_size): #cells[i,0,0] init porefix sum sequentially
            self.prefix_sum[i,0,0] = self.prefix_sum[i-1, 0, 0] + self.layer_sum[i-1]

        for i in range(self.grid_size):
            for j in range(1, self.grid_size): #cells[i,j,0] sequentially accumulate row_count to prefix
                self.prefix_sum[i,j,0] = self.prefix_sum[i, j-1, 0] + self.row_sum[i,j-1]

        # Flatten the i and j loops for parallel execution
        total_cells = self.grid_size * self.grid_size
        for linear_idx in range(total_cells):
            i = linear_idx // self.grid_size
            j = linear_idx % self.grid_size

            # Sequential loop f or k
            for k in range(self.grid_size):
                if k == 0:
                    self.prefix_sum[i, j, k] += self.par_count[i, j, k]
                else:
                    self.prefix_sum[i, j, k] = self.prefix_sum[i, j, k - 1] + self.par_count[i, j, k]

                linear_idx_flat = flatten_index(i, j, k)

                self.head_pointer[linear_idx_flat] = self.prefix_sum[i, j, k] - self.par_count[
                    i, j, k]  # start of cell pointer
                self.tail_pointer[linear_idx_flat] = self.prefix_sum[i, j, k]  # end of cell pointer
                self.cur_pointer[linear_idx_flat] = self.head_pointer[linear_idx_flat]  # current in-cell pointer
        '''
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                for k in range(self.grid_size):
                    if k == 0:
                        self.prefix_sum[i,j,k] += self.par_count[i,j,k]
                    else:
                        self.prefix_sum[i,j,k] = self.prefix_sum[i,j,k-1] + self.par_count[i,j,k]

                    linear_idx = flatten_index(i, j, k)

                    self.head_pointer[linear_idx] = self.prefix_sum[i,j,k] - self.par_count[i,j,k] #start of cell pointer
                    self.tail_pointer[linear_idx] = self.prefix_sum[i,j,k] #end of cell pointer
                    self.cur_pointer[linear_idx] = self.head_pointer[linear_idx] #current in-cell pointer'''




    @ti.func
    def populate_par_id(self, pfield: ti.template()):
        #ti.loop_config(serialize=True)
        for i in range(NUM_PARTICLES):
            cell_id = self.get_cell_id(pfield[i].p)
            linear_idx = flatten_index(cell_id[0], cell_id[1], cell_id[2])
            par_location = ti.atomic_add(self.cur_pointer[linear_idx], 1)
            self.par_id[par_location] = i





    @ti.func
    def density_correction(self, pfield: ti.template(), near_boundary: ti.i32, i: ti.i32):
        if near_boundary > 0:
            pfield[i].dens *= ti.pow(dens_cor_coef, near_boundary)
            #pfield[i].near_dens *= ti.pow(dens_cor_coef, near_boundary)


    @ti.func
    def boundary_search(self, cell_id):
        near_boundary = 0
        for i in range(3):
            if cell_id[i] == 0 or cell_id[i]  == self.grid_size - 1:
                near_boundary =+ 1
                assert near_boundary <= 3
        return near_boundary

    #search for particles based on radius
    @ti.func
    def calculate_dens(self, pfield: ti.template(), radius: ti.f32):
        cell_radius = ti.math.ceil(radius / self.cell_size[0], dtype=ti.i32)

        for i in range(NUM_PARTICLES):
            #near_boundary = 0
            cell_id = self.get_cell_id(pfield[i].p)
            pfield[i].knn = 0
            pfield[i].dens = 0.0
            pfield[i].near_dens = 0.0
            #pfield[i] = apply_gravity(pfield[i])

            near_boundary = self.boundary_search(cell_id)


            x_begin = max(cell_id[0] - cell_radius, 0)
            x_end = min(cell_id[0] + cell_radius + 1, self.grid_size)

            y_begin = max(cell_id[1] - cell_radius, 0)
            y_end = min(cell_id[1] + cell_radius + 1, self.grid_size)

            z_begin = max(cell_id[2] - cell_radius, 0)
            z_end = min(cell_id[2] + cell_radius + 1, self.grid_size)

            for cell_i in range(x_begin, x_end):
                for cell_j in range(y_begin, y_end):
                    for cell_k in range(z_begin, z_end):
                        #if cell is empty skip
                        if self.par_count[cell_i, cell_j, cell_k] == 0:
                            continue
                        linear_idx = flatten_index(cell_i, cell_j, cell_k)
                        h_pointer = self.head_pointer[linear_idx]
                        t_pointer = self.tail_pointer[linear_idx]
                        for p_id in range(h_pointer, t_pointer):
                            j = self.par_id[p_id]
                            rel_pos = pfield[j].p - pfield[i].p
                            dist = rel_pos.norm()

                            if dist < radius:
                                '''if i < j:  # no overlapping iterations
                                    resolve_collision(pfield, i, j)'''
                                pfield[i].dens += p_V * cubic_kernel(dist, radius)
                                pfield[i].near_dens +=  p_V * near_cubic_kernel(dist, radius)
                                #pfield[i].dens_near += self.smoothing_kernel(radius/2, dist) * pfield[j].m
                                pfield[i].knn += 1
            pfield[i].dens *= density
            pfield[i].near_dens *= density
            self.density_correction(pfield, near_boundary, i) #correction for near boundary particles

    @ti.func
    def calculate_pressure(self, pfield: ti.template()):
        for i in range(NUM_PARTICLES):
            #Tait equation of state
            pfield[i].pressure = p_stifness * (ti.pow(pfield[i].dens / density, exponent) - 1)
            #Modified Tait equation of state
            temp_dens = min(pfield[i].near_dens, density)
            #pfield[i].near_pressure = p_stifness_near * (ti.pow(pfield[i].near_dens / density ,exponent))  #Clavet et al. 2005
            pfield[i].near_pressure = p_stifness_near * (ti.pow(temp_dens / density ,exponent))  #Clavet et al. 2005

    @ti.func
    def calculate_pressure_force(self, pfield: ti.template(), i: ti.i32, j: ti.i32, rel_pos: ti.f32, radius: ti.f32) -> ti.f32:
        dpi = pfield[i].pressure / (pfield[i].dens ** 2)
        dpj = pfield[j].pressure / (pfield[j].dens ** 2)
        a = -density * p_V * (dpi + dpj) * cubic_kernel_derivative(rel_pos, radius)

        dpi = pfield[i].near_pressure / (pfield[i].near_dens ** 2)
        dpj = pfield[j].near_pressure / (pfield[j].near_dens ** 2)
        #a += -density * p_V * (dpi + dpj) * near_cubic_kernel_derivative(rel_pos, radius)
        a += -density * p_V * (dpi + dpj) * cubic_kernel_derivative(rel_pos, radius)
        return a



    @ti.func
    def calculate_viscosity(self, pfield: ti.template(), i: ti.i32, j: ti.i32, rel_pos: ti.f32, radius: ti.f32) -> ti.f32:
        d = 10 #scalling factored based on dimensionality for 3D
        rel_vel = (pfield[i].v - pfield[j].v).dot(rel_pos) #relative velocity based on the normal
        a = d * viscosity * (pfield[j].m / pfield[j].dens) * rel_vel/(rel_pos.norm() ** 2 + 0.01 * radius ** 2) * cubic_kernel_derivative(rel_pos, radius)
        return a

    @ti.func
    def calculate_surface_tension(self, pfield: ti.template(), i: ti.i32, j: ti.i32, rel_pos: ti.f32, radius: ti.f32) -> ti.f32:
        d2 = (2 * PARTICLE_RADIUS) ** 2
        r2 = rel_pos.dot(rel_pos)
        a = ti.Vector([0.0, 0.0, 0.0])
        if r2 > d2:
            a = surface_tension / pfield[i].m * pfield[j].m * rel_pos * cubic_kernel(rel_pos.norm(), radius)
        else:
            a = surface_tension / pfield[i].m * pfield[j].m * rel_pos * cubic_kernel(ti.Vector([2* PARTICLE_RADIUS, 0.0, 0.0]).norm(), radius)
        return a



    @ti.kernel
    def calculate_forces(self, pfield: ti.template(), radius: ti.f32):
        for i in range(NUM_PARTICLES):
            pfield[i] = apply_gravity(pfield[i])  # gravity force
            pfield[i] = apply_bc(pfield[i])  # boundary conditions

        self.calculate_dens(pfield, radius)
        self.calculate_pressure(pfield)

        cell_radius = ti.math.ceil(radius / self.cell_size[0], dtype=ti.i32)

        for i in range(NUM_PARTICLES):

            cell_id = self.get_cell_id(pfield[i].p)

            # min & max assure boundary conditions
            x_begin = max(cell_id[0] - cell_radius, 0)
            x_end = min(cell_id[0] + cell_radius + 1, self.grid_size)

            y_begin = max(cell_id[1] - cell_radius, 0)
            y_end = min(cell_id[1] + cell_radius + 1, self.grid_size)

            z_begin = max(cell_id[2] - cell_radius, 0)
            z_end = min(cell_id[2] + cell_radius + 1, self.grid_size)

            for cell_i in range(x_begin, x_end):
                for cell_j in range(y_begin, y_end):
                    for cell_k in range(z_begin, z_end):
                        #if cell is empty skip
                        if self.par_count[cell_i, cell_j, cell_k] == 0:
                            continue
                        linear_idx = flatten_index(cell_i, cell_j, cell_k)
                        h_pointer = self.head_pointer[linear_idx]
                        t_pointer = self.tail_pointer[linear_idx]
                        for p_id in range(h_pointer, t_pointer):
                            j = self.par_id[p_id]
                            rel_pos = pfield[i].p - pfield[j].p
                            dist = rel_pos.norm()
                            scale = (dist - 0.3 * PARTICLE_RADIUS) / dist #??????????????????other scale?
                            if dist < radius:
                                #pfield[i].a -= self.calculate_surface_tension(pfield, i, j,  rel_pos, radius)
                                pfield[i].a += self.calculate_viscosity(pfield, i, j,  rel_pos, radius)
                                pfield[i].a += self.calculate_pressure_force(pfield, i, j, rel_pos, radius)
                                #pfield[i].a += self.calculate_pressure_force(pfield, i, j, scale * rel_pos, radius/2)






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

    @ti.kernel
    def average_frnn(self, pfield: ti.template()) -> ti.f32:
        sum = 0
        for i in range(NUM_PARTICLES):
            sum += pfield[i].knn

        return sum / NUM_PARTICLES

