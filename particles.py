import taichi as ti
from Constants import *
from math_base import *
import math

#ti.init(arch=ti.gpu)

@ti.dataclass
class fluidPar:     #particle struct
    id:  ti.i32     #particle id
    f: vec3         #force
    p: vec3         #position
    v: vec3         #velocity
    a: vec3         #acceleration
    color: vec3     #color value
    m: ti.f32       #mass calculated based on given density
    dens: ti.f32    #density
    pressure: ti.f32      #pressure
    knn: ti.i32     #number of particles in neighborhood
    #rig: ti.i32    #if 1 represents rigid body voxel


@ti.kernel
def init_particles_pos(pfield: ti.template(), start: ti.template(), reset: ti.i32):
    for index in range(NUM_PARTICLES):
        i = index // (y_dim * z_dim)
        j = (index // z_dim) % y_dim
        k = index % z_dim
        pfield[index].p.x = start[0].x + i * ((PARTICLE_RADIUS * 2) + PADDING) + (ti.random() * (PARTICLE_RADIUS / 4))
        pfield[index].p.y = start[0].y + j * ((PARTICLE_RADIUS * 2) + PADDING) + (ti.random() * (PARTICLE_RADIUS / 4))
        pfield[index].p.z = start[0].z + k * ((PARTICLE_RADIUS * 2) + PADDING) + (ti.random() * (PARTICLE_RADIUS / 4))
        #pfield[index].m = (4 / 3) * density * math.pi * PARTICLE_RADIUS ** 3
        pfield[index].m = p_V * density
        pfield[index].dens = density

        pfield[index].id = index
        if reset == 1:
            pfield[index].v = vec3(0.0, 0.0, 0.0)
            pfield[index].a = vec3(0.0, 0.0, 0.0)
            pfield[index].f = vec3(0.0, 0.0, 0.0)





@ti.func
def update(fp):
    a = fp.f / fp.m # a = F/m
    fp.v += ((fp.a + a) * dt / 2.0) #verlet integration
    fp.p += fp.v * dt + 0.5 * a * dt ** 2
    fp.a = a

    return fp

@ti.func
def update_particle(fp):
    #tempfp = apply_gravity(fp)
    #tempfp = grid.collision_detection(pfield)
    #tempfp = apply_bc(fp)
    tempfp = update(fp)

    return tempfp

@ti.kernel
def step(pfield: ti.template()):
    #grid.collision_detection(pfield)
    for i in pfield:
        pfield[i] = update_particle(pfield[i])
        #if ti.math.length(pfield[i].v) > max_vel:
        #    max_vel = pfield[i].v