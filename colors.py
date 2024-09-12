from Constants import *

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
    alpha = 1.0 #smoothing variable
    max_vel = 6.0
    r = 0.0
    g = 0.0
    b = 0.0
    for i in range(NUM_PARTICLES):
        linear_vel = pfield[i].v.norm()
        smoothed_vel = (alpha * linear_vel) + (1 - alpha) * pfield[i].v.norm()
        rel_vel = min((smoothed_vel / max_vel) , 1.0)
        # if i == 1000:
        #     print(f"rel_vel: {rel_vel} || linear_vel: {linear_vel} || max_vel: {max_vel}")
        if rel_vel <= 0.25:
            r = 0.0
            g = max(rel_vel / 0.25, 0.0)
            b = 1.0
        elif rel_vel <= 0.4:
            r = 0.0
            g = 1.0
            b = 1.0 - (rel_vel - 0.25) / 0.25
        elif rel_vel <= 0.7:
            r = (rel_vel - 0.5) / 0.25
            g = 1.0
            b = 0.0
        elif rel_vel <= 1.0:
            r = 1.0
            g = 1 - (rel_vel - 0.75) / 0.25
            b = 0.0
        prev_r, prev_g, prev_b = cfield[i].xyz
        r = (alpha * r) + (1- alpha) * prev_r
        g = (alpha * g) + (1- alpha) * prev_g
        b = (alpha * b) + (1- alpha) * prev_b
        #g = g + (cfield[i].y * 0.2)
        #b = b + (cfield[i].z * 0.2)
        cfield[i].xyz = r, g, b

@ti.kernel
def density_color(cfield: ti.template(), pfield: ti.template()):
    alpha = 0.9 #smoothing variable
    max_dens = 1100.0 - density
    r = 0.0
    g = 0.0
    b = 0.0
    for i in range(NUM_PARTICLES):
        dens = pfield[i].dens - density
        rel_dens = min((dens / max_dens) , 1.0)
        if rel_dens <= 0.25:
            r = 0.1
            g = min(rel_dens / 0.25, 0.2)
            b = 1.0
        elif rel_dens <= 0.5:
            r = 0.1
            g = 1.0
            b = 1.0 - (rel_dens - 0.25) / 0.25
        elif rel_dens <= 0.75:
            r = (rel_dens - 0.5) / 0.25
            g = 1.0
            b = 0.0
        elif rel_dens <= 1.0:
            r = 1.0
            g = 1 - (rel_dens - 0.75) / 0.25
            b = 0.0
        prev_r, prev_g, prev_b = cfield[i].xyz
        r = (alpha * r) + (1- alpha) * prev_r
        g = (alpha * g) + (1- alpha) * prev_g
        b = (alpha * b) + (1- alpha) * prev_b
        cfield[i].xyz = r, g, b