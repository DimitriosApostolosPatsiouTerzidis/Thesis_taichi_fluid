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
    alpha = 0.3 #smoothing variable
    max_vel = 4.0
    r = 0.0
    g = 0.0
    b = 0.0
    for i in range(NUM_PARTICLES):
        linear_vel = pfield[i].v.norm()
        rel_vel = min((linear_vel / max_vel) , 1.0)

        if rel_vel <= 0.25:
            r = 0.0
            g = rel_vel / 0.25
            b = 1.0
        elif rel_vel <= 0.5:
            r = 0.0
            g = 1.0
            b = 1.0 - (rel_vel - 0.25) / 0.25
        elif rel_vel <= 0.75:
            r = (rel_vel - 0.5) / 0.25
            g = 1.0
            b = 0.0
        else:
            r = 1.0
            g = 1.0 - (rel_vel - 0.75) / 0.25
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
    alpha = 0.2  # Smoothing factor for color transitions
    max_dens = 1050.0 - density  # Adjust this based on your simulation's density range

    for i in range(NUM_PARTICLES):
        dens = pfield[i].dens - density  # Calculate the density difference from the reference
        rel_dens = min((dens / max_dens), 1.0)  # Normalize density to [0, 1] range

        # Initialize color values
        r, g, b = 0.0, 0.0, 0.0

        # Map relative density to a heatmap color
        if rel_dens <= 0.25:
            r = 0.1
            g = min(rel_dens / 0.25, 1.0)
            b = 1.0
        elif rel_dens <= 0.5:
            r = 0.1
            g = 1.0
            b = 1.0 - (rel_dens - 0.25) / 0.25
        elif rel_dens <= 0.75:
            r = (rel_dens - 0.5) / 0.25
            g = 1.0
            b = 0.0
        else:
            r = 1.0
            g = 1.0 - (rel_dens - 0.75) / 0.25
            b = 0.0

        # Retrieve the previous color
        prev_r, prev_g, prev_b = cfield[i].xyz

        # Apply smoothing using alpha
        r = alpha * r + (1 - alpha) * prev_r
        g = alpha * g + (1 - alpha) * prev_g
        b = alpha * b + (1 - alpha) * prev_b

        # Assign the smoothed color back to the color field
        cfield[i].xyz = r, g, b


@ti.kernel
def pressure_color(cfield: ti.template(), pfield: ti.template()):
    alpha = 0.2  # Smoothing factor for color transitions
    min_pressure = -400.0
    max_pressure = 6000.0

    for i in range(NUM_PARTICLES):
        pressure = pfield[i].pressure + pfield[i].near_pressure  # Combine pressure and near pressure

        # Normalize pressure to [0, 1] range, with min_pressure being the lower bound and max_pressure the upper bound
        rel_pressure = (pressure - min_pressure) / (max_pressure - min_pressure)
        rel_pressure = min(max(rel_pressure, 0.0), 1.0)  # Clamp to [0, 1]

        # Initialize color values
        r, g, b = 0.0, 0.0, 0.0

        # Map normalized pressure to heatmap colors
        if rel_pressure <= 0.25:  # Cool colors (blue to cyan)
            r = 0.0
            g = rel_pressure / 0.25
            b = 1.0
        elif rel_pressure <= 0.5:  # Transition from cyan to green
            r = 0.0
            g = 1.0
            b = 1.0 - (rel_pressure - 0.25) / 0.25
        elif rel_pressure <= 0.75:  # Transition from green to yellow
            r = (rel_pressure - 0.5) / 0.25
            g = 1.0
            b = 0.0
        else:  # Warm colors (yellow to red)
            r = 1.0
            g = 1.0 - (rel_pressure - 0.75) / 0.25
            b = 0.0

        # Retrieve the previous color
        prev_r, prev_g, prev_b = cfield[i].xyz

        # Apply smoothing using alpha
        r = alpha * r + (1 - alpha) * prev_r
        g = alpha * g + (1 - alpha) * prev_g
        b = alpha * b + (1 - alpha) * prev_b

        # Assign the smoothed color back to the color field
        cfield[i].xyz = r, g, b
