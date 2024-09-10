import taichi as ti
import math
from Constants import *



@ti.func
def apply_gravity(fp):
    fp.f = vec3(0.0, gravity * fp.m, 0.0)
    return fp

@ti.func
def resolve_collision(pfield: ti.template(), i: ti.i32, j: ti.i32):
    # TODO: implement NN
    # TODO: ensure non compresion
    rel_pos = pfield[j].p - pfield[i].p
    dist = ti.sqrt(rel_pos[0] ** 2 + rel_pos[1] ** 2 + rel_pos[2] ** 2)
    #dist = dist * 1.4
    delta = -dist + (2 * PARTICLE_RADIUS)  # distance with radius accounted
    if delta > 0:  # in contact
        normal = rel_pos / dist
        f1 = normal * delta * stifness
        # Damping force
        M = (pfield[i].m * pfield[j].m) / (pfield[i].m + pfield[j].m)
        K = stifness
        C = (1. / ti.sqrt(1. + (math.pi / ti.log(restitution_coef)) ** 2)) * ti.sqrt(K * M)
        V = (pfield[j].v - pfield[i].v) * normal
        f2 = C * V * normal
        pfield[i].f += f2 - f1
        pfield[j].f -= f2 - f1

@ti.func
def cubic_kernel(r_norm, radius):
    res = ti.cast(0.0, ti.f32)
    h = radius
    # value of cubic spline smoothing kernel
    k = 8 / (math.pi * (h ** 3))
    q = r_norm / h
    #print(f"Q: {q} ")
    if q <= 1.0:
        if q <= 0.5:
            q2 = q * q
            q3 = q2 * q
            res = k * (6.0 * q3 - 6.0 * q2 + 1)
        else:
            res = k * 2 * ti.pow(1 - q, 3.0)
    return res

@ti.func
def cubic_kernel_derivative(r, radius):
    h = radius
    # derivative of cubic spline smoothing kernel
    k = 8 / math.pi
    k = 6. * k / h ** 3
    r_norm = r.norm()
    q = r_norm / h
    res = ti.Vector([0.0 for _ in range(3)])
    if r_norm > 1e-5 and q <= 1.0:
        grad_q = r / (r_norm * h)
        if q <= 0.5:
            res = k * q * (3.0 * q - 2.0) * grad_q
        else:
            factor = 1.0 - q
            res = k * (-factor * factor) * grad_q
    return res
'''

@ti.func
def cubic_kernel(r_norm, radius):
    res = ti.cast(0.0, ti.f32)
    h = radius
    # value of cubic spline smoothing kernel
    k = 8 / (math.pi * (h ** 3))
    q = r_norm / h
    #print(f"Q: {q} ")
    if q <= 1.0:
        q2 = q * q
        q3 = q2 * q
        res = k * (1- 1.5 * q2 - 0.75 * q3)
    return res

@ti.func
def cubic_kernel_derivative(r, radius):
    h = radius
    # derivative of cubic spline smoothing kernel
    k = 8 / math.pi
    k = 6. * k / h ** 3
    r_norm = r.norm()
    q = r_norm / h
    res = ti.Vector([0.0 for _ in range(3)])
    if r_norm > 1e-5 and q <= 1.0:
        grad_q = r / (r_norm * h)
        res = k  * (-3.0 * q - 2.25 * q*q) * grad_q

    return res

@ti.func
def cubic_kernel(r_norm, radius):
    scale = 15/(2 * math.pi * radius ** 5)
    v = radius - r_norm
    return scale * v ** 2

@ti.func
def cubic_kernel_derivative(r, radius):
    scale = 15/ (ti.pow(radius,5) * math.pi)
    v = radius - r.norm()
    return (-v) * scale
'''


@ti.func
def apply_bc(fp):
    velocity_damping = 0.7
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

"""
@ti.func
def compute_pressure(pfield: ti.template(), i: ti.i32, j: ti.i32):
"""


