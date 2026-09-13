import math
import os

R = 0.5
z0 = -0.1
z1 = 0.2
N = 96

os.makedirs("constant/triSurface", exist_ok=True)

def sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])

def cross(a, b, c):
    # (b-a) x (c-a)
    u = sub(b, a)
    v = sub(c, a)
    return (
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0]
    )

def dot(u, v):
    return u[0] * v[0] + u[1] * v[1] + u[2] * v[2]

def normalize(v):
    mag = math.sqrt(dot(v, v))
    if mag == 0:
        return (0.0, 0.0, 0.0)
    return (v[0] / mag, v[1] / mag, v[2] / mag)

def write_triangle(f, desired_normal, a, b, c):
    n = cross(a, b, c)

    if dot(n, desired_normal) < 0:
        a, b, c = c, b, a
        n = (-n[0], -n[1], -n[2])

    n = normalize(n)

    f.write("facet normal {:.6f} {:.6f} {:.6f}\n".format(n[0], n[1], n[2]))
    f.write(" outer loop\n")
    f.write("  vertex {:.6f} {:.6f} {:.6f}\n".format(a[0], a[1], a[2]))
    f.write("  vertex {:.6f} {:.6f} {:.6f}\n".format(b[0], b[1], b[2]))
    f.write("  vertex {:.6f} {:.6f} {:.6f}\n".format(c[0], c[1], c[2]))
    f.write(" endloop\n")
    f.write("endfacet\n")

with open("constant/triSurface/cylinder.stl", "w") as f:
    f.write("solid cylinder\n")

    # Side surface
    for i in range(N):
        a0 = 2.0 * math.pi * i / N
        a1 = 2.0 * math.pi * (i + 1) / N

        b0 = (R * math.cos(a0), R * math.sin(a0), z0)
        b1 = (R * math.cos(a1), R * math.sin(a1), z0)
        t0 = (R * math.cos(a0), R * math.sin(a0), z1)
        t1 = (R * math.cos(a1), R * math.sin(a1), z1)

        amid = 0.5 * (a0 + a1)
        outward = (math.cos(amid), math.sin(amid), 0.0)

        write_triangle(f, outward, b0, b1, t1)
        write_triangle(f, outward, b0, t1, t0)

    # Top cap
    center_top = (0.0, 0.0, z1)
    for i in range(N):
        a0 = 2.0 * math.pi * i / N
        a1 = 2.0 * math.pi * (i + 1) / N

        p0 = (R * math.cos(a0), R * math.sin(a0), z1)
        p1 = (R * math.cos(a1), R * math.sin(a1), z1)

        write_triangle(f, (0.0, 0.0, 1.0), center_top, p0, p1)

    # Bottom cap
    center_bottom = (0.0, 0.0, z0)
    for i in range(N):
        a0 = 2.0 * math.pi * i / N
        a1 = 2.0 * math.pi * (i + 1) / N

        p0 = (R * math.cos(a0), R * math.sin(a0), z0)
        p1 = (R * math.cos(a1), R * math.sin(a1), z0)

        write_triangle(f, (0.0, 0.0, -1.0), center_bottom, p1, p0)

    f.write("endsolid cylinder\n")

print("Created constant/triSurface/cylinder.stl")
