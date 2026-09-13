import glob
import json
import os

import numpy as np
import pyvista as pv


# -----------------------------
# Fixed sampling grid definition
# -----------------------------
X_MIN, X_MAX = -5.0, 15.0
Y_MIN, Y_MAX = -5.0, 5.0
Z_MID = 0.05

NX, NY = 128, 64
R = 0.5

os.makedirs("dataset", exist_ok=True)

grid = pv.ImageData(
    dimensions=(NX + 1, NY + 1, 1),
    spacing=((X_MAX - X_MIN) / NX, (Y_MAX - Y_MIN) / NY, 1.0),
    origin=(X_MIN, Y_MIN, Z_MID),
)

X, Y = np.meshgrid(
    np.linspace(X_MIN, X_MAX, NX + 1),
    np.linspace(Y_MIN, Y_MAX, NY + 1),
)

# 1 = fluid, 0 = cylinder/solid
mask = (X**2 + Y**2 > R**2).astype(np.float32)


def array_names(mesh):
    """Return available field names on a mesh."""
    names = []
    if hasattr(mesh, "point_data"):
        names += list(mesh.point_data.keys())
    if hasattr(mesh, "cell_data"):
        names += list(mesh.cell_data.keys())
    return names


def has_U_and_p(mesh):
    names = array_names(mesh)
    return ("U" in names) and ("p" in names)


def collect_valid_blocks(obj, blocks=None):
    """
    Recursively collect non-empty PyVista blocks from a mesh or MultiBlock object.
    Skips None blocks safely.
    """
    if blocks is None:
        blocks = []

    if obj is None:
        return blocks

    if isinstance(obj, pv.MultiBlock):
        for block in obj:
            collect_valid_blocks(block, blocks)
    else:
        try:
            if obj.n_cells > 0 or obj.n_points > 0:
                blocks.append(obj)
        except Exception:
            pass

    return blocks


def pick_fluid_mesh(data):
    """
    Pick the best block from a VTK/MultiBlock file.
    Preference:
    1. block containing U and p
    2. largest block by number of cells
    """
    blocks = collect_valid_blocks(data)

    if len(blocks) == 0:
        return None

    blocks_with_fields = [b for b in blocks if has_U_and_p(b)]

    if blocks_with_fields:
        return max(blocks_with_fields, key=lambda b: b.n_cells)

    return max(blocks, key=lambda b: b.n_cells)


def find_mesh_file(vtk_dir):
    """
    Find likely internal mesh VTK file.
    Prefer filenames containing 'internal'.
    Otherwise choose the largest VTK-like file.
    """
    vtk_files = [
        f for f in glob.glob(os.path.join(vtk_dir, "*"))
        if f.endswith((".vtk", ".vtu", ".vtm", ".vtp"))
    ]

    if not vtk_files:
        return None

    for f in vtk_files:
        if "internal" in os.path.basename(f).lower():
            return f

    return max(vtk_files, key=os.path.getsize)


def get_array(sampled, name):
    """
    Get sampled array from point_data or cell_data.
    After grid.sample(), arrays are usually in point_data.
    """
    if name in sampled.point_data:
        return np.asarray(sampled.point_data[name])
    if name in sampled.cell_data:
        return np.asarray(sampled.cell_data[name])
    raise KeyError(f"{name} not found in sampled data")


cases = []

for cd in sorted(glob.glob("runs/case_Re_*")):
    vtk_dir = os.path.join(cd, "VTK")

    if not os.path.isdir(vtk_dir):
        print("SKIP no VTK directory:", cd)
        continue

    mesh_file = find_mesh_file(vtk_dir)

    if mesh_file is None:
        print("SKIP no VTK files:", cd)
        continue

    try:
        raw = pv.read(mesh_file)
    except Exception as e:
        print("FAILED reading:", mesh_file)
        print("Reason:", e)
        continue

    mesh = pick_fluid_mesh(raw)

    if mesh is None:
        print("SKIP no valid mesh block:", cd)
        continue

    if not has_U_and_p(mesh):
        print("WARNING: selected mesh does not directly show U and p:", cd)
        print("Available arrays:", array_names(mesh))

    try:
        sampled = grid.sample(mesh)
    except Exception as e:
        print("FAILED sampling:", cd)
        print("Reason:", e)
        continue

    try:
        U = get_array(sampled, "U")
        p = get_array(sampled, "p")
    except KeyError as e:
        print("SKIP missing sampled arrays:", cd)
        print("Reason:", e)
        print("Sampled point arrays:", list(sampled.point_data.keys()))
        print("Sampled cell arrays:", list(sampled.cell_data.keys()))
        continue

    # Replace NaNs and invalid values with zeros.
    # The mask will later exclude cylinder/solid region from training loss.
    U = np.nan_to_num(U, nan=0.0, posinf=0.0, neginf=0.0)
    p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)

    expected_points = (NY + 1) * (NX + 1)

    if U.shape[0] != expected_points:
        print("SKIP unexpected U shape:", cd, U.shape)
        continue

    if p.shape[0] != expected_points:
        print("SKIP unexpected p shape:", cd, p.shape)
        continue

    ux = U[:, 0].reshape(NY + 1, NX + 1)
    uy = U[:, 1].reshape(NY + 1, NX + 1)
    pp = p.reshape(NY + 1, NX + 1)

    Re_path = os.path.join(cd, "Re.txt")
    U_path = os.path.join(cd, "U.txt")

    Re = float(open(Re_path).read().strip())
    U_in = float(open(U_path).read().strip())

    name = os.path.basename(cd)

    np.savez_compressed(
        f"dataset/{name}.npz",
        ux=(ux / U_in).astype(np.float32),
        uy=(uy / U_in).astype(np.float32),
        p=(pp / (U_in**2)).astype(np.float32),
        mask=mask.astype(np.float32),
        Re=np.float32(Re),
        U_in=np.float32(U_in),
    )

    cases.append(
        {
            "file": f"{name}.npz",
            "Re": Re,
            "U_in": U_in,
        }
    )

    print("converted", name, "Re =", Re)


cases.sort(key=lambda c: c["Re"])

# Split by case/Re, never by grid cells.
for i, c in enumerate(cases):
    if i % 10 == 5:
        c["split"] = "test"
    elif i % 10 == 8:
        c["split"] = "val"
    else:
        c["split"] = "train"


manifest = {
    "source": "OpenFOAM simpleFoam steady laminar cylinder sweep, Re 5-40",
    "grid": {
        "x": [X_MIN, X_MAX],
        "y": [Y_MIN, Y_MAX],
        "z_mid": Z_MID,
        "nx": NX + 1,
        "ny": NY + 1,
    },
    "normalisation": "ux/U_in, uy/U_in, p/U_in^2",
    "re_max": 40.0,
    "cases": cases,
}

with open("dataset/manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)

print("DONE:", len(cases), "cases converted to dataset/")
