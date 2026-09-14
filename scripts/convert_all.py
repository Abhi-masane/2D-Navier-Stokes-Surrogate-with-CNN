import json
from pathlib import Path

import numpy as np
import pyvista as pv


X_MIN = -5.0
X_MAX = 15.0
Y_MIN = -5.0
Y_MAX = 5.0
Z_MID = 0.05

NX = 128
NY = 64
CYLINDER_RADIUS = 0.5

RUNS_DIR = Path("runs")
DATASET_DIR = Path("dataset")


def field_names(mesh):
    names = set(mesh.point_data.keys())
    names.update(mesh.cell_data.keys())
    return names


def has_flow_fields(mesh):
    names = field_names(mesh)
    return "U" in names and "p" in names


def mesh_blocks(data):
    blocks = []

    if isinstance(data, pv.MultiBlock):
        for block in data:
            if block is not None:
                blocks.extend(mesh_blocks(block))
        return blocks

    try:
        if data.n_cells > 0 or data.n_points > 0:
            blocks.append(data)
    except Exception:
        pass

    return blocks


def choose_mesh(data):
    blocks = mesh_blocks(data)
    if not blocks:
        return None

    flow_blocks = [block for block in blocks if has_flow_fields(block)]
    if flow_blocks:
        return max(flow_blocks, key=lambda block: block.n_cells)

    return max(blocks, key=lambda block: block.n_cells)


def find_vtk_file(vtk_dir):
    valid_suffixes = {".vtk", ".vtu", ".vtm", ".vtp"}
    files = [path for path in vtk_dir.iterdir() if path.suffix.lower() in valid_suffixes]

    if not files:
        return None

    for path in files:
        if "internal" in path.name.lower():
            return path

    return max(files, key=lambda path: path.stat().st_size)


def sampled_array(sampled, name):
    if name in sampled.point_data:
        return np.asarray(sampled.point_data[name])
    if name in sampled.cell_data:
        return np.asarray(sampled.cell_data[name])
    raise KeyError(name)


def make_sampling_grid():
    return pv.ImageData(
        dimensions=(NX + 1, NY + 1, 1),
        spacing=((X_MAX - X_MIN) / NX, (Y_MAX - Y_MIN) / NY, 1.0),
        origin=(X_MIN, Y_MIN, Z_MID),
    )


def make_mask():
    x = np.linspace(X_MIN, X_MAX, NX + 1)
    y = np.linspace(Y_MIN, Y_MAX, NY + 1)
    xx, yy = np.meshgrid(x, y)
    return (xx**2 + yy**2 > CYLINDER_RADIUS**2).astype(np.float32)


def convert_case(case_dir, grid, mask):
    vtk_dir = case_dir / "VTK"
    if not vtk_dir.is_dir():
        print("skip, no VTK directory:", case_dir)
        return None

    mesh_file = find_vtk_file(vtk_dir)
    if mesh_file is None:
        print("skip, no VTK file:", case_dir)
        return None

    try:
        raw = pv.read(mesh_file)
    except Exception as exc:
        print("failed to read", mesh_file, "-", exc)
        return None

    mesh = choose_mesh(raw)
    if mesh is None:
        print("skip, no valid mesh:", case_dir)
        return None

    try:
        sampled = grid.sample(mesh)
        velocity = sampled_array(sampled, "U")
        pressure = sampled_array(sampled, "p")
    except Exception as exc:
        print("failed to sample", case_dir, "-", exc)
        return None

    velocity = np.nan_to_num(velocity, nan=0.0, posinf=0.0, neginf=0.0)
    pressure = np.nan_to_num(pressure, nan=0.0, posinf=0.0, neginf=0.0)

    expected_points = (NX + 1) * (NY + 1)
    if velocity.shape[0] != expected_points or pressure.shape[0] != expected_points:
        print("skip, unexpected sampled shape:", case_dir)
        return None

    ux = velocity[:, 0].reshape(NY + 1, NX + 1)
    uy = velocity[:, 1].reshape(NY + 1, NX + 1)
    pressure = pressure.reshape(NY + 1, NX + 1)

    reynolds = float((case_dir / "Re.txt").read_text().strip())
    inlet_velocity = float((case_dir / "U.txt").read_text().strip())

    output_name = f"{case_dir.name}.npz"
    output_path = DATASET_DIR / output_name

    np.savez_compressed(
        output_path,
        ux=(ux / inlet_velocity).astype(np.float32),
        uy=(uy / inlet_velocity).astype(np.float32),
        p=(pressure / inlet_velocity**2).astype(np.float32),
        mask=mask,
        Re=np.float32(reynolds),
        U_in=np.float32(inlet_velocity),
    )

    print("converted", case_dir.name, "Re =", reynolds)
    return {
        "file": output_name,
        "Re": reynolds,
        "U_in": inlet_velocity,
    }


def assign_splits(cases):
    for index, case in enumerate(cases):
        if index % 10 == 5:
            case["split"] = "test"
        elif index % 10 == 8:
            case["split"] = "val"
        else:
            case["split"] = "train"


def main():
    DATASET_DIR.mkdir(exist_ok=True)
    grid = make_sampling_grid()
    mask = make_mask()

    cases = []
    for case_dir in sorted(RUNS_DIR.glob("case_Re_*")):
        converted = convert_case(case_dir, grid, mask)
        if converted is not None:
            cases.append(converted)

    cases.sort(key=lambda case: case["Re"])
    assign_splits(cases)

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

    with open(DATASET_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print("finished:", len(cases), "cases written to", DATASET_DIR)


if __name__ == "__main__":
    main()
