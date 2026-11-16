"""Arrange the exported STLs on one build plate, in the same places and orientations
as plate_v05_ABS_all_parts.3mf. Run after chestpod.py."""
import os

import matplotlib
import numpy as np
import trimesh

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.patches import Rectangle

BED = 256.0   # Bambu X1 Carbon / P1 / A1
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
PLATE = "plate_v06_all_parts"
PARTS = [("tray_print", 8, 8), ("cover_print", 112, 8), ("cam_head_print", 8, 130)]  # bottom-left corners


def place(name, x, y):
    m = trimesh.load(os.path.join(OUT, name + ".stl"))
    lo = m.bounds[0]
    m.apply_translation([x - lo[0], y - lo[1], -lo[2]])
    return m


meshes = [(n, place(n, x, y)) for n, x, y in PARTS]
for n, m in meshes:
    lo, hi = m.bounds
    assert lo.min() >= 0 and hi[:2].max() <= BED, (n, lo, hi)
    assert m.is_watertight, n
for i, (na, a) in enumerate(meshes):
    for nb, b in meshes[i + 1:]:
        apart = (a.bounds[1][:2] + 4 <= b.bounds[0][:2]) | (b.bounds[1][:2] + 4 <= a.bounds[0][:2])
        assert apart.any(), (na, nb)

scene = trimesh.Scene()
for n, m in meshes:
    scene.add_geometry(m, node_name=n, geom_name=n)
scene.export(os.path.join(OUT, PLATE + ".3mf"))

fig, ax = plt.subplots(figsize=(6, 6))
ax.add_patch(Rectangle((0, 0), BED, BED, fill=False, lw=2))
for n, m in meshes:
    ax.add_collection(PolyCollection(m.vertices[m.faces][:, :, :2], facecolor="#e8862a", alpha=0.25))
    lo, hi = m.bounds
    ax.text((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, f"{n}\n{hi[2]:.0f} mm tall",
            ha="center", va="center", fontsize=7)
    print(f"{n:16s} {np.round(hi - lo, 1)}  {m.volume / 1000:.1f} cm3")
ax.set_xlim(-5, BED + 5); ax.set_ylim(-5, BED + 5); ax.set_aspect("equal")
ax.set_title(f"{PLATE}  ({BED:.0f} x {BED:.0f} mm bed)")
plt.tight_layout()
plt.savefig(os.path.join(OUT, PLATE + ".png"), dpi=110)
print(PLATE, "OK")
