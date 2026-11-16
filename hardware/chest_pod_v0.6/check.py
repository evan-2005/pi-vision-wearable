"""Interference checks against a stand-in Pi + case + plugs + ribbon, then renders.
Run after chestpod.py. Exits non-zero if anything collides."""
import os
import sys
import tempfile

import cadquery as cq
import matplotlib
import numpy as np
import trimesh

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import chestpod as P
from chestpod import box

tray, cover, cover_bare = P.make_tray(), P.make_cover(), P.make_cover(springs=False)
head, info = P.make_cam_head()
Y0 = P.BACK
XG, XH = P.X_GPIO_EDGE, P.X_HDMI_EDGE
PCB_Y = Y0 + P.PCB_UP

# ---- stand-ins, ports DOWN ----
shell = box(XG, XH, Y0, Y0 + P.CASE_T, 0, P.CASE_L)
sockets = box(XH, XH + P.SOCKET_OUT, PCB_Y, PCB_Y + 3.5, P.from_sd(45), P.from_sd(5))
ports = box(XG + 1.5, XH - 2, PCB_Y, PCB_Y + 16.2, P.PORT_TIP_Z, 20)
case = shell.union(sockets).union(ports)
thick_case = box(XG, XH, Y0, Y0 + P.CASE_T + 1.0, 0, P.CASE_L)        # a case 1 mm thicker than spec
sd = box(P.SD_X - 6, P.SD_X + 6, PCB_Y - 3.2, PCB_Y - 1.7, P.CASE_L - 1, P.from_sd(-P.SD_OUT) + 0.5)
usb_plugs = [box(x - 8, x + 8, PCB_Y + 0.5, PCB_Y + 16.5, P.PORT_TIP_Z - 40, P.PORT_TIP_Z) for x in P.USB_X]
usbc_plug = box(P.PW / 2, P.PW / 2 + 35, PCB_Y + 1.6 - 3.75, PCB_Y + 1.6 + 3.75, P.USBC_Z - 6.5, P.USBC_Z + 6.5)
dupont = box(XG + 0.9, XG + 6.1, PCB_Y + 2.5, P.Y_FRONT + 6, P.GPIO_Z[1] - 16, P.GPIO_Z[1])  # pins 1-12
# ribbon: three flat layers in the left column (the S-fold that stores the 200 mm cable), then out the slit
ribbon = box(P.CSI_X - 6, P.CSI_X + 6, Y0 + P.CASE_T + 0.3, P.Y_SPLIT + P.FRONT_GAP - 0.3, 22, P.CASE_L).union(
    box(P.CSI_X - 6, P.CSI_X + 6, P.Y_SPLIT + 0.4, P.Y_SPLIT + 2.4, P.CASE_L - 1, P.Z_TOP + 4))
# camera stand-in in head-local coords
ypf, zpb = info["y_pcb_front"], info["z_pcb_bot"]
cam = box(-P.CAM_W / 2, P.CAM_W / 2, ypf - P.CAM_PCB_T, ypf, zpb, zpb + P.CAM_H).union(
    box(-5.4, 5.4, ypf, ypf + P.CAM_LENS_UP, info["lens_z"] - 5.4, info["lens_z"] + 5.4)).union(
    box(-9.8, 9.8, ypf - P.CAM_PCB_T - 2.75, ypf - P.CAM_PCB_T, zpb, zpb + 5.71))


def vol(a, b):
    try:
        return round(a.intersect(b).val().Volume(), 2)
    except Exception:
        return 0.0


clear = {                       # every one of these must be 0
    "case vs tray": vol(case, tray),
    "case vs cover (springs removed)": vol(case, cover_bare),
    "1 mm thicker case vs cover (springs removed)": vol(thick_case, cover_bare),
    "microSD vs tray": vol(sd, tray),
    "tray vs cover": vol(tray, cover),
    "USB-C plug vs tray": vol(usbc_plug, tray),
    "USB-C plug vs cover": vol(usbc_plug, cover),
    "button jumper wires vs cover": vol(dupont, cover),
    "ribbon path vs cover": vol(ribbon, cover),
    "ribbon path vs tray": vol(ribbon, tray),
    "camera vs head": vol(cam, head),
}
for i, p in enumerate(usb_plugs):
    clear[f"USB-A plug {i + 1} vs tray"] = vol(p, tray)
    clear[f"USB-A plug {i + 1} vs cover"] = vol(p, cover)
for t in (-15, 0, 15, 30, 45):
    hp = P.place_head(head, t)
    clear[f"tilt {t:>3}: head vs cover"] = vol(hp, cover)
    clear[f"tilt {t:>3}: head vs tray"] = vol(hp, tray)
for k, v in clear.items():
    print(f"{k:46s} {v}")

spring = vol(shell, cover)      # must be > 0: the tongue posts are meant to press on the fins
preload = P.CASE_T - P.SPRING_REACH
strain = 1.5 * P.TONGUE_T * (preload + 1.0) / P.TONGUE_L ** 2
print(f"{'spring posts overlap the case (wanted)':46s} {spring}  (preload {preload:.1f} mm)")
print(f"tongue strain with a case 1 mm over spec: {strain:.2%} (keep under 1%)")
for n, w in (("tray", tray), ("cover", cover), ("head", head)):
    bb = w.val().BoundingBox()
    print(f"{n:6s} {bb.xlen:6.1f} x {bb.ylen:5.1f} x {bb.zlen:6.1f} mm  {w.val().Volume() / 1000:5.1f} cm3")

bad = [k for k, v in clear.items() if v > 0]
assert not bad, f"collisions: {bad}"
assert spring > 0 and strain < 0.01

# ---- renders ----
def mesh(w):
    tmp = os.path.join(tempfile.gettempdir(), "_chestpod_mesh.stl")
    cq.exporters.export(w, tmp, tolerance=0.05, angularTolerance=0.3)
    return trimesh.load(tmp)


items = [(mesh(tray), "#d9d9d9", 1.0), (mesh(cover), "#3d7dd8", 0.5),
         (mesh(P.place_head(head, 15)), "#e8862a", 1.0), (mesh(case), "#333333", 1.0),
         (mesh(P.place_head(cam, 15)), "#2e9e4a", 1.0), (mesh(ribbon), "#f0a020", 1.0),
         (mesh(usbc_plug), "#b03030", 1.0)]
views = [("FRONT, as you face the wearer (camera 15 deg down)", 8, 90),
         ("TOP: exhaust slots, microSD window, ribbon slit", 89.9, 90),
         ("BACK (chest side): band slots, lanyard tab", 8, -90),
         ("LEFT side seen from the front: USB-C window", 5, 0)]
fig = plt.figure(figsize=(16, 13))
for i, (title, el, az) in enumerate(views):
    ax = fig.add_subplot(2, 2, i + 1, projection="3d")
    for m, col, a in items:
        sh = 0.55 + 0.45 * np.clip(m.face_normals @ np.array([0.3, 0.6, 0.75]), 0, 1)
        pc = Poly3DCollection(m.vertices[m.faces], edgecolor="none")
        pc.set_facecolor(np.c_[np.outer(sh, matplotlib.colors.to_rgb(col)), np.full(len(sh), a)])
        ax.add_collection3d(pc)
    ax.set_xlim(-55, 55); ax.set_ylim(-35, 75); ax.set_zlim(-10, 135)
    ax.set_box_aspect((110, 110, 145)); ax.view_init(el, az); ax.set_title(title); ax.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(P.OUT, "preview_assembly.png"), dpi=80)
print("all clear; saved preview_assembly.png")
