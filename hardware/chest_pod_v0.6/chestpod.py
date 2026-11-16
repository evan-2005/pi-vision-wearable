"""
Chest-worn housing for a Raspberry Pi 5 in an aluminium armour dual-fan case
+ Raspberry Pi Camera Module 3 Wide, worn on a single chest band.

Parametric CadQuery model. Edit the PARAMETERS block, then:
    python chestpod.py && python check.py && python plates.py
Outputs STL + STEP into ./out

v0.6 (Nov 2026): re-dimensioned from the official drawings instead of photos.
  - pocket depth 31.5 -> 25 (the case is 24 mm thick, not 31.5)
  - side wings and screw pads merged into one slim rail per side (89 -> 78 mm wide)
  - spring tongues in the cover hold the case, so +-1 mm of case thickness still fits
  - rounded corners, chamfered chest-side edge, thinner back
  - camera lens height corrected (8.3 mm, was 11.3), power-button poke hole

Orientation is unchanged from v0.5 (as worn, looking AT the wearer from the front):
  X = across the chest, +X = wearer's RIGHT = viewer's LEFT.
      USB-C/HDMI edge, CSI ribbon, Ethernet and camera on the viewer's LEFT (+X);
      GPIO header on the RIGHT (-X). USB-C top-left, USB-A + Ethernet along the bottom.
  Y = out of the chest (+Y = forward). Y=0 is the back face against the chest/band.
  Z = up. Z=0 is the ledge the aluminium shell stands on. USB/Ethernet point DOWN
      and stick out 2 mm below Z=0.

Parts
  tray      - back plate + walls that cradle the case; strap rails; lanyard tab
  cover     - front grille, spring tongues, ribbon slit, GPIO slot, tilt-hinge ears
  cam_head  - camera holder, tilts on an M3 bolt; 24-tooth rosette = 15 deg steps
  fit_A_pocket_ring - 4 mm outline of the pocket: 10-minute print to check the fit
"""
import math
import os
import cadquery as cq

# ============================== PARAMETERS ==================================
# --- Raspberry Pi 5 (official mechanical drawing RP-008347-DS) ---
PCB_L = 85.0           # board length
PCB_W = 56.0           # board width
USB_OUT = 3.0          # USB-A / Ethernet overhang past the board edge
SOCKET_OUT = 1.7       # USB-C / micro-HDMI overhang past the board edge
SD_OUT = 1.8           # inserted microSD card past the board edge
D_USBC = 11.2          # along the board from the microSD edge
D_CSI = 54.9           # CAM/DISP connector nearest the Ethernet jack (the other is 48.7)
D_GPIO = (7.1, 57.9)   # 40-pin header
E_CSI = 8.4            # across the board from the HDMI edge: ribbon centre
E_SD = 28.0            # microSD slot centre
E_PWR = 13.3           # power button (on the microSD edge)
E_USB = (47.0, 29.1)   # USB-A stack centres

# --- Aluminium armour dual-fan case (seller spec 87 x 56 x 24; CHECK WITH A RULER) ---
CASE_L = 87.0          # shell length. The shell stands 1 mm proud of the board at each end
CASE_W = PCB_W + SOCKET_OUT   # 57.7: shell is board-width, the sockets stick out on +X
CASE_T = 24.0          # shell thickness, back face to fin/fan tops
PCB_UP = 6.2           # board top surface above the shell's back face (photo estimate)
FAN_X = (-21.3, 3.7)   # the two 25 mm fans, from the photos
FAN_Z = (24.0, 78.5)

CLR = 0.5              # pocket clearance per side (covers ABS shrink)

# --- Shell ---
WALL = 2.4
BACK = 2.4
POCKET_T = CASE_T + 1.0   # wall height above the back plate; the springs take up the slack
FRONT_GAP = 4.0        # fan intake plenum + folded camera ribbon
COVER_T = 2.4
LIP_Z = 3.0            # the tray reaches this far below the shell ledge
STOP_H = 4.0           # ledge height (Y) under the ports; the ports clear it by 2 mm
R_OUT = 3.0            # outer corner radius
BACK_CH = 1.0          # chamfer on the chest-side edge

# --- Fasteners (M3 screws into captive nuts) ---
M3_HOLE = 3.4
M3_HEAD_D = 6.2
M3_NUT_AF = 6.4
M3_NUT_T = 2.9
M2_PILOT = 1.7

# --- Strap rails: two screw pillars per side joined by a bar the band wraps ---
RAIL_X = 7.0           # how far a rail stands out from the wall
SLOT_X = 3.5           # band slot between wall and bar
BAR_T = 5.0            # bar thickness (Y)
PAD_Z = (17.0, 61.0)   # screw pillar centres; the top one stays below the USB-C window
PAD_LEN = 9.0
WEB_SLOT = 27.0        # slot length for 25 mm webbing

# --- Openings ---
USBC_WIN_Y = (BACK + 2.0, BACK + 14.0)
USBC_WIN_HALF = 8.5
GRILLE_SLOT = 3.0
GRILLE_BAR = 2.0
RIBBON_SLIT_L = 20.0
RIBBON_SLIT_W = 3.0    # open at its back edge: the ribbon slides in as the cover goes on

# --- Spring tongues in the cover (press on the fins between the fans and the ribbon) ---
TONGUE_X = (6.2, 11.2)
TONGUE_ROOT_Z = (56.0, 60.0)
TONGUE_L = 22.0
TONGUE_T = 1.6
SPRING_REACH = CASE_T - 0.7   # tips touch any case thicker than this

# --- Camera Module 3 Wide (official drawing RP-008155-DS) ---
CAM_W = 25.0
CAM_H = 23.862
CAM_PCB_T = 1.12
CAM_LENS_UP = 8.3           # lens tip above the PCB front face
CAM_LENS_FROM_TOP = 14.4    # lens centre from the edge opposite the connector
CAM_HOLES = [(-10.5, 2.0), (10.5, 2.0), (-10.5, 14.5), (10.5, 14.5)]  # (x, z from that edge)
LENS_RECESS = 0.5           # lens tip sits this far behind the head's front face
LENS_SLOT_W = 17.0
BOSS_R = 2.2
CAM_CLR = 0.3

# --- Tilt hinge ---
AXIS_UP = 11.5
EAR_T = 4.5
EAR_R = 10.5
TEETH = 24
TOOTH_H = 0.8
HEAD_WALL = 2.0
RIBBON_GAP = 19.0

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
# ============================================================================

PW = CASE_W + 2 * CLR
PL = CASE_L + 2 * CLR
OX = PW / 2 + WALL
Z_TOP = PL + WALL
ZB = -LIP_Z
Y_SPLIT = BACK + POCKET_T
Y_FRONT = Y_SPLIT + FRONT_GAP + COVER_T

X_GPIO_EDGE = -PW / 2 + CLR            # board edges in pocket coordinates
X_HDMI_EDGE = X_GPIO_EDGE + PCB_W
PCB_Z0 = (CASE_L - PCB_L) / 2          # board's port edge above the ledge
PORT_TIP_Z = PCB_Z0 - USB_OUT


def from_sd(d):
    return PCB_Z0 + PCB_L - d


CSI_X = X_HDMI_EDGE - E_CSI
CSI_Z = from_sd(D_CSI)
SD_X = X_HDMI_EDGE - E_SD
PWR_X = X_HDMI_EDGE - E_PWR
USB_X = tuple(X_HDMI_EDGE - e for e in E_USB)
USBC_Z = from_sd(D_USBC)
GPIO_Z = (from_sd(D_GPIO[1]), from_sd(D_GPIO[0]))
GPIO_SLOT_X = (-PW / 2 - 0.4, X_GPIO_EDGE + 6.4)
GPIO_SLOT_Z = (GPIO_Z[0] - 2.0, GPIO_Z[1] + 1.6)
GRILLE_X = (GPIO_SLOT_X[1] + 2.0, FAN_X[1])

SCREW_X = OX + 3.2
RAIL_Z = (PAD_Z[0] - PAD_LEN / 2, PAD_Z[1] + PAD_LEN / 2)
STRAP_Z = sum(PAD_Z) / 2
X_CAM = CSI_X - 0.75                    # keeps the outer hinge ear inside the rail width
AXIS_Y = Y_SPLIT + 5.5
EAR_IN = CAM_W / 2 + CAM_CLR + HEAD_WALL + 0.3 + TOOTH_H
HX = EAR_IN + EAR_T
AXIS_Z = Z_TOP + AXIS_UP


def box(x0, x1, y0, y1, z0, z1):
    return (cq.Workplane("XY").box(x1 - x0, y1 - y0, z1 - z0, centered=False)
            .translate((x0, y0, z0)))


def rbox(x0, x1, y0, y1, z0, z1, r):
    """Box with its four front-view corners rounded."""
    return box(x0, x1, y0, y1, z0, z1).edges("|Y").fillet(r)


def cyl_x(r, x0, x1, y, z):
    return cq.Workplane("YZ").circle(r).extrude(x1 - x0).translate((x0, y, z))


def cyl_y(r, y0, y1, x, z):
    return cq.Workplane("XZ").circle(r).extrude(-(y1 - y0)).translate((x, y0, z))


def side(s, a, b):
    """X range between two offsets on the +X (s=1) or -X (s=-1) side."""
    return sorted((s * a, s * b))


def rosette(x_face, direction, y, z, phase=0.0):
    """Radial teeth on a plane x = x_face, protruding in +X (direction=1) or -X (-1)."""
    r0, r1 = 4.0, EAR_R - 1.0
    p0, p1 = 0.9 * 2 * math.pi * r0 / TEETH, 0.9 * 2 * math.pi * r1 / TEETH
    tri = lambda p: [(0, -p / 2), (TOOTH_H * direction, 0), (0, p / 2)]
    tooth = (cq.Workplane("XY").workplane(offset=r0).polyline(tri(p0)).close()
             .workplane(offset=r1 - r0).polyline(tri(p1)).close().loft())
    ring = None
    for i in range(TEETH):
        t = tooth.rotate((0, 0, 0), (1, 0, 0), 360.0 * (i + phase) / TEETH)
        ring = t if ring is None else ring.union(t)
    return ring.translate((x_face, y, z))


def in_rail(z0, z1):
    return RAIL_Z[0] - 1 < z1 and z0 < RAIL_Z[1] + 1


# ------------------------------------------------------------------ TRAY
def make_tray():
    t = rbox(-OX, OX, 0, Y_SPLIT, ZB, Z_TOP, R_OUT).faces("<Y").chamfer(BACK_CH)
    for s in (-1, 1):                                   # strap rails
        xa, xb = side(s, OX - 1.0, OX + RAIL_X)
        rail = rbox(xa, xb, 0, Y_SPLIT, RAIL_Z[0], RAIL_Z[1], 2.0).faces("<Y").chamfer(BACK_CH)
        rail = rail.cut(box(xa - 1, xb + 1, BAR_T, Y_SPLIT + 1,
                            PAD_Z[0] + PAD_LEN / 2, PAD_Z[1] - PAD_LEN / 2))
        t = t.union(rail)
    t = t.union(rbox(-8, 8, 0, 3.5, Z_TOP - 3, Z_TOP + 8, 2.0))        # lanyard tab
    # ---- cuts
    for s in (-1, 1):                                   # band slots
        xa, xb = side(s, OX, OX + SLOT_X)
        t = t.cut(box(xa, xb, -1, BAR_T + 1, STRAP_Z - WEB_SLOT / 2, STRAP_Z + WEB_SLOT / 2))
    t = t.cut(box(-5, 5, -1, 4.5, Z_TOP + 2.5, Z_TOP + 5.5))            # lanyard slot
    # top wall: microSD window, power-button poke hole, exhaust slots (hot air rises out here)
    t = t.cut(box(SD_X - 9, SD_X + 9, BACK + 1.5, BACK + 6.5, PL - 1, Z_TOP + 1))
    t = t.cut(cq.Workplane("XY").circle(2.5).extrude(WALL + 2)
              .translate((PWR_X, BACK + PCB_UP + 1.5, PL - 1)))
    xv = -PW / 2 + 3
    while xv + 3 <= PW / 2 - 2:
        t = t.cut(box(xv, xv + 3, BACK + 9.0, Y_SPLIT - 1.6, PL - 1, Z_TOP + 1))
        xv += 5.5
    # bottom of the GPIO-side wall opened so a wide USB-A plug in the outer port clears it
    t = t.cut(box(-OX - 0.1, -PW / 2 + 0.1, BACK + STOP_H, Y_SPLIT + 1, ZB - 1, 1.0))
    # USB-C window, +X wall (top-left from the front)
    t = t.cut(box(PW / 2 - 0.1, OX + 1, USBC_WIN_Y[0], USBC_WIN_Y[1],
                  USBC_Z - USBC_WIN_HALF, USBC_Z + USBC_WIN_HALF))
    for s in (-1, 1):                                   # side ladder vents
        zv = 3.0
        while zv + 3 <= PL - 3:
            blocked = in_rail(zv, zv + 3) or (
                s > 0 and USBC_Z - USBC_WIN_HALF - 2 < zv + 3 and zv < USBC_Z + USBC_WIN_HALF + 2)
            if not blocked:
                xa, xb = side(s, PW / 2 - 0.1, OX + 0.5)
                t = t.cut(box(xa, xb, BACK + 2.5, Y_SPLIT - 2.0, zv, zv + 3))
            zv += 6.0
    for s in (-1, 1):                                   # screws + nut slots
        for zp in PAD_Z:
            t = t.cut(cyl_y(M3_HOLE / 2, Y_SPLIT - 16, Y_SPLIT + 1, s * SCREW_X, zp))
            xa, xb = side(s, SCREW_X, OX + RAIL_X + 1)
            t = t.cut(box(xa, xb, Y_SPLIT - 9 - M3_NUT_T / 2, Y_SPLIT - 9 + M3_NUT_T / 2,
                          zp - M3_NUT_AF / 2, zp + M3_NUT_AF / 2))
    t = t.cut(box(-PW / 2, PW / 2, BACK, Y_SPLIT + 1, 0, PL))                    # pocket
    t = t.cut(box(-PW / 2, PW / 2, BACK + STOP_H, Y_SPLIT + 1, ZB - 1, 0.01))    # open bottom
    return t


# ------------------------------------------------------------------ COVER
def tongue_spans():
    """(root z, tip z) for the two spring tongues, pointing away from each other."""
    return [(TONGUE_ROOT_Z[1], TONGUE_ROOT_Z[1] + TONGUE_L),
            (TONGUE_ROOT_Z[0], TONGUE_ROOT_Z[0] - TONGUE_L)]


def make_cover(springs=True):
    y0, y1, y2 = Y_SPLIT, Y_SPLIT + FRONT_GAP, Y_FRONT
    c = rbox(-OX, OX, y0, y2, ZB, Z_TOP, R_OUT).faces(">Y").chamfer(BACK_CH)
    c = c.cut(box(-PW / 2, PW / 2, y0 - 1, y1, ZB - 1, PL))      # leaves side + top skirts
    # solid block under the outer hinge ear (it overhangs the side wall)
    c = c.union(box(OX - 0.01, X_CAM + HX, y0, y2, PL - 6, Z_TOP))
    for s in (-1, 1):                                            # screw pads
        for zp in PAD_Z:
            xa, xb = side(s, OX - 1.0, OX + RAIL_X)
            c = c.union(rbox(xa, xb, y0, y2, zp - PAD_LEN / 2, zp + PAD_LEN / 2, 2.0))
    for s in (-1, 1):                                            # hinge ears, centred on X_CAM
        xa, xb = sorted((X_CAM + s * EAR_IN, X_CAM + s * HX))
        ear = cyl_x(EAR_R, xa, xb, AXIS_Y, AXIS_Z)
        ear = ear.union(box(xa, xb, AXIS_Y - EAR_R, AXIS_Y + EAR_R * 0.4, Z_TOP - 0.01, AXIS_Z))
        ear = ear.intersect(box(xa, xb, y0 - 20, y2, Z_TOP - 0.01, AXIS_Z + EAR_R + 1))
        c = c.union(ear)
        c = c.union(rosette(X_CAM + s * EAR_IN, -s, AXIS_Y, AXIS_Z, phase=0.5)
                    .intersect(box(-80, 80, y0 - 20, y2, Z_TOP, AXIS_Z + 20)))
    # ---- cuts
    c = c.cut(box(-80, 80, y0 - 30, y0, ZB - 1, Z_TOP + 0.3))    # ears clear the tray top
    rib_x = sum(GRILLE_X) / 2                                    # fan grille, centre rib
    z = FAN_Z[0]
    while z + GRILLE_SLOT <= FAN_Z[1]:
        for xa, xb in ((GRILLE_X[0], rib_x - 1.0), (rib_x + 1.0, GRILLE_X[1])):
            c = c.cut(box(xa, xb, y1 - 1, y2 + 1, z, z + GRILLE_SLOT))
        z += GRILLE_SLOT + GRILLE_BAR
    c = c.cut(box(GPIO_SLOT_X[0], GPIO_SLOT_X[1], y0 - 1, y2 + 1, GPIO_SLOT_Z[0], GPIO_SLOT_Z[1]))
    # ribbon slit in the top skirt, above the CSI connector, behind the hinge bolt
    c = c.cut(box(CSI_X - RIBBON_SLIT_L / 2, CSI_X + RIBBON_SLIT_L / 2,
                  y0 - 1.0, y0 + RIBBON_SLIT_W, PL - 1, Z_TOP + 1))
    for s in (-1, 1):                                            # side-skirt vents
        for zv in range(5, int(PL) - 8, 7):
            if in_rail(zv, zv + 4):
                continue
            xa, xb = side(s, PW / 2 - 1, OX + 1)
            c = c.cut(box(xa, xb, y0 + 0.8, y1 - 0.5, zv, zv + 4))
    for s in (-1, 1):                                            # screw holes, counterbored
        for zp in PAD_Z:
            c = c.cut(cyl_y(M3_HOLE / 2, y0 - 1, y2 + 1, s * SCREW_X, zp))
            c = c.cut(cyl_y(M3_HEAD_D / 2, y2 - 3.0, y2 + 1, s * SCREW_X, zp))
    # hinge bolt: nut trap on the outer (+X) ear, head counterbore on the inner ear
    c = c.cut(cyl_x(M3_HOLE / 2, X_CAM - HX - 1, X_CAM + HX + 1, AXIS_Y, AXIS_Z))
    c = c.cut(cq.Workplane("YZ").polygon(6, M3_NUT_AF / math.cos(math.pi / 6)).extrude(3.0)
              .translate((X_CAM + HX - 2.6, AXIS_Y, AXIS_Z)))
    c = c.cut(cyl_x(M3_HEAD_D / 2, X_CAM - HX - 1, X_CAM - HX + 2.6, AXIS_Y, AXIS_Z))
    # spring tongues: a U-slot frees each one, the inside is thinned, a post reaches the fins
    x0, x1 = TONGUE_X
    for root, tip in tongue_spans():
        d = 1 if tip > root else -1
        za, zb = sorted((root, tip + d * 1.0))
        slot = box(x0 - 1.0, x1 + 1.0, y1 - 1, y2 + 1, za, zb)
        c = c.cut(slot.cut(box(x0, x1, y1 - 2, y2 + 2, *sorted((root, tip)))))
        c = c.cut(box(x0, x1, y1 - 1, y2 - TONGUE_T, *sorted((root, tip))))
        if springs:
            c = c.union(box(x0, x1, BACK + SPRING_REACH, y2 - TONGUE_T, *sorted((tip - d * 4.0, tip))))
    return c


# ------------------------------------------------------------------ CAMERA HEAD
def make_cam_head():
    """Local coords: hinge axis at origin, lens looking +Y, PCB in the XZ plane.
    PCB connector edge at the bottom; the ribbon drops through the middle gap behind the bolt."""
    iw = CAM_W + 2 * CAM_CLR
    ow = iw + 2 * HEAD_WALL
    yf = 10.5                         # front face
    y_pcb_front = yf - (CAM_LENS_UP + LENS_RECESS)
    y_back = y_pcb_front - CAM_PCB_T - 5.2
    z_pcb_bot = 5.8
    z_pcb_top = z_pcb_bot + CAM_H
    z_top = z_pcb_top + CAM_CLR + HEAD_WALL
    fw = 2.5                          # front wall thickness

    h = box(-ow / 2, ow / 2, yf - fw, yf, z_pcb_bot - 1.5, z_top)                 # front wall
    h = h.union(box(-ow / 2, -iw / 2, y_back, yf, z_pcb_bot - 1.5, z_top))       # side walls
    h = h.union(box(iw / 2, ow / 2, y_back, yf, z_pcb_bot - 1.5, z_top))
    h = h.union(box(-ow / 2, ow / 2, y_back, yf, z_pcb_top + CAM_CLR, z_top))    # roof
    for s in (-1, 1):
        xa, xb = sorted((s * RIBBON_GAP / 2, s * ow / 2))
        h = h.union(box(xa, xb, y_back, yf, z_pcb_bot - 1.5, z_pcb_bot - 0.01))  # bottom ledge
        lug = cyl_x(EAR_R, xa, xb, 0, 0).union(box(xa, xb, -EAR_R, EAR_R * 0.4, 0, z_pcb_bot))
        h = h.union(lug.intersect(box(xa, xb, y_back - 5, yf, -EAR_R - 1, z_pcb_bot)))
    for (cx, cz) in CAM_HOLES:                                   # bosses the PCB front sits on
        zz = z_pcb_top - cz
        h = h.union(cyl_y(BOSS_R, y_pcb_front, yf - fw + 0.01, cx, zz))
        xa, xb = (cx - 0.01, iw / 2 + 0.01) if cx > 0 else (-iw / 2 - 0.01, cx + 0.01)
        h = h.union(box(xa, xb, y_pcb_front, yf - fw + 0.01, zz - 1.4, zz + 1.4))
    for s in (-1, 1):
        h = h.union(rosette(s * ow / 2, s, 0, 0))
    # ---- cuts
    lens_z = z_pcb_top - CAM_LENS_FROM_TOP
    h = h.cut(box(-LENS_SLOT_W / 2, LENS_SLOT_W / 2, yf - fw - 1, yf + 1,
                  z_pcb_bot - 3.0, z_pcb_top + CAM_CLR))         # open lens slot
    for sx in (-1, 1):                                           # bevel it out of the wide view
        h = h.cut(cq.Workplane("XY").polyline([(0, 0), (sx * 1.8, 0), (0, -1.8)]).close()
                  .extrude(z_pcb_top + CAM_CLR - (z_pcb_bot - 3.0))
                  .translate((sx * LENS_SLOT_W / 2, yf, z_pcb_bot - 3.0)))
    for (cx, cz) in CAM_HOLES:
        h = h.cut(cyl_y(M2_PILOT / 2, y_pcb_front - 0.5, yf - 0.8, cx, z_pcb_top - cz))
    h = h.cut(box(-iw / 2, iw / 2, y_back - 1, y_pcb_front, z_pcb_bot, z_pcb_top + CAM_CLR))
    h = h.cut(box(-RIBBON_GAP / 2, RIBBON_GAP / 2, y_back - 1, yf - fw, -EAR_R - 2, z_pcb_bot + 0.5))
    h = h.cut(cyl_x(M3_HOLE / 2, -ow, ow, 0, 0))
    return h, dict(ow=ow, yf=yf, y_back=y_back, y_pcb_front=y_pcb_front,
                   z_pcb_bot=z_pcb_bot, lens_z=lens_z, z_top=z_top)


def place_head(h, tilt_deg=0.0):
    return h.rotate((0, 0, 0), (1, 0, 0), -tilt_deg).translate((X_CAM, AXIS_Y, AXIS_Z))


# ------------------------------------------------------------------ COUPON
def fit_A_pocket_ring():
    """4 mm outline of the pocket. Lay the case in it, ports over the closed end."""
    H = 4.0
    r = rbox(-OX, OX, 0, H, ZB, Z_TOP, R_OUT)
    r = r.cut(box(-PW / 2, PW / 2, -1, H + 1, 0, PL))
    r = r.cut(box(SD_X - 9, SD_X + 9, 1.5, H + 1, PL - 0.01, Z_TOP + 1))    # microSD relief
    return r


# ------------------------------------------------------------------ EXPORT
def export(wp, name, step=True):
    os.makedirs(OUT, exist_ok=True)
    bb = wp.val().BoundingBox()
    wp = wp.translate((-(bb.xmin + bb.xmax) / 2, -(bb.ymin + bb.ymax) / 2, -bb.zmin))
    cq.exporters.export(wp, os.path.join(OUT, name + ".stl"), tolerance=0.02, angularTolerance=0.15)
    if step:
        cq.exporters.export(wp, os.path.join(OUT, name + ".step"))


if __name__ == "__main__":
    tray, cover = make_tray(), make_cover()
    head, info = make_cam_head()
    # print orientations (same as v0.5): tray back-down, cover front-down, head front-down
    export(tray.rotate((0, 0, 0), (1, 0, 0), 90), "tray_print")
    export(cover.rotate((0, 0, 0), (1, 0, 0), -90), "cover_print")
    export(head.rotate((0, 0, 0), (1, 0, 0), -90), "cam_head_print")
    export(fit_A_pocket_ring().rotate((0, 0, 0), (1, 0, 0), 90), "fit_A_pocket_ring", step=False)
    asm = (cq.Assembly().add(tray, name="tray").add(cover, name="cover")
           .add(place_head(head, 15), name="cam_head"))
    asm.export(os.path.join(OUT, "assembly_as_worn.step"))
    print("pocket %.1f x %.1f x %.1f | body %.1f wide, %.1f deep, %.1f tall"
          % (PW, PL, POCKET_T, 2 * (OX + RAIL_X), Y_FRONT, Z_TOP - ZB))
