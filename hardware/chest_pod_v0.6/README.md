# Chest pod v0.6

Raspberry Pi 5 in its aluminium armour dual-fan case + Camera Module 3 Wide, worn on one chest band.
Same layout and print orientation as v0.5; resized from published dimensions.

**Print this:** `out/plate_v06_all_parts.3mf` (tray, cover, camera head; 256 mm bed).

## Size

| | Printed v0.2 (`plate_2_final_parts`) | v0.5 | **v0.6** |
|---|---|---|---|
| Pocket (W × L × depth) | 65.6 × 100.6 × 31.5 | 60.3 × 89.8 × 30.9 | **58.7 × 88.0 × 25.0** |
| Body width incl. strap loops | 94.4 | 89.1 | **77.5** |
| Body depth off the chest | about 48 | about 40 | **33.8** |
| Plastic | 83 cm³ | 58 cm³ | **57 cm³** |

## Where the numbers come from

| Dimension | Value | Source |
|---|---|---|
| Pi 5 board | 85 × 56 mm | [Pi 5 mechanical drawing](https://datasheets.raspberrypi.com/rpi5/raspberry-pi-5-mechanical-drawing.pdf) |
| USB-A / Ethernet overhang | 3.0 mm | same |
| USB-C / HDMI overhang | 1.7 mm | measured off the same drawing's geometry |
| USB-C centre | 11.2 mm from the microSD edge | same |
| Camera connector used | 54.9 mm from the microSD edge, ribbon centre 8.4 mm from the HDMI edge | measured off the drawing; matches your photos |
| Armour dual-fan case | 87 × 56 × 24 mm | [seller spec for this style of case](https://www.kiwi-electronics.com/en/heat-sink-case-with-dual-fan-for-rpi-5-black-19935); your photos agree on thickness |
| Camera Module 3 Wide | PCB 25 × 23.862 × 1.12, holes 21 × 12.5, lens centre 14.4 from the far edge, lens tip 8.3 above the PCB | [Camera Module 3 Wide drawing](https://datasheets.raspberrypi.com/camera/camera-module-3-wide-mechanical-drawing.pdf) |
| Fan positions, board height inside the case (6.2 mm) | | estimated from your photos |

## Check before the 4-hour print

The case size is a seller's figure, not a measurement of your unit. With a ruler:

1. **Thickness**, back of the case to the top of the fans: expected 24 mm. Anything from 23.5 to 25 fits (spring tongues in the cover take up the difference).
2. **Width** across the board including the USB-C/HDMI socket tips: expected 57.7 mm, pocket is 58.7.
3. **Length** of the aluminium shell, without the USB ports: expected 87 mm, pocket is 88.

Or print `out/fit_A_pocket_ring.stl` (10 minutes) and lay the case in it.
If a number is off, change `CASE_T` / `CASE_W` / `CASE_L` in `chestpod.py` and re-run.

## What changed from v0.5

- **Thinner:** the pocket was 31.5 mm deep for a 24 mm case.
- **Narrower:** the 12 mm strap wings and separate screw pads are now one slim rail per side. The band wraps the bar between the two screw pillars.
- **Spring tongues** in the cover press the case onto the back plate, so it does not rattle and a slightly thicker or thinner case still fits.
- **Comfort:** rounded corners, chamfered chest-side edge, lower lanyard tab.
- **Camera head:** lens height corrected to 8.3 mm (was 11.3), so the head is shallower and the lens sits 0.5 mm behind the front face.
- **Power-button poke hole** in the top wall, next to the microSD window.
- **Cooling:** unchanged idea. Fans draw through the front grille, air leaves through the top slots and the open bottom.

## Layout (seen from the front, as someone else sees it worn)

- Camera head top LEFT, above the ribbon
- USB-C window in the LEFT wall near the top
- Ethernet bottom left, USB-A ×2 bottom centre and right; the bottom is open
- GPIO slot on the RIGHT of the cover (button wires for pins 1, 9, 11 come out here)
- Fan grille right of centre

## Print

0.20 mm layers, 3 walls, 20 % infill, no supports, brim on. ABS or PETG (PLA softens at 55–60 °C).
Tray prints back-down, cover and camera head front-down, as on the v0.5 plate.

## Hardware (unchanged)

- 4 × M3×16 + 4 × M3 nuts: cover to tray (nuts slide into the pillar slots from the side)
- 1 × M3×40 + nut: camera tilt hinge
- 4 × M2 self-tapping screws: camera to head
- 25 mm webbing band; optional neck cord through the top tab

## Assembly

1. Slide the 4 nuts into the pillar slots.
2. Stand the case in the tray, ports toward the open bottom, USB-C facing the left-wall window.
3. Lay the ribbon up the left column on top of the case. Store the spare length as one flat S-fold there.
4. Screw the camera into the head, plug the ribbon in from below, bolt the head between the ears with the ribbon behind the bolt.
5. Lower the cover; the ribbon slides into the open-backed slit. Fit the 4 screws.
6. Thread the band: in front of each bar, through the slot, across the back of the pod.

## Regenerate

```
pip install cadquery trimesh matplotlib
python chestpod.py && python check.py && python plates.py
```

`check.py` fails if the case, plugs, ribbon, button wires or camera collide with any part, at tilts from -15° to 45°.
