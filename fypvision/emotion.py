"""Read facial expression from a frame, for a blind or low-vision wearer.

Two small ONNX models, both run offline through OpenCV's DNN module:
  - YuNet detects faces (this OpenCV build ships no Haar cascades)
  - FER+ classifies each face into one of eight expressions

Both load lazily, so a wearer who never uses this mode pays nothing for it.

Read the output with care. FER+ was trained on frontal, well-lit, cropped
faces; a chest-mounted camera gives it small, angled, motion-blurred ones,
so accuracy is markedly worse than the published benchmarks. Expression
also is not feeling - a neutral face is not a neutral person. Phrasing here
stays deliberately hedged ("looks happy", never "is happy"), and anything
below CONFIDENCE_FLOOR is reported as uncertain rather than guessed at.
"""

import math
import os
from pathlib import Path

import cv2
import numpy as np

MODEL_DIR = Path(os.environ.get("FYP_MODEL_DIR", str(Path.home() / "fyp" / "models")))
FACE_MODEL = MODEL_DIR / "face_detection_yunet_2023mar.onnx"
EMOTION_MODEL = MODEL_DIR / "emotion-ferplus-8.onnx"

# FER+ output order, fixed by the model.
EMOTIONS = ["neutral", "happy", "surprised", "sad",
            "angry", "disgusted", "afraid", "contemptuous"]

FACE_SCORE_THRESHOLD = 0.75   # YuNet detection confidence
CONFIDENCE_FLOOR = 0.40       # below this, say "unclear" instead of guessing
FER_INPUT_SIZE = 64           # FER+ expects 64x64 greyscale
MAX_FACES_SPOKEN = 3
# FER+ was trained on crops a little looser than YuNet's box, which stops at
# the hairline and chin; the crop side is the box's longer side times this.
FACE_CROP_MARGIN = 1.1
# A face narrower than this many pixels has too little detail left to read
# an expression from; it is reported as too far away instead of guessed at.
MIN_FACE_PX = 40

_face_detector = None
_emotion_net = None


def available() -> bool:
    """True if both model files are present."""
    return FACE_MODEL.exists() and EMOTION_MODEL.exists()


def _load():
    """Load both nets once. Returns False if the models are missing."""
    global _face_detector, _emotion_net
    if _face_detector is not None and _emotion_net is not None:
        return True
    if not available():
        print(f"[EMOTION] models not found in {MODEL_DIR}")
        return False
    try:
        _face_detector = cv2.FaceDetectorYN.create(
            str(FACE_MODEL), "", (320, 320), FACE_SCORE_THRESHOLD)
        _emotion_net = cv2.dnn.readNetFromONNX(str(EMOTION_MODEL))
        return True
    except Exception as e:
        print(f"[EMOTION] failed to load models: {e}")
        _face_detector = _emotion_net = None
        return False


def _softmax(x):
    shifted = x - np.max(x)
    exp = np.exp(shifted)
    return exp / exp.sum()


def _aligned_face(frame_bgr, face) -> np.ndarray:
    """Square greyscale crop of one YuNet face, eyes level, FER+ sized.

    A chest-worn camera sees heads tilted and off-square. FER+ only ever saw
    upright, centred faces, so the crop is rotated about the face centre
    until the line between the eyes is horizontal.
    """
    x, y, w, h = (float(v) for v in face[:4])
    right_eye_x, right_eye_y, left_eye_x, left_eye_y = (
        float(v) for v in face[4:8])
    roll = math.degrees(math.atan2(left_eye_y - right_eye_y,
                                   left_eye_x - right_eye_x))
    scale = FER_INPUT_SIZE / (FACE_CROP_MARGIN * max(w, h))
    centre = (x + w / 2.0, y + h / 2.0)
    matrix = cv2.getRotationMatrix2D(centre, roll, scale)
    matrix[:, 2] += (FER_INPUT_SIZE / 2.0 - centre[0],
                     FER_INPUT_SIZE / 2.0 - centre[1])
    crop = cv2.warpAffine(frame_bgr, matrix, (FER_INPUT_SIZE, FER_INPUT_SIZE),
                          flags=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)
    return cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)


def _classify(face_grey) -> tuple:
    """(emotion, confidence, runner-up) for one aligned 64x64 face."""
    # FER+ takes raw 0-255 intensities, not a 0-1 normalised blob.
    blob = face_grey.astype(np.float32).reshape(
        1, 1, FER_INPUT_SIZE, FER_INPUT_SIZE)
    _emotion_net.setInput(blob)
    probabilities = _softmax(_emotion_net.forward()[0])
    first, second = (int(i) for i in np.argsort(probabilities)[::-1][:2])
    return (EMOTIONS[first], float(probabilities[first]),
            f"{EMOTIONS[second]} {probabilities[second]:.2f}")


def detect_faces(frame_bgr) -> list:
    """Faces with expressions: [{box, label, conf, face_score}, ...].

    `label` is None for a face too small to read an expression from.
    """
    if not _load():
        return []

    height, width = frame_bgr.shape[:2]
    _face_detector.setInputSize((width, height))
    _, faces = _face_detector.detect(frame_bgr)
    if faces is None:
        return []

    results = []
    for face in faces:
        x, y, w, h = (int(v) for v in face[:4])
        # Clamp: YuNet can return boxes that run past the frame edge.
        x, y = max(0, x), max(0, y)
        w, h = min(w, width - x), min(h, height - y)
        if w <= 0 or h <= 0:
            continue
        label, confidence = None, 0.0
        if min(w, h) >= MIN_FACE_PX:
            try:
                label, confidence, runner_up = _classify(
                    _aligned_face(frame_bgr, face))
                print(f"[EMOTION] face {w}x{h}: {label} {confidence:.2f} "
                      f"(then {runner_up})")
            except Exception as e:
                print(f"[EMOTION] classification failed: {e}")
                continue
        results.append({
            "box": (x, y, x + w, y + h),
            "label": label,
            "conf": confidence,
            "face_score": float(face[-1]),
        })
    return results


def describe(faces: list) -> str:
    """Phrase the expressions for speech, hedged to match real accuracy."""
    if not faces:
        return "I cannot see anyone's face."

    # Biggest face first: the closest person is the one being spoken to.
    ordered = sorted(
        faces, key=lambda f: (f["box"][2] - f["box"][0]) * (f["box"][3] - f["box"][1]),
        reverse=True)[:MAX_FACES_SPOKEN]

    parts = []
    for face in ordered:
        if face["label"] is None:
            parts.append("someone too far away to read")
        elif face["conf"] < CONFIDENCE_FLOOR:
            parts.append("someone whose expression I cannot read")
        else:
            parts.append(f"someone who looks {face['label']}")

    if len(parts) == 1 and len(faces) == 1:
        # Not str.capitalize(): it lowercases the rest, turning "I" into "i".
        return parts[0][0].upper() + parts[0][1:] + "."
    listing = ", ".join(parts[:-1]) + " and " + parts[-1]
    if len(faces) > len(parts):
        return f"{len(faces)} faces. The nearest are {listing}."
    return f"{len(faces)} faces: {listing}."
