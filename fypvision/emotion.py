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


def _classify(face_bgr):
    """(emotion, confidence) for one cropped face."""
    grey = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(grey, (FER_INPUT_SIZE, FER_INPUT_SIZE),
                         interpolation=cv2.INTER_AREA)
    # FER+ takes raw 0-255 intensities, not a 0-1 normalised blob.
    blob = resized.astype(np.float32).reshape(1, 1, FER_INPUT_SIZE, FER_INPUT_SIZE)
    _emotion_net.setInput(blob)
    probabilities = _softmax(_emotion_net.forward()[0])
    index = int(np.argmax(probabilities))
    return EMOTIONS[index], float(probabilities[index])


def detect_faces(frame_bgr) -> list:
    """Faces with expressions: [{box, emotion, conf, face_score}, ...]."""
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
        crop = frame_bgr[y:y + h, x:x + w]
        if crop.size == 0:
            continue
        try:
            emotion, confidence = _classify(crop)
        except Exception as e:
            print(f"[EMOTION] classification failed: {e}")
            continue
        results.append({
            "box": (x, y, x + w, y + h),
            "label": emotion,
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
        if face["conf"] < CONFIDENCE_FLOOR:
            parts.append("someone whose expression I cannot read")
        else:
            parts.append(f"someone who looks {face['label']}")

    if len(faces) == 1:
        # Not str.capitalize(): it lowercases the rest, turning "I" into "i".
        return parts[0][0].upper() + parts[0][1:] + "."
    listing = ", ".join(parts[:-1]) + " and " + parts[-1]
    return f"{len(faces)} faces: {listing}."
