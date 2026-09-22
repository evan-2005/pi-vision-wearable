#!/usr/bin/env bash
# =============================================================================
# FYP Assistive Vision Wearable - one-command installer
# =============================================================================
# Target: Raspberry Pi 5, Raspberry Pi OS Bookworm 64-bit, Camera Module 3
#
# Usage (on a freshly flashed Pi with network access):
#     git clone <your-repo-url> ~/fyp-src
#     cd ~/fyp-src && bash install.sh
#
# Re-running is safe: every step is idempotent (skips what already exists).
#
# What this does:
#   1. apt packages   (tesseract, bluez/bluealsa, python tooling, camera libs)
#   2. Disables PipeWire's Bluetooth module (it fights BlueALSA for the headset)
#   3. ~/fyp project dir + Python venv (--system-site-packages: REQUIRED so
#      the apt-installed picamera2 stays importable inside the venv)
#   4. pip packages   (ultralytics, ncnn, opencv, pytesseract, piper-tts)
#   5. YOLOv8n download + NCNN export
#   6. Piper voice model download (en_US-lessac-medium)
#   7. Copies main.py into ~/fyp
#   8. Installs a systemd service (disabled by default; see end of output)
#
# What this does NOT do (one-time manual steps):
#   - Bluetooth pairing: pair/trust/connect your headset via bluetoothctl,
#     then put its MAC in fyp.service (FYP_HEADSET_MAC) or main.py.
#   - Enabling the service: do that only after the GPIO button is wired.
# =============================================================================
set -euo pipefail

FYP_DIR="$HOME/fyp"
VENV="$FYP_DIR/venv"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

log()  { echo -e "\n\033[1;32m==> $*\033[0m"; }
warn() { echo -e "\033[1;33m[warn]\033[0m $*"; }

# -----------------------------------------------------------------------------
log "1/8 Installing apt packages"
# -----------------------------------------------------------------------------
sudo apt-get update
sudo apt-get install -y \
    python3-pip python3-venv git \
    tesseract-ocr libtesseract-dev \
    bluez bluez-alsa-utils \
    python3-picamera2 rpicam-apps \
    alsa-utils

# -----------------------------------------------------------------------------
log "2/8 Disabling PipeWire's Bluetooth handling"
# -----------------------------------------------------------------------------
# Raspberry Pi OS ships PipeWire + WirePlumber, which claim any paired headset
# and park it in HFP/mSBC over a SCO link as the default sink AND source. With
# BlueALSA also driving the same headset over A2DP you get two concurrent
# Bluetooth audio links to one device, fighting each other. BlueALSA owns audio
# here, so take PipeWire out of Bluetooth entirely.
sudo mkdir -p /etc/wireplumber/wireplumber.conf.d
sudo tee /etc/wireplumber/wireplumber.conf.d/50-disable-bluez.conf > /dev/null <<'WPEOF'
# Installed by the FYP wearable installer. BlueALSA drives the headset, so
# WirePlumber must not also claim it. Delete this file to hand Bluetooth back
# to PipeWire - and then use one stack or the other, never both.
wireplumber.profiles = {
  main = {
    monitor.bluez = disabled
    monitor.bluez.seat-monitoring = disabled
  }
}
WPEOF

if XDG_RUNTIME_DIR="/run/user/$(id -u)"    systemctl --user restart wireplumber pipewire-pulse 2>/dev/null; then
    echo "    wireplumber restarted - Bluetooth is now BlueALSA-only"
else
    warn "could not restart wireplumber from here; reboot to apply"
fi

# -----------------------------------------------------------------------------
log "3/8 Creating project directory and virtual environment"
# -----------------------------------------------------------------------------
mkdir -p "$FYP_DIR/voices"

if [ ! -d "$VENV" ]; then
    # --system-site-packages is NOT optional: picamera2 comes from apt and
    # must remain importable inside the venv on Bookworm.
    python3 -m venv --system-site-packages "$VENV"
else
    warn "venv already exists, skipping creation"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

# -----------------------------------------------------------------------------
log "4/8 Installing Python packages"
# -----------------------------------------------------------------------------
pip install --upgrade pip
pip install \
    "ultralytics>=8.4" \
    ncnn \
    opencv-python \
    pytesseract \
    "piper-tts>=1.4"

# -----------------------------------------------------------------------------
log "5/8 YOLOv8n -> NCNN export"
# -----------------------------------------------------------------------------
cd "$FYP_DIR"
if [ ! -d "$FYP_DIR/yolov8n_ncnn_model" ]; then
    # Downloads yolov8n.pt automatically if absent, then exports.
    # Export runs on-device; takes a few minutes on a Pi 5.
    yolo export model=yolov8n.pt format=ncnn
else
    warn "yolov8n_ncnn_model already exists, skipping export"
fi

# -----------------------------------------------------------------------------
log "6/8 Downloading Piper voice (en_US-lessac-medium)"
# -----------------------------------------------------------------------------
if [ ! -f "$FYP_DIR/voices/en_US-lessac-medium.onnx" ]; then
    cd "$FYP_DIR/voices"
    python -m piper.download_voices en_US-lessac-medium
else
    warn "voice model already present, skipping download"
fi

# -----------------------------------------------------------------------------
log "7/8 Installing main.py"
# -----------------------------------------------------------------------------
if [ -f "$REPO_DIR/main.py" ]; then
    cp "$REPO_DIR/main.py" "$FYP_DIR/main.py"
else
    warn "main.py not found next to install.sh; copy it to $FYP_DIR manually"
fi

# -----------------------------------------------------------------------------
log "8/8 Installing systemd service (left disabled)"
# -----------------------------------------------------------------------------
sudo tee /etc/systemd/system/fyp.service > /dev/null <<EOF
[Unit]
Description=FYP Assistive Vision Wearable
# Bluetooth must be up before we try to speak through it
After=bluetooth.target sound.target
Wants=bluetooth.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$FYP_DIR
Environment=FYP_USE_GPIO=1
# Set your paired headset MAC here (or edit the default in main.py):
Environment=FYP_HEADSET_MAC=3C:B0:ED:B6:18:A9
ExecStart=$VENV/bin/python $FYP_DIR/main.py
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF
sudo systemctl daemon-reload

# -----------------------------------------------------------------------------
echo
log "Install complete."
cat <<'DONE'

Manual steps remaining (one-time):

  1. Pair your Bluetooth headset (headset in pairing mode first):
         bluetoothctl
         power on
         scan on            # find your headset's MAC
         scan off
         pair <MAC>
         trust <MAC>        # required for auto-reconnect after reboot
         connect <MAC>
         exit
     Then set the MAC in /etc/systemd/system/fyp.service (FYP_HEADSET_MAC)
     if it differs from the default.

  2. Bench test (keyboard mode, no button needed):
         cd ~/fyp && source venv/bin/activate && python3 main.py

  3. Once the GPIO button is wired (GPIO17 <-> GND) and tested:
         sudo systemctl enable --now fyp.service
         journalctl -u fyp.service -f      # live logs

DONE
