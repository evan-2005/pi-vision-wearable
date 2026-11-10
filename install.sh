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
#   3. Stops NetworkManager's endless DHCP retry loop on eth0
#   4. ~/fyp project dir + Python venv (--system-site-packages: REQUIRED so
#      the apt-installed picamera2 stays importable inside the venv)
#   5. pip packages   (ultralytics, ncnn, opencv, pytesseract, piper-tts)
#   6. YOLOv8n download + NCNN export
#   7. Piper voice + feature models (face, emotion, offline speech-to-text)
#   8. Copies main.py and fypvision/ into ~/fyp
#   9. Installs a systemd service (disabled by default; see end of output)
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
log "1/9 Installing apt packages"
# -----------------------------------------------------------------------------
sudo apt-get update
sudo apt-get install -y \
    python3-pip python3-venv git \
    tesseract-ocr libtesseract-dev \
    bluez bluez-alsa-utils \
    unzip curl \
    python3-picamera2 rpicam-apps \
    alsa-utils

# -----------------------------------------------------------------------------
log "2/9 Configuring Bluetooth audio (PipeWire off, HFP on)"
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

if XDG_RUNTIME_DIR="/run/user/$(id -u)" \
   systemctl --user restart wireplumber pipewire-pulse 2>/dev/null; then
    echo "    wireplumber restarted - Bluetooth is now BlueALSA-only"
else
    warn "could not restart wireplumber from here; reboot to apply"
fi

# Voice input needs the headset microphone, which exists only over HFP. The Pi
# acts as the Audio Gateway (hfp-ag), the headset as the hands-free device.
# A2DP playback is unaffected: main.py never holds the mic open while speaking,
# because an SCO link plus A2DP music together are what made this headset
# stutter in the first place.
sudo mkdir -p /etc/systemd/system/bluealsa.service.d
sudo tee /etc/systemd/system/bluealsa.service.d/10-hfp.conf > /dev/null <<'HFPEOF'
# Installed by the FYP wearable installer: adds HFP so the headset mic can be
# recorded from. Delete this file to return to playback-only Bluetooth.
[Service]
ExecStart=
ExecStart=/usr/bin/bluealsa -S -p a2dp-source -p a2dp-sink -p hfp-ag
HFPEOF
sudo systemctl daemon-reload
sudo systemctl restart bluealsa || warn "bluealsa restart failed; reboot to apply"
echo "    BlueALSA restarted with HFP enabled (headset mic available)"

# -----------------------------------------------------------------------------
log "3/9 Stabilising eth0 addressing"
# -----------------------------------------------------------------------------
# eth0 is usually a direct Pi-to-laptop debug cable with no DHCP server on it.
# NetworkManager's default (dhcp4: true) then fails and retries forever:
#
#   ip-config -> failed -> disconnected -> prepare -> config -> ip-config
#
# Every pass through 'disconnected' tears down the interface's addresses and
# kills any SSH session over that cable - which looks exactly like the Pi
# crashing and recovering every ~45 seconds. It is not a crash; uptime never
# resets.
#
# Setting ipv6.method=link-local is the part that fixes it: it succeeds
# immediately, so activation completes and the cycle never starts. IPv4 still
# tries DHCP and falls back to a 169.254.x.x address when no server answers.
# (ipv4.link-local=fallback on its own is NOT enough - the interface still
# tore down after ~90s in testing.)
#
# Trade-off: plugged into a real router, eth0 gets IPv4 via DHCP but no IPv6
# SLAAC. To restore it:  sudo nmcli connection modify <conn> ipv6.method auto
#
# A Pi has a single ethernet port, so the first 802-3-ethernet profile is it.
ETH_CONN="$(nmcli -t -f NAME,TYPE connection show 2>/dev/null \
            | awk -F: '$2 == "802-3-ethernet" { print $1; exit }' || true)"
if [ -n "${ETH_CONN:-}" ]; then
    if sudo nmcli connection modify "$ETH_CONN" \
            ipv6.method link-local \
            ipv4.method auto \
            ipv4.link-local fallback \
            ipv4.dhcp-timeout 15 2>/dev/null; then
        echo "    '$ETH_CONN' now activates without waiting on DHCP"
        sudo nmcli connection up "$ETH_CONN" > /dev/null 2>&1 || true
    else
        warn "could not modify eth0 connection '$ETH_CONN'; leaving as-is"
    fi
else
    echo "    no NetworkManager ethernet profile found; nothing to do"
fi

# -----------------------------------------------------------------------------
log "4/9 Creating project directory and virtual environment"
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
log "5/9 Installing Python packages"
# -----------------------------------------------------------------------------
pip install --upgrade pip
pip install \
    "ultralytics>=8.4" \
    ncnn \
    opencv-python \
    pytesseract \
    "piper-tts>=1.4" \
    vosk

# -----------------------------------------------------------------------------
log "6/9 YOLOv8n -> NCNN export"
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
log "7/9 Downloading voice and feature models"
# -----------------------------------------------------------------------------
if [ ! -f "$FYP_DIR/voices/en_US-lessac-medium.onnx" ]; then
    cd "$FYP_DIR/voices"
    python -m piper.download_voices en_US-lessac-medium
else
    warn "voice model already present, skipping download"
fi

# Models for the emotion and voice-command features. Each is optional: the
# matching feature reports itself unavailable rather than crashing when a file
# is missing, so a failed download here never breaks the wearable.
MODEL_DIR="$FYP_DIR/models"
mkdir -p "$MODEL_DIR"
cd "$MODEL_DIR"

fetch() {   # fetch <file> <url> <description>
    if [ -e "$1" ]; then
        warn "$1 already present, skipping"
        return 0
    fi
    echo "    downloading $3 ..."
    # Download to .part and rename, so an interrupted transfer is never
    # mistaken for a complete model on the next run.
    if curl -sSL -m 600 -o "$1.part" "$2"; then
        mv "$1.part" "$1"
    else
        rm -f "$1.part"
        warn "could not download $3; that feature will stay disabled"
    fi
}

fetch face_detection_yunet_2023mar.onnx \
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx" \
    "YuNet face detector (230 KB)"

fetch emotion-ferplus-8.onnx \
    "https://github.com/onnx/models/raw/main/validated/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx" \
    "FER+ emotion classifier (34 MB)"

if [ ! -d vosk-model-small-en-us-0.15 ]; then
    fetch vosk-small.zip \
        "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip" \
        "Vosk offline speech model (40 MB)"
    [ -f vosk-small.zip ] && { unzip -q vosk-small.zip && rm -f vosk-small.zip; }
else
    warn "vosk speech model already present, skipping"
fi
cd "$FYP_DIR"

# -----------------------------------------------------------------------------
log "8/9 Installing main.py"
# -----------------------------------------------------------------------------
if [ -f "$REPO_DIR/main.py" ]; then
    cp "$REPO_DIR/main.py" "$FYP_DIR/main.py"
else
    warn "main.py not found next to install.sh; copy it to $FYP_DIR manually"
fi

# Feature modules: distance, finder, emotion, currency, voice, reporting.
if [ -d "$REPO_DIR/fypvision" ]; then
    rm -rf "$FYP_DIR/fypvision"
    cp -r "$REPO_DIR/fypvision" "$FYP_DIR/fypvision"
    echo "    installed fypvision/ feature modules"
else
    warn "fypvision/ not found next to install.sh; features will not import"
fi

# -----------------------------------------------------------------------------
log "9/9 Installing systemd service (left disabled)"
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
