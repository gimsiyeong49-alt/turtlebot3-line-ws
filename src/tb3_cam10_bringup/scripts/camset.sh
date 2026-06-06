#!/bin/bash

CAM_DEV=$(v4l2-ctl --list-devices | awk '
/Arducam USB Camera/ {found=1; next}
found && /\/dev\/video/ {print $1; exit}
')

if [ -z "$CAM_DEV" ]; then
  echo "[CAMSET ERROR] Arducam not found"
  v4l2-ctl --list-devices
  exit 1
fi

echo "[CAMSET] device=$CAM_DEV"
echo "[CAMSET] Applying safe daytime parameters..."

# FPS 흔들림 방지
v4l2-ctl -d "$CAM_DEV" -c exposure_dynamic_framerate=0 2>/dev/null || true

# 색 틀어짐 방지: 화이트밸런스 자동 유지
v4l2-ctl -d "$CAM_DEV" -c white_balance_automatic=1 2>/dev/null || true

# 색감은 기본 근처로 복구
v4l2-ctl -d "$CAM_DEV" -c brightness=10 2>/dev/null || true
v4l2-ctl -d "$CAM_DEV" -c contrast=32 2>/dev/null || true
v4l2-ctl -d "$CAM_DEV" -c saturation=64 2>/dev/null || true
v4l2-ctl -d "$CAM_DEV" -c gamma=100 2>/dev/null || true
v4l2-ctl -d "$CAM_DEV" -c gain=0 2>/dev/null || true
v4l2-ctl -d "$CAM_DEV" -c sharpness=2 2>/dev/null || true
v4l2-ctl -d "$CAM_DEV" -c power_line_frequency=2 2>/dev/null || true

# 일단 자동노출 유지. 수동노출은 색/밝기 틀어짐이 커서 보류.
v4l2-ctl -d "$CAM_DEV" -c auto_exposure=3 2>/dev/null || true

echo "[CAMSET] Done."
v4l2-ctl -d "$CAM_DEV" --all | grep -Ei "exposure|gain|white|brightness|contrast|saturation|gamma|sharp|frame|interval"
