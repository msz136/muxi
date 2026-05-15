#!/bin/bash
# /data/msz/start-tailscale.sh
# Run after reboot to restore Tailscale SSH access.
# Safe to run multiple times — idempotent.

set -e

echo "[tailscale] Ensuring TUN device node..."
mkdir -p /dev/net
mknod /dev/net/tun c 10 200 2>/dev/null || true

echo "[tailscale] Stopping stale tailscaled..."
pkill -9 tailscaled 2>/dev/null || true
sleep 1

echo "[tailscale] Starting tailscaled (userspace-networking)..."
mkdir -p /run/tailscale /var/lib/tailscale
setsid /usr/sbin/tailscaled \
  --state=/var/lib/tailscale/tailscaled.state \
  --socket=/run/tailscale/tailscaled.sock \
  --port=41641 \
  --tun=userspace-networking \
  </dev/null >/tmp/tailscaled.log 2>&1 &
disown

sleep 3

if ! pgrep tailscaled >/dev/null; then
    echo "[tailscale] ERROR: tailscaled failed to start. Check /tmp/tailscaled.log"
    exit 1
fi

echo "[tailscale] Bringing up (reuses existing auth)..."
tailscale up --ssh 2>&1 &
sleep 3

if tailscale status 2>&1 | grep -q "Logged out"; then
    echo "[tailscale] Needs re-auth. Visit the URL below:"
    grep -o 'https://login.tailscale.com/a/[a-zA-Z0-9]*' /tmp/tailscaled.log | tail -1
    echo "[tailscale] Then run: tailscale up --ssh"
else
    echo "[tailscale] Connected. Tailscale IP: $(tailscale ip -4 2>/dev/null || echo 'checking...')"
    echo "[tailscale] Done."
fi
