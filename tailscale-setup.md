# Tailscale Setup — /data/msz

SSH remote access via Tailscale (userspace networking, chroot-compatible).

## Environment

- **Host**: NF5468-M7 (IEIT SYSTEMS), Ubuntu Jammy chroot on EL9 kernel 5.14.0
- **Constraint**: Running inside chroot — no systemctl, no kernel module loading, iptables restricted
- **Internet**: eth0 10.120.0.16/21, gateway 10.120.0.1, public IP 101.230.144.128 (NAT)
- **Tailscale mode**: `--tun=userspace-networking` (no /dev/net/tun kernel tun required, but device node created as fallback)

## Prerequisites (already done, persisted across reboot)

### 1. Tailscale installed via apt

```bash
# One-time setup (already done):
wget -O- https://tailscale.com/install.sh | sh
# Installs tailscale 1.96.4 from pkgs.tailscale.com
```

### 2. TUN device node

The kernel supports TUN but `/dev/net/tun` may not exist in chroot. Create if missing:

```bash
mkdir -p /dev/net
mknod /dev/net/tun c 10 200 2>/dev/null || true
```

## Startup on reboot

Run these steps in order after every reboot:

### Step 1: Ensure TUN device exists

```bash
ls /dev/net/tun || (mkdir -p /dev/net && mknod /dev/net/tun c 10 200)
```

### Step 2: Start tailscaled daemon (detached)

```bash
# Kill any stale instance first
pkill -9 tailscaled 2>/dev/null || true
sleep 1

# Start tailscaled detached (uses userspace WireGuard, no kernel tun needed)
setsid /usr/sbin/tailscaled \
  --state=/var/lib/tailscale/tailscaled.state \
  --socket=/run/tailscale/tailscaled.sock \
  --port=41641 \
  --tun=userspace-networking \
  </dev/null >/tmp/tailscaled.log 2>&1 &
disown

sleep 3
pgrep -a tailscaled  # Verify it's running
```

### Step 3: Bring up Tailscale with SSH

```bash
tailscale up --ssh
```

This reuses the persistent state at `/var/lib/tailscale/tailscaled.state`, so **no re-authentication needed** after reboot.

### Step 4: Verify

```bash
tailscale status         # Should show connected, not "Logged out"
tailscale ip -4          # Show this machine's Tailscale IP
```

## Key flags explained

| Flag | Purpose |
|------|---------|
| `--state=/var/lib/tailscale/tailscaled.state` | Persists auth/node identity across reboots |
| `--socket=/run/tailscale/tailscaled.sock` | CLI <-> daemon communication |
| `--port=41641` | UDP port for WireGuard mesh (NAT traversal) |
| `--tun=userspace-networking` | Use Go userspace WireGuard — no kernel TUN needed, works in chroot/containers |

## Troubleshooting

### tailscaled won't start

```bash
# Check log
tail -50 /tmp/tailscaled.log

# Common issues:
# - "Permission denied" on iptables: normal in chroot, non-fatal with userspace-networking
# - "no such file: /dev/net/tun": run Step 1 to create device node
# - Stale socket: rm -f /run/tailscale/tailscaled.sock && restart
```

### Logged out after reboot

If state file was lost, re-run `tailscale up --ssh` — it will print an auth URL. Visit it in a browser to re-authenticate.

```bash
tailscale up --ssh 2>&1 | grep -o 'https://login.tailscale.com/a/[a-zA-Z0-9]*'
```

### Check if other devices are on the tailnet

```bash
tailscale status
```

## Connecting from other devices

1. Install Tailscale on the other device: https://tailscale.com/download
2. Log in with the same account
3. Find this server's Tailscale IP:

```bash
# On this server:
tailscale ip -4
# Example output: 100.x.y.z

# From other device:
ssh root@<tailscale-ip>
```

SSH runs on standard port 22. No port forwarding, no firewall changes needed.

## State files

| Path | Content |
|------|---------|
| `/var/lib/tailscale/tailscaled.state` | Node identity, keys, network config |
| `/var/lib/tailscale/files/` | Taildrop received files |
| `/var/lib/tailscale/derpmap.cached.json` | DERP relay map (discovery servers) |
| `/run/tailscale/tailscaled.sock` | Unix socket for CLI control |

These are all in the chroot filesystem and survive reboots.
