#!/usr/bin/env bash
set -euo pipefail

DNS1="${DNS1:-223.5.5.5}"
DNS2="${DNS2:-119.29.29.29}"
RESOLV_CONF="/etc/resolv.conf"
BACKUP_DIR="/data/msz/network-backups"
STAMP="$(date +%Y%m%d_%H%M%S)"

mkdir -p "$BACKUP_DIR"

if [ -f "$RESOLV_CONF" ]; then
  cp "$RESOLV_CONF" "$BACKUP_DIR/resolv.conf.$STAMP.bak" || true
fi

cat > "$RESOLV_CONF" <<EOF
nameserver $DNS1
nameserver $DNS2
options timeout:2 attempts:2
EOF

export RES_OPTIONS='timeout:2 attempts:2'
export no_proxy="${no_proxy:-localhost,127.0.0.1,::1,.svc,.svc.cluster.local,10.0.0.0/8,100.64.0.0/10}"
export NO_PROXY="${NO_PROXY:-$no_proxy}"

echo "[setup] wrote $RESOLV_CONF"
cat "$RESOLV_CONF"

python3 - <<'PY'
import socket, ssl

hosts = ["www.baidu.com", "developer.metax-tech.com"]
for host in hosts:
    try:
        addr = socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)[0][4]
        raw = socket.create_connection(addr, timeout=8)
        s = ssl.create_default_context().wrap_socket(raw, server_hostname=host)
        print(f"[setup] OK {host} {addr} {s.version()}")
        s.close()
    except Exception as e:
        print(f"[setup] FAIL {host}: {type(e).__name__}: {e}")
PY

cat <<'EOF'

[setup] done.

If you need Hugging Face or other blocked sites, set proxy before running download commands, for example:

  export http_proxy=http://A.B.C.D:PORT
  export https_proxy=http://A.B.C.D:PORT
  export HTTP_PROXY=$http_proxy
  export HTTPS_PROXY=$https_proxy

EOF
