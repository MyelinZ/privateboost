#!/usr/bin/env bash
#
# Post-deploy smoke check for the Hetzner stack. Verifies the TLS handshake on
# each phone-facing port against the deployment CA, then confirms the aggregator
# came up and prints its recent log so the operator can read the session state.
# The VM IP comes from `terraform output` unless given as $1.
#
# REMOTE_DIR is a fixed local constant; interpolating it into the remote ssh
# command string (expanding client-side) is intended, so SC2029 is silenced.
# shellcheck disable=SC2029
set -euo pipefail

readonly PORTS=(42800 42801 42802 42803)
readonly REMOTE_DIR="/opt/pbr"
readonly SSH_KEY="$HOME/.ssh/id_ed25519"

usage() {
  cat >&2 <<'EOF'
usage: smoke.sh [VM_IP]

Checks TLS verification on ports 42800-42803 against secrets/ca.crt and tails
the aggregator container log for "aggregator up". Exits non-zero on any failure.
VM_IP defaults to `terraform -chdir=infra output -raw server_ip`.
EOF
}

main() {
  local script_dir ca vm_ip remote port out logs failed=0
  local -a ssh_opts

  case "${1:-}" in -h | --help) usage; exit 0 ;; esac

  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  ca="$script_dir/secrets/ca.crt"

  [ -f "$ca" ] || {
    echo "smoke.sh: missing $ca; run gen-certs.sh <VM_IP> first" >&2
    exit 1
  }

  if [ -n "${1:-}" ]; then
    vm_ip="$1"
  else
    command -v terraform >/dev/null 2>&1 || {
      echo "smoke.sh: terraform not on PATH; pass the VM IP as an argument or run inside the devenv shell" >&2
      exit 1
    }
    vm_ip="$(terraform -chdir="$script_dir/infra" output -raw server_ip 2>/dev/null || true)"
    [ -n "$vm_ip" ] || {
      echo "smoke.sh: could not read server_ip from terraform; apply the infra first or pass the IP as an argument" >&2
      exit 1
    }
  fi

  echo "== TLS handshake ($vm_ip) =="
  for port in "${PORTS[@]}"; do
    # s_client exits 0 on a completed handshake regardless of trust, so the
    # verdict is the reported verify code, not the exit status. -verify_ip
    # additionally pins the cert's IP SAN to this VM: a stale server.crt
    # from a previous VM IP still chains to the CA, and clients reject it.
    out="$(openssl s_client -connect "$vm_ip:$port" -verify_ip "$vm_ip" -CAfile "$ca" </dev/null 2>/dev/null || true)"
    if grep -q "Verify return code: 0" <<<"$out"; then
      echo "  PASS  :$port  TLS verified against ca.crt (IP SAN matches)"
    else
      echo "  FAIL  :$port  TLS handshake, chain, or IP-SAN verification failed"
      failed=1
    fi
  done

  echo
  echo "== aggregator =="
  remote="root@$vm_ip"
  ssh_opts=(-i "$SSH_KEY" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20)
  logs="$(ssh "${ssh_opts[@]}" "$remote" "cd $REMOTE_DIR && docker compose logs --tail=40 aggregator" 2>/dev/null || true)"

  if [ -z "$logs" ]; then
    echo "  FAIL  could not read aggregator logs over ssh"
    failed=1
  else
    if grep -q "aggregator up" <<<"$logs"; then
      echo "  PASS  aggregator up"
    else
      echo "  FAIL  'aggregator up' not found in recent logs"
      failed=1
    fi
    echo "  --- last aggregator log lines ---"
    printf '%s\n' "$logs"
  fi

  echo
  if [ "$failed" -eq 0 ]; then
    echo "smoke: PASS"
  else
    echo "smoke: FAIL"
    exit 1
  fi
}

main "$@"
