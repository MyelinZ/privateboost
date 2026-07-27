#!/usr/bin/env bash
#
# Build the pbr-server:deploy image, load it onto the Hetzner VM, ship the
# compose file, the rendered configs, and the TLS/FCM secrets to /opt/pbr, and
# bring the four-service stack up. Run from anywhere; paths resolve against the
# repo root.
#
# The committed aggregator config carries a literal <VM_IP> placeholder (it
# advertises the shareholders' public endpoints). This substitutes the real IP
# into rendered copies under a temp dir and ships those; the tracked files are
# never touched. The IP comes from `terraform output` unless given as $1.
#
# REMOTE_DIR is a fixed local constant; interpolating it into the remote ssh
# command strings (expanding client-side) is intended, so SC2029 is silenced.
# shellcheck disable=SC2029
set -euo pipefail

readonly IMAGE="pbr-server:deploy"
readonly REMOTE_DIR="/opt/pbr"
readonly SSH_KEY="$HOME/.ssh/id_ed25519"

usage() {
  cat >&2 <<'EOF'
usage: ship.sh [VM_IP]

Builds pbr-server:deploy, loads it onto the VM, ships docker-compose.yml, the
rendered configs, and secrets/ to /opt/pbr, then runs `docker compose up -d`.
VM_IP defaults to `terraform -chdir=infra output -raw server_ip`.
EOF
}

staging=""
cleanup() {
  if [ -n "$staging" ]; then
    rm -rf "$staging"
  fi
}
trap cleanup EXIT

main() {
  local script_dir vm_ip remote
  local -a ssh_opts missing

  case "${1:-}" in -h | --help) usage; exit 0 ;; esac

  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  cd "$script_dir/../.."

  # VM IP: explicit argument wins; otherwise read it from terraform state.
  if [ -n "${1:-}" ]; then
    vm_ip="$1"
  else
    command -v terraform >/dev/null 2>&1 || {
      echo "ship.sh: terraform not on PATH; pass the VM IP as an argument or run inside the devenv shell" >&2
      exit 1
    }
    vm_ip="$(terraform -chdir="$script_dir/infra" output -raw server_ip 2>/dev/null || true)"
    [ -n "$vm_ip" ] || {
      echo "ship.sh: could not read server_ip from terraform; apply the infra first or pass the IP as an argument" >&2
      exit 1
    }
  fi

  # server.key, fcm-service-account.json, firestore-sa.json, and admin-token
  # are git-ignored and generated per VM; without them the stack cannot
  # terminate TLS, send round-open pushes, write per-tree eval metrics (the
  # committed aggregator.toml sets [eval] unconditionally, so a missing key
  # would otherwise only surface as a remote aggregator startup failure), or
  # authorize CreateSession (so the phones would have no dataset session to
  # join).
  missing=()
  for f in server.crt server.key fcm-service-account.json firestore-sa.json admin-token; do
    [ -f "$script_dir/secrets/$f" ] || missing+=("secrets/$f")
  done
  if [ "${#missing[@]}" -gt 0 ]; then
    echo "ship.sh: missing ${missing[*]}; run gen-certs.sh $vm_ip, place fcm-service-account.json and firestore-sa.json, and 'openssl rand -hex 32 > deploy/hetzner/secrets/admin-token' (see README)" >&2
    exit 1
  fi

  echo "==> building $IMAGE"
  docker build -t "$IMAGE" -f deploy/hetzner/Dockerfile .

  # Stage what ships: the compose file, the rendered configs, and the secrets.
  # cp -p preserves server.key's 0600 mode; tar -xp restores it on the VM.
  staging="$(mktemp -d)"
  mkdir -p "$staging/configs" "$staging/secrets"
  cp "$script_dir/docker-compose.yml" "$staging/docker-compose.yml"
  cp "$script_dir"/configs/*.toml "$staging/configs/"
  sed -i "s/<VM_IP>/$vm_ip/g" "$staging"/configs/*.toml
  if grep -l '<VM_IP>' "$staging"/configs/*.toml >/dev/null 2>&1; then
    echo "ship.sh: <VM_IP> placeholder still present after render; aborting" >&2
    exit 1
  fi
  # admin-token is hex (openssl rand -hex), so it carries no sed-special
  # characters; substitute it the same way and confirm it rendered.
  local admin_token
  admin_token="$(tr -d '[:space:]' < "$script_dir/secrets/admin-token")"
  [ -n "$admin_token" ] || {
    echo "ship.sh: secrets/admin-token is empty; regenerate with 'openssl rand -hex 32 > deploy/hetzner/secrets/admin-token'" >&2
    exit 1
  }
  sed -i "s/<ADMIN_TOKEN>/$admin_token/g" "$staging"/configs/*.toml
  if grep -l '<ADMIN_TOKEN>' "$staging"/configs/*.toml >/dev/null 2>&1; then
    echo "ship.sh: <ADMIN_TOKEN> placeholder still present after render; aborting" >&2
    exit 1
  fi
  # aggregator.toml now embeds the plaintext admin token; restrict it to 0600
  # like server.key (tar -xp below preserves the mode onto the VM) so it does
  # not land world-readable.
  chmod 600 "$staging/configs/aggregator.toml"
  cp -p "$script_dir/secrets/server.crt" "$script_dir/secrets/server.key" \
    "$script_dir/secrets/fcm-service-account.json" \
    "$script_dir/secrets/firestore-sa.json" "$staging/secrets/"
  # ca.crt is public and harmless on the VM; having it there lets the operator
  # run openssl checks from the box itself.
  if [ -f "$script_dir/secrets/ca.crt" ]; then
    cp -p "$script_dir/secrets/ca.crt" "$staging/secrets/"
  fi

  remote="root@$vm_ip"
  ssh_opts=(-i "$SSH_KEY" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20)

  echo "==> loading image onto $vm_ip"
  docker save "$IMAGE" | gzip | ssh "${ssh_opts[@]}" "$remote" 'gunzip | docker load'

  echo "==> shipping compose, configs, and secrets to $REMOTE_DIR"
  ssh "${ssh_opts[@]}" "$remote" "mkdir -p $REMOTE_DIR"
  tar -C "$staging" -cf - . | ssh "${ssh_opts[@]}" "$remote" "tar -C $REMOTE_DIR -xpf -"

  echo "==> starting the stack"
  ssh "${ssh_opts[@]}" "$remote" "cd $REMOTE_DIR && docker compose up -d"

  echo "==> stack status"
  ssh "${ssh_opts[@]}" "$remote" "cd $REMOTE_DIR && docker compose ps"

  echo "done. Verify with: deploy/hetzner/smoke.sh $vm_ip"
}

main "$@"
