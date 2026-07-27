#!/usr/bin/env bash
#
# Generate the deployment's TLS material: a self-signed CA and one server
# certificate whose IP SAN pins the VM address. All four services (aggregator
# plus three shareholders) present this one server certificate; rustls checks
# the IP SAN, so the address is carried as an IP SAN rather than a DNS name.
set -euo pipefail

readonly VALIDITY_DAYS=825

tmpdir=""
cleanup() {
  if [ -n "$tmpdir" ]; then
    rm -rf "$tmpdir"
  fi
}
trap cleanup EXIT

usage() {
  cat >&2 <<'EOF'
usage: gen-certs.sh <VM_IP> [--out <dir>]

Writes ca.crt, ca.key, server.crt, server.key to <dir> (default: the
secrets/ directory beside this script). Re-running overwrites in place.
EOF
}

main() {
  local script_dir vm_ip out_dir csr ext
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  vm_ip=""
  out_dir="$script_dir/secrets"

  while [ "$#" -gt 0 ]; do
    case "$1" in
      --out) out_dir="${2:?--out needs a directory}"; shift 2 ;;
      --out=*) out_dir="${1#--out=}"; shift ;;
      -h | --help) usage; exit 0 ;;
      -*) echo "gen-certs.sh: unknown option: $1" >&2; usage; exit 1 ;;
      *)
        if [ -n "$vm_ip" ]; then
          echo "gen-certs.sh: unexpected argument: $1" >&2
          usage
          exit 1
        fi
        vm_ip="$1"
        shift
        ;;
    esac
  done

  if [ -z "$vm_ip" ]; then
    echo "gen-certs.sh: VM_IP is required" >&2
    usage
    exit 1
  fi

  mkdir -p "$out_dir"
  tmpdir="$(mktemp -d)"
  csr="$tmpdir/server.csr"
  ext="$tmpdir/server.ext"

  # Self-signed CA. It signs the one leaf below and nothing else, so pathlen:0
  # forbids any intermediate CA under it.
  openssl genpkey -quiet -algorithm RSA -pkeyopt rsa_keygen_bits:2048 \
    -out "$out_dir/ca.key"
  openssl req -x509 -key "$out_dir/ca.key" -out "$out_dir/ca.crt" \
    -days "$VALIDITY_DAYS" -subj "/CN=PrivateBoost Deployment CA" \
    -addext "basicConstraints=critical,CA:TRUE,pathlen:0" \
    -addext "keyUsage=critical,keyCertSign,cRLSign"

  # Leaf key + CSR. The subject CN is cosmetic; rustls authenticates on the IP
  # SAN added at signing time, not the subject.
  openssl genpkey -quiet -algorithm RSA -pkeyopt rsa_keygen_bits:2048 \
    -out "$out_dir/server.key"
  openssl req -new -key "$out_dir/server.key" -out "$csr" -subj "/CN=$vm_ip"

  cat >"$ext" <<EOF
subjectAltName = IP:$vm_ip
basicConstraints = critical,CA:FALSE
keyUsage = critical,digitalSignature,keyEncipherment
extendedKeyUsage = serverAuth
EOF

  # A random 128-bit serial keeps each leaf distinct without a CA serial file.
  openssl x509 -req -in "$csr" \
    -CA "$out_dir/ca.crt" -CAkey "$out_dir/ca.key" \
    -set_serial "0x$(openssl rand -hex 16)" \
    -days "$VALIDITY_DAYS" -out "$out_dir/server.crt" \
    -extfile "$ext"

  chmod 600 "$out_dir/ca.key" "$out_dir/server.key"
  chmod 644 "$out_dir/ca.crt" "$out_dir/server.crt"

  # The leaf must chain to the CA and actually carry the IP SAN rustls needs.
  openssl verify -CAfile "$out_dir/ca.crt" "$out_dir/server.crt" >/dev/null
  if ! openssl x509 -in "$out_dir/server.crt" -noout -text \
    | grep -q "IP Address:$vm_ip"; then
    echo "gen-certs.sh: server.crt is missing the IP:$vm_ip SAN" >&2
    exit 1
  fi

  echo "gen-certs.sh: wrote CA + IP:$vm_ip server cert to $out_dir"
}

main "$@"
