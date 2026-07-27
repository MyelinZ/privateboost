# Hetzner VM (Terraform)

Provisions one Hetzner Cloud VM to run the four-service PrivateBoost cluster
(aggregator + three shareholders) under Docker Compose with host networking.
This module is infra only: it creates a reachable VM with Docker installed and
opens the phone-facing ports. Generating TLS material and running the services
is covered by the parent `deploy/hetzner/README.md`. `terraform destroy` removes
everything and stops billing.

## One-time: Hetzner token

1. Sign up at https://www.hetzner.com/cloud and open the Cloud Console.
2. Create a project, then **Security -> API Tokens -> Generate API Token** with
   **Read & Write** scope. Copy it (shown once).
3. Export it where Terraform reads it:

   ```
   export HCLOUD_TOKEN=<your-token>
   ```

## Bring up the VM

```bash
cd deploy/hetzner/infra
terraform init          # downloads the hcloud provider (needs internet, no token)
terraform apply
```

`apply` prints `server_ip` and `ssh_command`. Log in and confirm cloud-init
finished installing Docker:

```bash
ssh root@<server_ip>
docker --version
docker compose version
```

Feed `server_ip` to `deploy/hetzner/gen-certs.sh <server_ip>` so the server
certificate's SAN pins the VM address.

## Firewall

Open inbound: `22` (SSH, from `ssh_allowed_cidrs`), ICMP, and `42800-42803`
(aggregator + three shareholders, TLS-terminated in process). The internal plane
`42811-42813` (aggregator to shareholder share fetch) is a trust boundary and
stays closed; those endpoints are reached over loopback on the VM.

## Defaults (override in `terraform.tfvars` or with `-var`)

| var                   | default                 | notes                                |
|-----------------------|-------------------------|--------------------------------------|
| `server_type`         | `cx23`                  | Intel 2 vCPU / 4 GB, cheapest x86    |
| `location`            | `fsn1`                  | any x86 region                       |
| `image`               | `ubuntu-24.04`          |                                      |
| `ssh_public_key_path` | `~/.ssh/id_ed25519.pub` | uploaded for root login              |
| `ssh_allowed_cidrs`   | open                    | lock to `YOUR.IP/32` to tighten SSH  |
| `name_prefix`         | `pbr`                   | resource + server name prefix        |

State is local (`terraform.tfstate`, gitignored).

## Tear down

```bash
terraform destroy
```
