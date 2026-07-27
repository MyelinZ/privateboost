terraform {
  required_version = ">= 1.5"
  required_providers {
    hcloud = {
      source  = "hetznercloud/hcloud"
      version = "~> 1.48"
    }
  }
}

# Token is read from the HCLOUD_TOKEN environment variable — never commit it.
provider "hcloud" {}

locals {
  # Hetzner stores public keys without the trailing comment and uses that
  # comment as the key's name. Strip it to the bare "type material" so the
  # resource matches Hetzner's stored form and does not force-replace on every
  # plan.
  ssh_public_key = join(" ", slice(split(" ", trimspace(file(pathexpand(var.ssh_public_key_path)))), 0, 2))
}

resource "hcloud_ssh_key" "default" {
  name       = "${var.name_prefix}-key"
  public_key = local.ssh_public_key
}

# SSH, ICMP, and the four phone-facing service ports. 42800 is the aggregator;
# 42801-42803 are the three shareholders. Every service terminates TLS in
# process, so the ports are open to the world.
#
# The internal plane (42811-42813, aggregator to shareholder share fetch) is
# deliberately absent. It is a trust boundary: those endpoints hand out Shamir
# shares and must never be reachable off-host. All four services run on this one
# VM under host networking, so the aggregator reaches them over loopback and no
# firewall rule is needed.
resource "hcloud_firewall" "cluster" {
  name = "${var.name_prefix}-fw"

  rule {
    direction  = "in"
    protocol   = "tcp"
    port       = "22"
    source_ips = var.ssh_allowed_cidrs
  }

  rule {
    direction  = "in"
    protocol   = "icmp"
    source_ips = ["0.0.0.0/0", "::/0"]
  }

  rule {
    direction  = "in"
    protocol   = "tcp"
    port       = "42800-42803"
    source_ips = ["0.0.0.0/0", "::/0"]
  }
}

resource "hcloud_server" "node" {
  name         = "${var.name_prefix}-node"
  server_type  = var.server_type
  image        = var.image
  location     = var.location
  ssh_keys     = [hcloud_ssh_key.default.id]
  firewall_ids = [hcloud_firewall.cluster.id]
  # Installs Docker so the VM can run the compose cluster. Editing this forces a
  # server rebuild, so it lives here from the start.
  user_data = file("${path.module}/cloud-init.yaml")

  labels = {
    project = var.name_prefix
  }
}
