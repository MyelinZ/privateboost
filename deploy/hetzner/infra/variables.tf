variable "server_type" {
  description = "Hetzner server type. cx23 = Intel 2 vCPU / 4 GB, cheapest x86 in-stock. ARM cax11 is a touch cheaper but frequently sold out."
  type        = string
  default     = "cx23"
}

variable "location" {
  description = "Hetzner location. x86 types are broadly available across regions."
  type        = string
  default     = "fsn1"
}

variable "image" {
  description = "OS image slug."
  type        = string
  default     = "ubuntu-24.04"
}

variable "ssh_public_key_path" {
  description = "SSH public key uploaded for root login on the server."
  type        = string
  default     = "~/.ssh/id_ed25519.pub"
}

variable "ssh_allowed_cidrs" {
  description = "CIDRs allowed to reach SSH. Defaults open (key-only auth); lock to YOUR.IP/32 to tighten."
  type        = list(string)
  default     = ["0.0.0.0/0", "::/0"]
}

variable "name_prefix" {
  description = "Prefix for created resource + server names."
  type        = string
  default     = "pbr"
}
