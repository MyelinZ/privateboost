output "server_ip" {
  description = "Public IPv4 of the VM. Pass it to gen-certs.sh for the server certificate SAN."
  value       = hcloud_server.node.ipv4_address
}

output "ssh_command" {
  description = "Ready-to-paste SSH command (Hetzner Ubuntu images log in as root)."
  value       = "ssh root@${hcloud_server.node.ipv4_address}"
}
