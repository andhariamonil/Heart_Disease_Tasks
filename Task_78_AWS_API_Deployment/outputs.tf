output "api_base_url" {
  value       = "https://${var.api_domain}"
  description = "HTTPS API base URL after DNS points api_domain at the ALB."
}

output "load_balancer_dns_name" {
  value       = aws_lb.api.dns_name
  description = "ALB DNS name for the DNS alias record."
}

output "ecr_repository_url" {
  value       = aws_ecr_repository.api.repository_url
  description = "Push the container image to this ECR repository."
}

output "ecs_cluster_name" {
  value = aws_ecs_cluster.api.name
}