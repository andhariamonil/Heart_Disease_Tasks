variable "aws_region" {
  type        = string
  description = "AWS region for the API service."
  default     = "us-east-1"
}

variable "vpc_id" {
  type        = string
  description = "Existing VPC ID."
}

variable "public_subnet_ids" {
  type        = list(string)
  description = "At least two public subnets in distinct availability zones for the ALB."
}

variable "private_subnet_ids" {
  type        = list(string)
  description = "Private subnets with outbound access (for example via NAT) for ECS tasks."
}

variable "certificate_arn" {
  type        = string
  description = "ACM certificate ARN matching api_domain."
}

variable "api_domain" {
  type        = string
  description = "Public DNS name covered by certificate_arn."
}

variable "api_token_secret_arn" {
  type        = string
  description = "Secrets Manager ARN containing a random API bearer token of at least 32 characters."
  sensitive   = true
}

variable "image_uri" {
  type        = string
  description = "Pushed ECR image URI including an immutable image tag."
}

variable "desired_count" {
  type        = number
  default     = 2
  description = "Initial ECS task count."
}

variable "min_capacity" {
  type    = number
  default = 2
}

variable "max_capacity" {
  type    = number
  default = 10
}