# GOOD TERRAFORM SCRIPT FOR TESTING CI/CD PIPELINE PASSING
# This script is modular, clean, has no hardcoded secrets, and is small in scope.

variable "aws_region" {
  description = "The AWS region to deploy to"
  type        = string
  default     = "us-east-1"
}

variable "bucket_name" {
  description = "The name of the S3 bucket"
  type        = string
}

# A simple, secure S3 bucket resource
resource "aws_s3_bucket" "secure_bucket" {
  bucket = var.bucket_name
}

resource "aws_s3_bucket_public_access_block" "secure_bucket_access" {
  bucket = aws_s3_bucket.secure_bucket.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_server_side_encryption_configuration" "secure_bucket_encryption" {
  bucket = aws_s3_bucket.secure_bucket.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}
