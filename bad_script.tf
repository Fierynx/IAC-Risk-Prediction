# BAD TERRAFORM SCRIPT FOR TESTING CI/CD PIPELINE REJECTION
# This script is intentionally designed with anti-patterns and high complexity to trigger the ML risk predictor.

variable "aws_region" {
  default = "us-east-1"
}

# ANTI-PATTERN: Hardcoded secret
variable "db_password" {
  description = "Database password"
  default     = "SuperSecretPassword123!"
}

# ANTI-PATTERN: Hardcoded API key
variable "api_key" {
  default = "AKIAIOSFODNN7EXAMPLE"
}

resource "aws_vpc" "main" {
  cidr_block = "10.0.0.0/16"
}

resource "aws_internet_gateway" "gw" {
  vpc_id = aws_vpc.main.id
}

# ANTI-PATTERN: Open security group to the world
resource "aws_security_group" "allow_all" {
  name        = "allow_all"
  description = "Allow all inbound traffic"
  vpc_id      = aws_vpc.main.id

  ingress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"] # Anti-pattern
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# Resource 1
resource "aws_instance" "web_server_1" {
  ami           = "ami-0c55b159cbfafe1f0"
  instance_type = "t2.micro"
  vpc_security_group_ids = [aws_security_group.allow_all.id]
  depends_on = [aws_internet_gateway.gw]
}

# Resource 2
resource "aws_instance" "web_server_2" {
  ami           = "ami-0c55b159cbfafe1f0"
  instance_type = "t2.micro"
  vpc_security_group_ids = [aws_security_group.allow_all.id]
  depends_on = [aws_internet_gateway.gw]
}

# Resource 3
resource "aws_db_instance" "default" {
  allocated_storage    = 10
  engine               = "mysql"
  engine_version       = "5.7"
  instance_class       = "db.t3.micro"
  name                 = "mydb"
  username             = "admin"
  password             = var.db_password
  skip_final_snapshot  = true
  vpc_security_group_ids = [aws_security_group.allow_all.id]
}

# Deep nesting to increase complexity
resource "aws_iam_role" "test_role" {
  name = "test_role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
      },
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          AWS = "*"
        }
      }
    ]
  })
}

# Lots of lines to increase LOC metric
# 1
# 2
# 3
# 4
# 5
# 6
# 7
# 8
# 9
# 10
# 11
# 12
# 13
# 14
# 15
# 16
# 17
# 18
# 19
# 20
# 21
# 22
# 23
# 24
# 25
# 26
# 27
# 28
# 29
# 30
# 31
# 32
# 33
# 34
# 35
# 36
# 37
# 38
# 39
# 40
# 41
# 42
# 43
# 44
# 45
# 46
# 47
# 48
# 49
# 50
