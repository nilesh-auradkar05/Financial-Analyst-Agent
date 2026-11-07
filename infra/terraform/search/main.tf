# CM-3 search stack (ADR-0014, ADR-0016): one next-generation OpenSearch Serverless vector
# collection that scales to zero when idle. Apply only with owner approval; destroy before the
# cost stack.
terraform {
  required_version = ">= 1.6"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.49" # collection_group.generation (NEXTGEN) arrived in 6.49.0
    }
  }
}

provider "aws" {
  region = "us-east-1"
  default_tags {
    tags = {
      Project   = "alpha-analyst"
      Stack     = "search"
      ManagedBy = "terraform"
    }
  }
}

variable "data_access_user" {
  type        = string
  description = "IAM user whose SigV4 requests may read and write the collection (set in the gitignored terraform.tfvars)."
}

locals {
  group_name      = "alpha-analyst-search"
  collection_name = "alpha-analyst-vectors"
}

data "aws_iam_user" "data_access" {
  user_name = var.data_access_user
}

resource "aws_opensearchserverless_collection_group" "search" {
  name             = local.group_name
  generation       = "NEXTGEN" # omitted → CLASSIC, which keeps 2 OCUs on around the clock
  standby_replicas = "ENABLED" # required by NEXTGEN

  # Minimums must be written as 0; leaving them out taints the group on apply.
  capacity_limits {
    min_indexing_capacity_in_ocu = 0
    max_indexing_capacity_in_ocu = 2
    min_search_capacity_in_ocu   = 0
    max_search_capacity_in_ocu   = 2
  }
}

resource "aws_opensearchserverless_security_policy" "encryption" {
  name = "${local.collection_name}-enc"
  type = "encryption"
  policy = jsonencode({
    Rules       = [{ ResourceType = "collection", Resource = ["collection/${local.collection_name}"] }]
    AWSOwnedKey = true
  })
}

# Public endpoint, IAM (SigV4) authorization only (ADR-0014 decision 6); a VPC endpoint is priced first.
resource "aws_opensearchserverless_security_policy" "network" {
  name = "${local.collection_name}-net"
  type = "network"
  policy = jsonencode([{
    Rules           = [{ ResourceType = "collection", Resource = ["collection/${local.collection_name}"] }]
    AllowFromPublic = true
  }])
}

# Explicit permissions, no aoss:* (CLAUDE.md §5).
resource "aws_opensearchserverless_access_policy" "data" {
  name = "${local.collection_name}-data"
  type = "data"
  policy = jsonencode([{
    Rules = [
      {
        ResourceType = "collection"
        Resource     = ["collection/${local.collection_name}"]
        Permission = [
          "aoss:CreateCollectionItems",
          "aoss:DeleteCollectionItems",
          "aoss:UpdateCollectionItems",
          "aoss:DescribeCollectionItems",
        ]
      },
      {
        ResourceType = "index"
        Resource     = ["index/${local.collection_name}/*"]
        Permission = [
          "aoss:CreateIndex",
          "aoss:DeleteIndex",
          "aoss:UpdateIndex",
          "aoss:DescribeIndex",
          "aoss:ReadDocument",
          "aoss:WriteDocument",
        ]
      },
    ]
    Principal = [data.aws_iam_user.data_access.arn]
  }])
}

resource "aws_opensearchserverless_collection" "vectors" {
  name                  = local.collection_name
  type                  = "VECTORSEARCH"
  collection_group_name = aws_opensearchserverless_collection_group.search.name # sets NEXTGEN

  depends_on = [
    aws_opensearchserverless_security_policy.encryption,
    aws_opensearchserverless_security_policy.network,
  ]
}

output "group_generation" {
  value = aws_opensearchserverless_collection_group.search.generation
}

output "collection_endpoint" {
  value = aws_opensearchserverless_collection.vectors.collection_endpoint
}
