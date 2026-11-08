# CM-3 search stack (ADR-0014, ADR-0016). The project uses the existing next-generation
# OpenSearch Serverless collection `zcech-banking-emb` (owner decision 2026-11-07). The
# collection, its group and its encryption/network policies are owner-managed and only read
# here; this stack owns the project's two data access policies (IAM user, runtime role), both
# limited to `sec_*` indexes. `destroy` removes those policies only.
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
  collection_name = "zcech-banking-emb"
  group_name      = "nextgen-sec-filings-vec-emb"          # the collection data source does not export its group
  max_ocu         = 2                                      # ADR-0014 decision 1; the budget assumes it
  index_pattern   = "index/${local.collection_name}/sec_*" # every project index is sec_filings[__<arm>]
}

data "aws_caller_identity" "current" {}

data "aws_iam_user" "data_access" {
  user_name = var.data_access_user
}

data "aws_opensearchserverless_collection" "vectors" {
  name = local.collection_name

  lifecycle {
    postcondition {
      condition     = self.type == "VECTORSEARCH"
      error_message = "Collection ${local.collection_name} must be a VECTORSEARCH collection."
    }
  }
}

# Cost guard: fail the plan if the owner-managed group drifts off NEXTGEN or above max_ocu.
data "aws_opensearchserverless_collection_group" "search" {
  name = local.group_name

  lifecycle {
    postcondition {
      condition = self.generation == "NEXTGEN" && alltrue([
        for c in self.capacity_limits : c.max_indexing_capacity_in_ocu <= local.max_ocu
        && c.max_search_capacity_in_ocu <= local.max_ocu
        && c.min_indexing_capacity_in_ocu == 0 && c.min_search_capacity_in_ocu == 0
      ])
      error_message = "Collection group must be NEXTGEN with min 0 and max <= ${local.max_ocu} OCU for indexing and search."
    }
  }
}

# Explicit permissions, no aoss:* (CLAUDE.md §5).
resource "aws_opensearchserverless_access_policy" "data" {
  name        = "alpha-analyst-agent-data"
  type        = "data"
  description = "alpha-analyst: ${var.data_access_user} reads and writes its indexes"
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
        Resource     = [local.index_pattern]
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

# Owner-made policy for the runtime role, imported 2026-11-08 (id `opensearch-access-policy/data`):
# it named `zech-banking-emb` (typo) and granted nothing. Same permissions as the owner wrote them.

resource "aws_opensearchserverless_access_policy" "runtime" {
  name = "opensearch-access-policy"
  type = "data"
  policy = jsonencode([{
    Rules = [
      {
        ResourceType = "collection"
        Resource     = ["collection/${local.collection_name}"]
        Permission = [
          "aoss:CreateCollectionItems",
          "aoss:DescribeCollectionItems",
          "aoss:UpdateCollectionItems",
        ]
      },
      {
        ResourceType = "index"
        Resource     = [local.index_pattern]
        Permission = [
          "aoss:CreateIndex",
          "aoss:DescribeIndex",
          "aoss:UpdateIndex",
          "aoss:DeleteIndex",
          "aoss:ReadDocument",
          "aoss:WriteDocument",
        ]
      },
    ]
    Principal   = ["arn:aws:iam::${data.aws_caller_identity.current.account_id}:role/alpha-analyst-runtime"]
    Description = "Runtime role: per-arm indexes and documents"
  }])
}

output "group_generation" {
  value = data.aws_opensearchserverless_collection_group.search.generation
}

output "collection_endpoint" {
  value = data.aws_opensearchserverless_collection.vectors.collection_endpoint
}
