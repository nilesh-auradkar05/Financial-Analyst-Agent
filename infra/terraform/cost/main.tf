# CM-1 cost controls (ADR-0016 decision 5, ADR-0006 D16): exist before any other cloud resource.
terraform {
  required_version = ">= 1.6"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.0"
    }
  }
}

provider "aws" {
  region = "us-east-1"
  default_tags {
    tags = {
      Project   = "alpha-analyst"
      Stack     = "cost"
      ManagedBy = "terraform"
    }
  }
}

variable "alert_email" {
  type        = string
  description = "Recipient of budget alerts and kill-switch notices (set in the gitignored terraform.tfvars)."
}

variable "kill_switch_users" {
  type        = list(string)
  description = "IAM users that make model and search calls; the $100 action attaches the deny policy to them."
}

variable "kill_switch_roles" {
  type        = list(string)
  description = "IAM roles that make model and search calls; the $100 action attaches the deny policy to them."
}

data "aws_partition" "current" {}

data "aws_caller_identity" "current" {}

# Account-wide on purpose: Bedrock and OpenSearch spend is caught even if a resource misses its tags.
resource "aws_budgets_budget" "monthly" {
  name         = "alpha-analyst-monthly"
  budget_type  = "COST"
  limit_amount = "100"
  limit_unit   = "USD"
  time_unit    = "MONTHLY"

  dynamic "notification" {
    for_each = [10, 25, 100]
    content {
      comparison_operator        = "GREATER_THAN"
      threshold                  = notification.value
      threshold_type             = "ABSOLUTE_VALUE"
      notification_type          = "ACTUAL"
      subscriber_email_addresses = [var.alert_email]
    }
  }

  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 100
    threshold_type             = "ABSOLUTE_VALUE"
    notification_type          = "FORECASTED"
    subscriber_email_addresses = [var.alert_email]
  }
}

# Kill switch: stops new model and search spend; deletes stay allowed so teardown still works.
data "aws_iam_policy_document" "kill_switch" {
  statement {
    effect = "Deny"
    actions = [
      "bedrock:InvokeModel",
      "bedrock:InvokeModelWithResponseStream",
      "bedrock:Rerank",
      "bedrock-mantle:CreateInference",
      "aoss:APIAccessAll",
      "aoss:CreateCollection",
      "aoss:CreateCollectionGroup",
      "aoss:UpdateCollection",
    ]
    resources = ["*"]
  }
}

resource "aws_iam_policy" "kill_switch" {
  name        = "alpha-analyst-budget-kill-switch"
  description = "Attached by the AWS Budgets action at $100 actual spend (ADR-0006 D16)."
  policy      = data.aws_iam_policy_document.kill_switch.json
}

data "aws_iam_policy_document" "budgets_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["budgets.${data.aws_partition.current.dns_suffix}"]
    }
    # Confused-deputy guard: only Budgets acting for this account's budgets may assume the role.
    condition {
      test     = "StringEquals"
      variable = "aws:SourceAccount"
      values   = [data.aws_caller_identity.current.account_id]
    }
    condition {
      test     = "ArnLike"
      variable = "aws:SourceArn"
      values   = ["arn:${data.aws_partition.current.partition}:budgets::${data.aws_caller_identity.current.account_id}:budget/*"]
    }
  }
}

resource "aws_iam_role" "budgets_action" {
  name               = "alpha-analyst-budgets-action"
  assume_role_policy = data.aws_iam_policy_document.budgets_assume.json
}

data "aws_iam_user" "kill_switch" {
  for_each  = toset(var.kill_switch_users)
  user_name = each.value
}

# Least privilege: the role may attach or detach only the kill-switch policy, only on the listed principals.
data "aws_iam_policy_document" "budgets_action" {
  statement {
    actions   = ["iam:AttachUserPolicy", "iam:DetachUserPolicy"]
    resources = [for u in data.aws_iam_user.kill_switch : u.arn]
    condition {
      test     = "ArnEquals"
      variable = "iam:PolicyARN"
      values   = [aws_iam_policy.kill_switch.arn]
    }
  }
  statement {
    actions = ["iam:AttachRolePolicy", "iam:DetachRolePolicy"]
    # Built from names, so the deployer needs no read rights on the runtime roles.
    resources = [for r in var.kill_switch_roles : "arn:${data.aws_partition.current.partition}:iam::${data.aws_caller_identity.current.account_id}:role/${r}"]
    condition {
      test     = "ArnEquals"
      variable = "iam:PolicyARN"
      values   = [aws_iam_policy.kill_switch.arn]
    }
  }
}

resource "aws_iam_role_policy" "budgets_action" {
  name   = "attach-kill-switch"
  role   = aws_iam_role.budgets_action.id
  policy = data.aws_iam_policy_document.budgets_action.json
}

resource "aws_budgets_budget_action" "kill_switch" {
  budget_name        = aws_budgets_budget.monthly.name
  action_type        = "APPLY_IAM_POLICY"
  approval_model     = "AUTOMATIC"
  notification_type  = "ACTUAL"
  execution_role_arn = aws_iam_role.budgets_action.arn

  action_threshold {
    action_threshold_type  = "ABSOLUTE_VALUE"
    action_threshold_value = 100
  }

  definition {
    iam_action_definition {
      policy_arn = aws_iam_policy.kill_switch.arn
      users      = var.kill_switch_users
      roles      = var.kill_switch_roles
    }
  }

  subscriber {
    address           = var.alert_email
    subscription_type = "EMAIL"
  }

  depends_on = [aws_iam_role_policy.budgets_action]
}
