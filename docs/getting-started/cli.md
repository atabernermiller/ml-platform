# CLI: Bootstrap & Check

The `ml-platform` CLI automates AWS environment setup and validates your configuration before you write service code.

## Preflight check

Verify that credentials, buckets, tables, and permissions are all in order:

```bash
ml-platform check \
  --service-name my-bandit \
  --s3-bucket my-checkpoints \
  --dynamodb-table my-bandit-context \
  --region us-east-1
```

The check command validates:

1. **AWS credentials** — resolves via boto3's default chain and calls `sts:GetCallerIdentity`
2. **S3 bucket** — verifies the bucket exists and is accessible
3. **S3 write permissions** — puts and deletes a test key
4. **DynamoDB table** — verifies the table exists, has the `request_id` key schema, and TTL is enabled
5. **CloudWatch** — verifies `PutMetricData` permission

Exits with code 0 if all checks pass, 1 if any fail. Checks are skipped for features not configured (e.g., no `--s3-bucket` skips S3 checks).

## Bootstrap resources

Create missing AWS resources and generate the IAM policy:

```bash
# Dry run — preview without making changes
ml-platform bootstrap \
  --service-name my-bandit \
  --s3-bucket my-checkpoints \
  --region us-east-1 \
  --dry-run

# Create resources
ml-platform bootstrap \
  --service-name my-bandit \
  --s3-bucket my-checkpoints \
  --region us-east-1
```

The bootstrap command will:

1. **Create the S3 bucket** with versioning enabled (skips if it already exists)
2. **Create the DynamoDB context table** with `request_id` partition key and TTL on the `ttl` attribute (waits for ACTIVE status)
3. **Print the minimum IAM policy JSON** tailored to your configuration

## CLI reference

::: ml_platform.cli.main
    options:
      show_root_heading: false
      members:
        - main

::: ml_platform.cli.check
    options:
      members:
        - run_check

::: ml_platform.cli.bootstrap
    options:
      members:
        - run_bootstrap
