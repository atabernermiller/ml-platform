"""Create AWS resources required by an ml-platform service.

Creates the S3 checkpoint bucket and DynamoDB context table if they
don't already exist, and prints the minimum IAM policy document that
should be attached to the service's execution role.
"""

from __future__ import annotations

import json

from ml_platform.config import ServiceConfig

_PASS = "\033[32m✓\033[0m"
_FAIL = "\033[31m✗\033[0m"
_SKIP = "\033[33m–\033[0m"
_INFO = "\033[36mℹ\033[0m"
_BOLD = "\033[1m"
_RESET = "\033[0m"


def _print(symbol: str, message: str) -> None:
    print(f"  {symbol}  {message}")


def _ensure_s3_bucket(bucket: str, region: str, dry_run: bool) -> bool:
    """Create the S3 bucket if it doesn't exist.

    Returns:
        ``True`` if the bucket exists (or was created), ``False`` on failure.
    """
    import boto3
    from botocore.exceptions import ClientError

    s3 = boto3.client("s3", region_name=region)

    try:
        s3.head_bucket(Bucket=bucket)
        _print(_SKIP, f"S3 bucket s3://{bucket} already exists")
        return True
    except ClientError as exc:
        code = exc.response["Error"]["Code"]
        if code == "403":
            _print(_FAIL, f"Bucket s3://{bucket} exists but access denied")
            return False
        if code != "404":
            _print(_FAIL, f"Unexpected S3 error: {exc}")
            return False

    if dry_run:
        _print(_INFO, f"Would create S3 bucket s3://{bucket}")
        return True

    try:
        create_kwargs: dict = {"Bucket": bucket}
        if region != "us-east-1":
            create_kwargs["CreateBucketConfiguration"] = {
                "LocationConstraint": region
            }
        s3.create_bucket(**create_kwargs)
        s3.put_bucket_versioning(
            Bucket=bucket,
            VersioningConfiguration={"Status": "Enabled"},
        )
        _print(_PASS, f"Created S3 bucket s3://{bucket} (versioning enabled)")
        return True
    except Exception as exc:
        _print(_FAIL, f"Failed to create bucket: {exc}")
        return False


def _ensure_dynamodb_table(table_name: str, region: str, dry_run: bool) -> bool:
    """Create the DynamoDB context table if it doesn't exist.

    Returns:
        ``True`` if the table exists (or was created), ``False`` on failure.
    """
    import boto3
    from botocore.exceptions import ClientError

    dynamodb = boto3.client("dynamodb", region_name=region)

    try:
        dynamodb.describe_table(TableName=table_name)
        _print(_SKIP, f"DynamoDB table {table_name} already exists")
        return True
    except ClientError as exc:
        if exc.response["Error"]["Code"] != "ResourceNotFoundException":
            _print(_FAIL, f"Unexpected DynamoDB error: {exc}")
            return False

    if dry_run:
        _print(_INFO, f"Would create DynamoDB table {table_name}")
        return True

    try:
        dynamodb.create_table(
            TableName=table_name,
            KeySchema=[
                {"AttributeName": "request_id", "KeyType": "HASH"},
            ],
            AttributeDefinitions=[
                {"AttributeName": "request_id", "AttributeType": "S"},
            ],
            BillingMode="PAY_PER_REQUEST",
        )
        waiter = dynamodb.get_waiter("table_exists")
        _print(_INFO, f"Waiting for table {table_name} to become active...")
        waiter.wait(TableName=table_name, WaiterConfig={"Delay": 2, "MaxAttempts": 30})

        dynamodb.update_time_to_live(
            TableName=table_name,
            TimeToLiveSpecification={"Enabled": True, "AttributeName": "ttl"},
        )
        _print(_PASS, f"Created DynamoDB table {table_name} (TTL on 'ttl' attribute)")
        return True
    except Exception as exc:
        _print(_FAIL, f"Failed to create table: {exc}")
        return False


def _generate_iam_policy(config: ServiceConfig) -> dict:
    """Build the minimum IAM policy for the given configuration.

    Args:
        config: Service configuration determining which statements to include.

    Returns:
        IAM policy document as a dictionary.
    """
    statements = []

    if config.s3_checkpoint_bucket:
        statements.append({
            "Sid": "S3Checkpoints",
            "Effect": "Allow",
            "Action": [
                "s3:PutObject",
                "s3:GetObject",
                "s3:DeleteObject",
                "s3:ListBucket",
            ],
            "Resource": [
                f"arn:aws:s3:::{config.s3_checkpoint_bucket}",
                f"arn:aws:s3:::{config.s3_checkpoint_bucket}/*",
            ],
        })

    statements.append({
        "Sid": "CloudWatchMetrics",
        "Effect": "Allow",
        "Action": "cloudwatch:PutMetricData",
        "Resource": "*",
        "Condition": {
            "StringEquals": {"cloudwatch:namespace": "MLPlatform"}
        },
    })

    if config.state_table_name:
        statements.append({
            "Sid": "DynamoDBContextStore",
            "Effect": "Allow",
            "Action": [
                "dynamodb:PutItem",
                "dynamodb:DeleteItem",
                "dynamodb:DescribeTable",
            ],
            "Resource": (
                f"arn:aws:dynamodb:{config.aws_region}:*"
                f":table/{config.state_table_name}"
            ),
        })

    return {"Version": "2012-10-17", "Statement": statements}


def run_bootstrap(config: ServiceConfig, *, dry_run: bool = False) -> bool:
    """Create AWS resources and print the IAM policy.

    Args:
        config: Service configuration driving resource creation.
        dry_run: If ``True``, only print what would be created.

    Returns:
        ``True`` if all operations succeeded.
    """
    mode = "DRY RUN" if dry_run else "bootstrap"
    print(f"\n{_BOLD}ml-platform {mode}{_RESET}")
    print(f"  Service: {config.service_name}")
    print(f"  Region:  {config.aws_region}\n")

    # Verify credentials first
    try:
        import boto3

        sts = boto3.client("sts", region_name=config.aws_region)
        identity = sts.get_caller_identity()
        _print(_PASS, f"Authenticated as {identity['Arn']}")
    except Exception as exc:
        _print(_FAIL, f"Credential check failed: {exc}")
        print(f"\n  Cannot bootstrap without valid AWS credentials.\n")
        return False

    results: list[bool] = []

    # S3 bucket
    print(f"\n{_BOLD}  S3 Checkpoint Bucket{_RESET}")
    if config.s3_checkpoint_bucket:
        ok = _ensure_s3_bucket(
            config.s3_checkpoint_bucket, config.aws_region, dry_run
        )
        results.append(ok)
    else:
        _print(_SKIP, "No S3 bucket configured — skipping")

    # DynamoDB table
    print(f"\n{_BOLD}  DynamoDB Context Table{_RESET}")
    if config.state_table_name:
        ok = _ensure_dynamodb_table(
            config.state_table_name, config.aws_region, dry_run
        )
        results.append(ok)
    else:
        _print(_SKIP, "No DynamoDB table configured — skipping")

    # IAM policy
    print(f"\n{_BOLD}  IAM Policy{_RESET}")
    policy = _generate_iam_policy(config)
    policy_json = json.dumps(policy, indent=2)
    _print(
        _INFO,
        "Attach the following policy to your service's execution role:\n",
    )
    print(policy_json)

    # Summary
    all_ok = all(results) if results else True
    symbol = _PASS if all_ok else _FAIL
    print(f"\n{symbol}  Bootstrap {'would complete' if dry_run else 'complete'}.")

    if not dry_run:
        print(f"\n  Next steps:")
        print(f"    1. Attach the IAM policy above to your service role")
        print(f"    2. Run 'ml-platform check --service-name {config.service_name}"
              f" --s3-bucket {config.s3_checkpoint_bucket}"
              f" --region {config.aws_region}' to verify")
        print()

    return all_ok
