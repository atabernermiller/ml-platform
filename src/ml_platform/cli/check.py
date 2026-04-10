"""Preflight validation for AWS credentials and resources.

Runs a series of read-only checks and prints a pass/fail report.
Exit code 0 if all checks pass, 1 if any fail.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any

from ml_platform.config import ServiceConfig

if TYPE_CHECKING:
    from mypy_boto3_cloudwatch.client import CloudWatchClient
    from mypy_boto3_dynamodb.client import DynamoDBClient
    from mypy_boto3_s3.client import S3Client
    from mypy_boto3_sts.client import STSClient

_PASS = "\033[32m✓\033[0m"
_FAIL = "\033[31m✗\033[0m"
_SKIP = "\033[33m–\033[0m"
_BOLD = "\033[1m"
_RESET = "\033[0m"


def _print(symbol: str, message: str) -> None:
    print(f"  {symbol}  {message}")


def _check_credentials(region: str) -> tuple[bool, str]:
    """Verify that boto3 can resolve credentials and call STS."""
    try:
        import boto3

        sts: STSClient = boto3.client("sts", region_name=region)
        identity = sts.get_caller_identity()
        account = identity["Account"]
        arn = identity["Arn"]
        return True, f"Authenticated as {arn} (account {account})"
    except Exception as exc:
        return False, f"Credential resolution failed: {exc}"


def _check_s3_bucket(bucket: str, prefix: str, region: str) -> tuple[bool, str]:
    """Verify the S3 bucket exists and is listable."""
    try:
        import boto3

        s3: S3Client = boto3.client("s3", region_name=region)
        s3.head_bucket(Bucket=bucket)

        response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=1)
        count_hint = "has existing objects" if response.get("Contents") else "empty"
        return True, f"Bucket s3://{bucket} exists ({count_hint} under {prefix})"
    except s3.exceptions.ClientError as exc:
        code = exc.response["Error"]["Code"]
        if code == "403":
            return False, f"Bucket s3://{bucket} exists but access is denied (check IAM)"
        elif code == "404":
            return False, f"Bucket s3://{bucket} does not exist"
        return False, f"S3 error: {exc}"
    except Exception as exc:
        return False, f"S3 check failed: {exc}"


def _check_s3_write(bucket: str, region: str) -> tuple[bool, str]:
    """Verify PutObject permission with a dry-run upload."""
    try:
        import boto3

        s3: S3Client = boto3.client("s3", region_name=region)
        test_key = ".ml-platform-check"
        s3.put_object(Bucket=bucket, Key=test_key, Body=b"")
        s3.delete_object(Bucket=bucket, Key=test_key)
        return True, "S3 write permission verified (PutObject + DeleteObject)"
    except Exception as exc:
        return False, f"S3 write check failed: {exc}"


def _check_dynamodb_table(table_name: str, region: str) -> tuple[bool, str]:
    """Verify the DynamoDB table exists and has the expected key schema."""
    try:
        import boto3

        dynamodb: DynamoDBClient = boto3.client("dynamodb", region_name=region)
        desc = dynamodb.describe_table(TableName=table_name)
        table = desc["Table"]

        key_schema = {k["AttributeName"]: k["KeyType"] for k in table["KeySchema"]}
        if "request_id" not in key_schema:
            return False, (
                f"Table {table_name} exists but partition key is "
                f"{list(key_schema.keys())} — expected 'request_id'"
            )

        ttl_resp = dynamodb.describe_time_to_live(TableName=table_name)
        ttl_status = ttl_resp["TimeToLiveDescription"]["TimeToLiveStatus"]
        ttl_ok = ttl_status in ("ENABLED", "ENABLING")
        ttl_msg = "TTL enabled" if ttl_ok else f"TTL is {ttl_status} (should be ENABLED on 'ttl' attribute)"

        status = table["TableStatus"]
        return True, f"Table {table_name} exists (status={status}, {ttl_msg})"
    except dynamodb.exceptions.ResourceNotFoundException:
        return False, f"Table {table_name} does not exist"
    except Exception as exc:
        return False, f"DynamoDB check failed: {exc}"


def _check_cloudwatch_put(region: str) -> tuple[bool, str]:
    """Verify PutMetricData permission (only needed for emit_direct)."""
    try:
        import boto3

        cw: CloudWatchClient = boto3.client("cloudwatch", region_name=region)
        cw.put_metric_data(
            Namespace="MLPlatform",
            MetricData=[
                {
                    "MetricName": "_ml_platform_check",
                    "Value": 0,
                    "Unit": "None",
                    "Dimensions": [{"Name": "service", "Value": "_check"}],
                }
            ],
        )
        return True, "CloudWatch PutMetricData permission verified"
    except Exception as exc:
        return False, f"CloudWatch PutMetricData check failed: {exc}"


def run_check(config: ServiceConfig) -> bool:
    """Run all preflight checks and print a report.

    Args:
        config: Service configuration driving which checks to run.

    Returns:
        ``True`` if all applicable checks passed.
    """
    print(f"\n{_BOLD}ml-platform preflight check{_RESET}")
    print(f"  Service: {config.service_name}")
    print(f"  Region:  {config.aws_region}\n")

    results: list[bool] = []

    # 1. Credentials
    ok, msg = _check_credentials(config.aws_region)
    _print(_PASS if ok else _FAIL, msg)
    results.append(ok)

    if not ok:
        print(f"\n{_FAIL}  Cannot continue without valid credentials.\n")
        return False

    # 2. S3 bucket
    if config.s3_checkpoint_bucket:
        ok, msg = _check_s3_bucket(
            config.s3_checkpoint_bucket, config.s3_checkpoint_prefix, config.aws_region
        )
        _print(_PASS if ok else _FAIL, msg)
        results.append(ok)

        if ok:
            ok_w, msg_w = _check_s3_write(config.s3_checkpoint_bucket, config.aws_region)
            _print(_PASS if ok_w else _FAIL, msg_w)
            results.append(ok_w)
    else:
        _print(_SKIP, "S3 checkpointing disabled (no bucket configured)")

    # 3. DynamoDB table
    if config.state_table_name:
        ok, msg = _check_dynamodb_table(config.state_table_name, config.aws_region)
        _print(_PASS if ok else _FAIL, msg)
        results.append(ok)
    else:
        _print(_SKIP, "DynamoDB context store disabled (no table configured)")

    # 4. CloudWatch direct put
    ok, msg = _check_cloudwatch_put(config.aws_region)
    _print(_PASS if ok else _FAIL, msg)
    results.append(ok)

    # Summary
    passed = sum(results)
    total = len(results)
    all_ok = all(results)
    symbol = _PASS if all_ok else _FAIL
    print(f"\n{symbol}  {passed}/{total} checks passed.\n")

    if not all_ok:
        print("  Run 'ml-platform bootstrap' to create missing resources.")
        print("  See the README for the minimum IAM policy.\n")

    return all_ok
