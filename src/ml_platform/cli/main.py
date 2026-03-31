"""Entry point for the ``ml-platform`` CLI.

Usage::

    ml-platform check  --service-name my-svc --s3-bucket my-ckpt --region us-east-1
    ml-platform bootstrap --service-name my-svc --s3-bucket my-ckpt --region us-east-1
"""

from __future__ import annotations

import argparse
import sys

from ml_platform.cli.check import run_check
from ml_platform.cli.bootstrap import run_bootstrap
from ml_platform.config import ServiceConfig


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ml-platform",
        description="Setup and validation utilities for ml-platform services.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument(
        "--service-name", required=True, help="Service name (used for resource naming)."
    )
    shared.add_argument(
        "--region", default="us-east-1", help="AWS region (default: us-east-1)."
    )
    shared.add_argument(
        "--s3-bucket", default="", help="S3 checkpoint bucket name."
    )
    shared.add_argument(
        "--s3-prefix", default="checkpoints/", help="S3 key prefix."
    )
    shared.add_argument(
        "--dynamodb-table", default="", help="DynamoDB context table name."
    )

    sub.add_parser(
        "check",
        parents=[shared],
        help="Validate AWS credentials and resource availability.",
    )

    bp = sub.add_parser(
        "bootstrap",
        parents=[shared],
        help="Create required AWS resources and generate IAM policy.",
    )
    bp.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be created without making changes.",
    )

    return parser


def _config_from_args(args: argparse.Namespace) -> ServiceConfig:
    table = args.dynamodb_table or f"{args.service_name}-context"
    return ServiceConfig(
        service_name=args.service_name,
        aws_region=args.region,
        s3_checkpoint_bucket=args.s3_bucket,
        s3_checkpoint_prefix=args.s3_prefix,
        state_table_name=table,
    )


def main() -> None:
    """Parse arguments and dispatch to the appropriate subcommand."""
    parser = _build_parser()
    args = parser.parse_args()
    config = _config_from_args(args)

    if args.command == "check":
        ok = run_check(config)
        sys.exit(0 if ok else 1)
    elif args.command == "bootstrap":
        ok = run_bootstrap(config, dry_run=args.dry_run)
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
