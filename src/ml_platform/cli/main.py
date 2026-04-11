"""Entry point for the ``ml-platform`` CLI.

Usage::

    ml-platform check     --service-name my-svc --s3-bucket my-ckpt
    ml-platform bootstrap --service-name my-svc --s3-bucket my-ckpt
    ml-platform bootstrap --service-name my-svc --github-oidc --repo myorg/my-svc
    ml-platform init      --template agent --name my-agent
    ml-platform deploy aws       --service-name my-chatbot
    ml-platform deploy sagemaker --service-name my-chatbot
    ml-platform destroy aws       --service-name my-chatbot
    ml-platform destroy sagemaker --service-name my-chatbot
"""

from __future__ import annotations

import argparse
import sys

from ml_platform.cli.check import run_check
from ml_platform.cli.bootstrap import run_bootstrap
from ml_platform.config import ServiceConfig, resolve_region


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ml-platform",
        description="Setup and validation utilities for ml-platform services.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # -- Shared args for check / bootstrap -----------------------------------
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument(
        "--service-name", required=True, help="Service name (used for resource naming)."
    )
    shared.add_argument(
        "--region", default=None, help="AWS region (default: auto-detected)."
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
    bp.add_argument(
        "--github-oidc",
        action="store_true",
        help="Create the GitHub Actions OIDC provider and deploy role.",
    )
    bp.add_argument(
        "--repo",
        default="",
        help="GitHub repository (owner/repo) for OIDC trust policy.",
    )

    # -- init ----------------------------------------------------------------
    init_p = sub.add_parser(
        "init",
        help="Scaffold a new project from a template.",
    )
    init_p.add_argument(
        "--template",
        choices=["agent", "chatbot", "bandit"],
        default="agent",
        help="Project template (default: agent).",
    )
    init_p.add_argument(
        "--name",
        required=True,
        help="Project / service name.",
    )
    init_p.add_argument(
        "--output-dir",
        default="",
        help="Parent directory for the new project (default: cwd).",
    )

    # -- deploy --------------------------------------------------------------
    deploy_p = sub.add_parser(
        "deploy",
        help="Deploy the service to a target environment.",
    )
    deploy_p.add_argument(
        "target",
        choices=["aws", "sagemaker", "local"],
        help="Deployment target.",
    )
    deploy_p.add_argument(
        "--service-name", default="", help="Override service name from manifest."
    )
    deploy_p.add_argument(
        "--manifest", default="ml-platform.yaml", help="Path to the project manifest."
    )
    deploy_p.add_argument(
        "--yes", action="store_true", help="Skip approval prompt (CI/CD)."
    )

    # -- destroy -------------------------------------------------------------
    destroy_p = sub.add_parser(
        "destroy",
        help="Tear down deployed resources.",
    )
    destroy_p.add_argument(
        "target",
        choices=["aws", "sagemaker", "local"],
        help="Deployment target.",
    )
    destroy_p.add_argument(
        "--service-name", required=True, help="Service to destroy."
    )
    destroy_p.add_argument(
        "--region", default=None, help="AWS region (default: auto-detected)."
    )
    destroy_p.add_argument(
        "--force", action="store_true", help="Skip name confirmation."
    )
    destroy_p.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify that resources are deleted, don't delete anything.",
    )

    return parser


def _config_from_args(args: argparse.Namespace) -> ServiceConfig:
    table = args.dynamodb_table or f"{args.service_name}-context"
    return ServiceConfig(
        service_name=args.service_name,
        aws_region=resolve_region(args.region),
        s3_checkpoint_bucket=args.s3_bucket,
        s3_checkpoint_prefix=args.s3_prefix,
        state_table_name=table,
    )


def main() -> None:
    """Parse arguments and dispatch to the appropriate subcommand."""
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "init":
        from ml_platform.cli.templates import generate_template

        output = args.output_dir or None
        path = generate_template(args.name, args.template, output)
        print(f"Created {args.template} project at {path}")
        print(f"  cd {args.name}")
        print(f"  pip install -e .")
        print(f"  python app.py")
        sys.exit(0)

    if args.command == "deploy":
        if args.target == "aws":
            from ml_platform.cli.deploy import run_deploy

            ok = run_deploy(
                service_name=args.service_name,
                auto_approve=args.yes,
                manifest_path=args.manifest,
            )
            sys.exit(0 if ok else 1)
        elif args.target == "sagemaker":
            from ml_platform.cli.deploy_sagemaker import run_deploy_sagemaker

            ok = run_deploy_sagemaker(
                service_name=args.service_name,
                auto_approve=args.yes,
                manifest_path=args.manifest,
            )
            sys.exit(0 if ok else 1)
        elif args.target == "local":
            print("Local deployment uses Docker Compose.")
            print("Run:  docker compose -f docker-compose.dev.yml up")
            sys.exit(0)

    if args.command == "destroy":
        if args.target == "aws":
            from ml_platform.cli.destroy import run_destroy

            ok = run_destroy(
                service_name=args.service_name,
                region=resolve_region(args.region),
                force=args.force,
                verify_only=args.verify_only,
            )
            sys.exit(0 if ok else 1)
        elif args.target == "sagemaker":
            from ml_platform.cli.destroy_sagemaker import run_destroy_sagemaker

            ok = run_destroy_sagemaker(
                service_name=args.service_name,
                region=resolve_region(args.region),
                force=args.force,
                verify_only=args.verify_only,
            )
            sys.exit(0 if ok else 1)
        elif args.target == "local":
            print("Run:  docker compose -f docker-compose.dev.yml down -v")
            sys.exit(0)

    config = _config_from_args(args)

    if args.command == "check":
        ok = run_check(config)
        sys.exit(0 if ok else 1)
    elif args.command == "bootstrap":
        if args.github_oidc:
            if not args.repo:
                parser.error("--github-oidc requires --repo owner/repo")
            from ml_platform.cli.github_oidc import bootstrap_github_oidc

            try:
                bootstrap_github_oidc(
                    repo=args.repo,
                    service_name=args.service_name,
                    region=resolve_region(args.region),
                    dry_run=args.dry_run,
                )
            except (ValueError, RuntimeError) as exc:
                print(f"\n  Error: {exc}\n", file=sys.stderr)
                sys.exit(1)
            sys.exit(0)

        ok = run_bootstrap(config, dry_run=args.dry_run)
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
