"""CI/CD pipeline generation for GitHub Actions and AWS CodePipeline.

Scaffolds CI/CD configuration files alongside the existing
``ml-platform init`` templates.

Usage::

    from ml_platform.cicd import generate_github_actions, generate_codepipeline

    yaml_str = generate_github_actions(
        service_name="inference-api",
        python_version="3.12",
    )
"""

from __future__ import annotations

import logging

from ml_platform.config import resolve_region

logger = logging.getLogger(__name__)

__all__ = [
    "generate_github_actions",
    "generate_codepipeline_buildspec",
]


def generate_github_actions(
    *,
    service_name: str,
    python_version: str = "3.12",
    aws_region: str | None = None,
    ecr_repository: str = "",
    run_tests: bool = True,
    deploy_ecs: bool = False,
    deploy_lambda: bool = False,
) -> str:
    """Generate a GitHub Actions CI/CD workflow YAML.

    Args:
        service_name: Name of the service (used in workflow name and
            resource references).
        python_version: Python version for the test/build matrix.
        aws_region: AWS region for deployment steps.
        ecr_repository: ECR repository URI for Docker builds. Empty
            disables Docker steps.
        run_tests: Include a test job.
        deploy_ecs: Include an ECS Fargate deployment step.
        deploy_lambda: Include a Lambda deployment step.

    Returns:
        GitHub Actions workflow YAML as a string.
    """
    resolved_region = resolve_region(aws_region)
    lines: list[str] = [
        f"name: {service_name} CI/CD",
        "",
        "on:",
        "  push:",
        "    branches: [main]",
        "  pull_request:",
        "    branches: [main]",
        "",
        "env:",
        f"  AWS_REGION: {resolved_region}",
    ]

    if ecr_repository:
        lines.append(f"  ECR_REPOSITORY: {ecr_repository}")

    lines.extend(["", "jobs:"])

    if run_tests:
        lines.extend([
            "  test:",
            "    runs-on: ubuntu-latest",
            "    steps:",
            "      - uses: actions/checkout@v4",
            f"      - uses: actions/setup-python@v5",
            "        with:",
            f"          python-version: '{python_version}'",
            "      - name: Install dependencies",
            "        run: pip install -e '.[dev]'",
            "      - name: Lint",
            "        run: ruff check src/ tests/",
            "      - name: Type check",
            "        run: mypy src/",
            "      - name: Test",
            "        run: pytest --cov=src/ --cov-report=xml -q",
            "      - name: Upload coverage",
            "        uses: codecov/codecov-action@v4",
            "        with:",
            "          file: coverage.xml",
            "",
        ])

    if ecr_repository:
        lines.extend([
            "  build:",
            f"    needs: {'test' if run_tests else ''}",
            "    runs-on: ubuntu-latest",
            "    if: github.ref == 'refs/heads/main'",
            "    permissions:",
            "      id-token: write",
            "      contents: read",
            "    steps:",
            "      - uses: actions/checkout@v4",
            "      - uses: aws-actions/configure-aws-credentials@v4",
            "        with:",
            "          role-to-assume: ${{ secrets.AWS_ROLE_ARN }}",
            "          aws-region: ${{ env.AWS_REGION }}",
            "      - uses: aws-actions/amazon-ecr-login@v2",
            "        id: ecr-login",
            "      - name: Build and push",
            "        run: |",
            "          IMAGE=${{ env.ECR_REPOSITORY }}:${{ github.sha }}",
            "          docker build -t $IMAGE .",
            "          docker push $IMAGE",
            "",
        ])

    if deploy_ecs:
        lines.extend([
            "  deploy-ecs:",
            "    needs: build",
            "    runs-on: ubuntu-latest",
            "    if: github.ref == 'refs/heads/main'",
            "    permissions:",
            "      id-token: write",
            "      contents: read",
            "    steps:",
            "      - uses: aws-actions/configure-aws-credentials@v4",
            "        with:",
            "          role-to-assume: ${{ secrets.AWS_ROLE_ARN }}",
            "          aws-region: ${{ env.AWS_REGION }}",
            "      - name: Deploy to ECS",
            "        run: |",
            f"          aws ecs update-service --cluster {service_name}-cluster \\",
            f"            --service {service_name} \\",
            "            --force-new-deployment",
            "",
        ])

    if deploy_lambda:
        needs = "build" if ecr_repository else ("test" if run_tests else "")
        lines.extend([
            "  deploy-lambda:",
            f"    needs: {needs}" if needs else "  deploy-lambda:",
            "    runs-on: ubuntu-latest",
            "    if: github.ref == 'refs/heads/main'",
            "    permissions:",
            "      id-token: write",
            "      contents: read",
            "    steps:",
            "      - uses: actions/checkout@v4",
            "      - uses: aws-actions/configure-aws-credentials@v4",
            "        with:",
            "          role-to-assume: ${{ secrets.AWS_ROLE_ARN }}",
            "          aws-region: ${{ env.AWS_REGION }}",
            f"      - uses: actions/setup-python@v5",
            "        with:",
            f"          python-version: '{python_version}'",
            "      - name: Package Lambda",
            "        run: |",
            "          pip install -e '.[stateful]' -t package/",
            "          cd package && zip -r ../lambda.zip .",
            "      - name: Deploy Lambda",
            "        run: |",
            f"          aws lambda update-function-code \\",
            f"            --function-name {service_name} \\",
            "            --zip-file fileb://lambda.zip",
            "",
        ])

    return "\n".join(lines)


def generate_codepipeline_buildspec(
    *,
    service_name: str,
    python_version: str = "3.12",
    run_tests: bool = True,
) -> str:
    """Generate an AWS CodeBuild buildspec.yml for CodePipeline.

    Args:
        service_name: Service name for build artifacts.
        python_version: Python runtime version.
        run_tests: Include test phase.

    Returns:
        Buildspec YAML as a string.
    """
    lines: list[str] = [
        "version: 0.2",
        "",
        "phases:",
        "  install:",
        "    runtime-versions:",
        f"      python: '{python_version}'",
        "    commands:",
        "      - pip install -e '.[dev]'",
    ]

    if run_tests:
        lines.extend([
            "  pre_build:",
            "    commands:",
            "      - ruff check src/ tests/",
            "      - pytest --cov=src/ -q",
        ])

    lines.extend([
        "  build:",
        "    commands:",
        "      - pip install -e '.[stateful]' -t dist/",
        f"      - cd dist && zip -r ../{service_name}.zip .",
        "",
        "artifacts:",
        "  files:",
        f"    - {service_name}.zip",
    ])

    return "\n".join(lines)
