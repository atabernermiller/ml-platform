"""Project manifest (``ml-platform.yaml``) reader, writer, and interactive generator.

The manifest declares what a project needs from its infrastructure.
The ``deploy`` command reads it to determine what CloudFormation
resources to create; the ``destroy`` command uses it to know what to
tear down.

If the manifest doesn't exist, :func:`interactive_create` walks the
user through a short questionnaire and writes the file.
"""

from __future__ import annotations

import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml  # type: ignore[import-untyped]

ServiceType = Literal["llm", "agent", "stateful", "stateless"]

_COMPUTE_SIZES: dict[str, tuple[int, int]] = {
    "small": (256, 512),       # 0.25 vCPU, 0.5 GB
    "medium": (512, 1024),     # 0.5 vCPU, 1 GB
    "large": (1024, 2048),     # 1 vCPU, 2 GB
    "xlarge": (2048, 4096),    # 2 vCPU, 4 GB
}


@dataclass
class ScalingConfig:
    """Auto-scaling parameters for ECS tasks."""

    min_tasks: int = 1
    max_tasks: int = 4
    scale_up_cpu: int = 70
    scale_down_cpu: int = 30


@dataclass
class FeaturesConfig:
    """Which optional platform features are enabled."""

    conversation_store: bool = False
    context_store: bool = False
    checkpointing: bool = False
    mlflow: bool = False


@dataclass
class ProjectManifest:
    """Parsed ``ml-platform.yaml`` manifest.

    Attributes:
        service_name: Unique service identifier.
        service_type: One of ``llm``, ``agent``, ``stateful``, ``stateless``.
        features: Which platform features are enabled.
        compute_size: Task size (``small``, ``medium``, ``large``, ``xlarge``).
        scaling: Auto-scaling configuration.
        region: AWS region for deployment.
    """

    service_name: str
    service_type: ServiceType = "llm"
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    compute_size: str = "medium"
    scaling: ScalingConfig = field(default_factory=ScalingConfig)
    region: str = "us-east-1"

    @property
    def cpu(self) -> int:
        return _COMPUTE_SIZES.get(self.compute_size, (512, 1024))[0]

    @property
    def memory(self) -> int:
        return _COMPUTE_SIZES.get(self.compute_size, (512, 1024))[1]


def load_manifest(path: Path | str = "ml-platform.yaml") -> ProjectManifest:
    """Load and validate a manifest from disk.

    Args:
        path: Path to the YAML file.

    Returns:
        Parsed manifest.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If required fields are missing.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Manifest not found: {p}")

    raw: dict[str, Any] = yaml.safe_load(p.read_text()) or {}

    if "service_name" not in raw:
        raise ValueError("ml-platform.yaml must include 'service_name'")

    features = FeaturesConfig(**raw.get("features", {}))
    scaling = ScalingConfig(**raw.get("scaling", {}))

    return ProjectManifest(
        service_name=raw["service_name"],
        service_type=raw.get("type", raw.get("service_type", "llm")),
        features=features,
        compute_size=raw.get("compute", {}).get("size", "medium")
        if isinstance(raw.get("compute"), dict)
        else raw.get("compute_size", "medium"),
        scaling=scaling,
        region=raw.get("region", "us-east-1"),
    )


def save_manifest(manifest: ProjectManifest, path: Path | str = "ml-platform.yaml") -> None:
    """Write a manifest to disk as YAML.

    Args:
        manifest: The manifest to save.
        path: Destination file path.
    """
    data: dict[str, Any] = {
        "service_name": manifest.service_name,
        "type": manifest.service_type,
        "region": manifest.region,
        "features": asdict(manifest.features),
        "compute": {"size": manifest.compute_size},
        "scaling": asdict(manifest.scaling),
    }
    Path(path).write_text(yaml.dump(data, default_flow_style=False, sort_keys=False))


def interactive_create(service_name: str = "") -> ProjectManifest:
    """Walk the user through a questionnaire and return a manifest.

    Prints to stdout, reads from stdin.

    Args:
        service_name: Pre-filled service name (prompts if empty).

    Returns:
        The generated manifest (not yet saved to disk).
    """
    print("\n  No ml-platform.yaml found. Let's figure out what your project needs.\n")

    if not service_name:
        service_name = _ask("  Service name: ").strip()
        if not service_name:
            print("  Service name is required.")
            sys.exit(1)

    print("  What type of service is this?")
    print("    [1] LLM app (chatbot, RAG, API proxy -- calls an LLM API)")
    print("    [2] Agent app (multi-step LLM + tools)")
    print("    [3] Stateful ML service (bandit, online learning -- has feedback loop)")
    print("    [4] Stateless inference (BentoML or similar)")
    choice = _ask("\n  Choice [1-4]: ").strip()
    type_map: dict[str, ServiceType] = {
        "1": "llm",
        "2": "agent",
        "3": "stateful",
        "4": "stateless",
    }
    svc_type = type_map.get(choice, "llm")

    features = FeaturesConfig()
    if svc_type in ("llm", "agent"):
        features.conversation_store = _yes_no("  Multi-turn conversations?")
    if svc_type == "stateful":
        features.context_store = True
        features.checkpointing = True
        features.conversation_store = False
    if svc_type in ("llm", "agent"):
        features.checkpointing = _yes_no("  S3 state checkpointing?")
    features.mlflow = _yes_no("  MLflow tracking server?")

    print("\n  Compute size:")
    print("    [1] small   (0.25 vCPU / 0.5 GB)")
    print("    [2] medium  (0.5 vCPU / 1 GB)")
    print("    [3] large   (1 vCPU / 2 GB)")
    print("    [4] xlarge  (2 vCPU / 4 GB)")
    size_choice = _ask("  Choice [1-4, default 2]: ").strip() or "2"
    size_map = {"1": "small", "2": "medium", "3": "large", "4": "xlarge"}
    compute_size = size_map.get(size_choice, "medium")

    region = _ask("  AWS region [us-east-1]: ").strip() or "us-east-1"

    manifest = ProjectManifest(
        service_name=service_name,
        service_type=svc_type,
        features=features,
        compute_size=compute_size,
        region=region,
    )
    return manifest


def _ask(prompt: str) -> str:
    try:
        return input(prompt)
    except (EOFError, KeyboardInterrupt):
        print("\n  Cancelled.")
        sys.exit(1)


def _yes_no(prompt: str) -> bool:
    answer = _ask(f"{prompt} [y/n]: ").strip().lower()
    return answer in ("y", "yes")
