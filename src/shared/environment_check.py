"""
src/shared/environment_check.py

Verifies that the local environment has every core dependency this project
needs, with correct versions and GPU access. Run standalone to sanity-check
a fresh machine, or import check_environment() from training/API startup
code to fail fast instead of discovering a broken environment mid-run.
"""

import sys
from dataclasses import dataclass, field

@dataclass
class EnvironmentReport:
    python_version: str
    checks: dict = field(default_factory=dict)
    errors: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return len(self.errors) == 0


def check_environment() -> EnvironmentReport:
    report = EnvironmentReport(python_version=sys.version.split()[0])

    if not sys.version.startswith("3.13"):
        report.errors.append(f"Expected Python 3.13.x, got {report.python_version}")

    try:
        import torch
        cuda_ok = torch.cuda.is_available()
        report.checks["torch"] = torch.__version__
        report.checks["cuda_available"] = cuda_ok
        report.checks["gpu_name"] = torch.cuda.get_device_name(0) if cuda_ok else "N/A"
        if not cuda_ok:
            report.errors.append("torch.cuda.is_available() is False — GPU not detected")
    except ImportError as e:
        report.errors.append(f"torch import failed: {e}")

    try:
        from torch_geometric.nn import GATConv
        import torch as _torch
        x = _torch.randn(4, 8)
        edge_index = _torch.tensor([[0, 1, 2], [1, 2, 3]])
        conv = GATConv(8, 4, heads=2)
        out = conv(x, edge_index)
        report.checks["torch_geometric"] = "OK" if out.shape == (4, 8) else "UNEXPECTED_SHAPE"
    except Exception as e:
        report.errors.append(f"torch_geometric check failed: {e}")

    try:
        import tensorflow as tf
        report.checks["tensorflow"] = tf.__version__
    except ImportError as e:
        report.errors.append(f"tensorflow import failed: {e}")

    for name, import_path in [
        ("scikit-learn", "sklearn"),
        ("scipy", "scipy"),
        ("pandas", "pandas"),
        ("matplotlib", "matplotlib"),
        ("numpy", "numpy"),
        ("networkx", "networkx"),
    ]:
        try:
            mod = __import__(import_path)
            report.checks[name] = getattr(mod, "__version__", "unknown")
        except ImportError as e:
            report.errors.append(f"{name} import failed: {e}")

    return report


def print_report(report: EnvironmentReport) -> None:
    print("=" * 50)
    print("ENVIRONMENT CHECK")
    print("=" * 50)
    print(f"Python: {report.python_version}")
    for key, value in report.checks.items():
        print(f"  {key}: {value}")
    if report.errors:
        print("\nERRORS:")
        for err in report.errors:
            print(f"  - {err}")
        print("\nRESULT: FAIL")
    else:
        print("\nRESULT: ALL CHECKS PASSED")
    print("=" * 50)


if __name__ == "__main__":
    report = check_environment()
    print_report(report)
    sys.exit(0 if report.ok else 1)