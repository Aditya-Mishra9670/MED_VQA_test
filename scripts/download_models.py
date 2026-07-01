"""
Script to download necessary models for the Medical VQA System.

Usage:
    python scripts/download_models.py --model all
    python scripts/download_models.py --model stllava
    python scripts/download_models.py --model localization

Can also be run as a module:
    python -m backend.models.model_manager
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is on path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def main():
    parser = argparse.ArgumentParser(
        description="Download model weights for the Medical VQA System"
    )
    parser.add_argument(
        "--model",
        choices=["stllava", "localization", "all"],
        default="all",
        help="Which models to download",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check availability without downloading",
    )
    args = parser.parse_args()

    from backend.models.model_manager import ModelManager

    manager = ModelManager()

    if args.check_only:
        print("\n=== Model Availability ===\n")
        status = manager.check_models()
        for name, available in status.items():
            icon = "✓" if available else "✗"
            print(f"  {icon} {name}: {'available' if available else 'not found'}")

        print("\n=== Model Validation ===\n")
        validation = manager.validate_models()
        for name, result in validation.items():
            icon = "✓" if result.get("valid") else "✗"
            path = result.get("path", "not downloaded")
            print(f"  {icon} {name}: {path}")
        return

    print("\n=== Medical VQA Model Downloader ===\n")

    if args.model in ["stllava", "all"]:
        print("--- STLLaVA-Med ---")
        manager.ensure_llava_package()
        manager.ensure_stllava_available()
        manager.ensure_llava_base_available()

    if args.model in ["localization", "all"]:
        print("\n--- Grounding DINO ---")
        manager.ensure_grounding_dino_available()

        print("\n--- SAM2 ---")
        manager.ensure_sam2_available()

    print("\n=== Download Complete ===")
    print("\nFinal status:")
    status = manager.check_models()
    for name, available in status.items():
        icon = "✓" if available else "✗"
        print(f"  {icon} {name}")


if __name__ == "__main__":
    main()
