"""
Output management utilities.

Handles UUID-based naming, directory creation, cleanup policies,
retention management, and path traversal protection for all
generated output files (heatmaps, masks, answers).
"""

from __future__ import annotations

import os
import time
import uuid
from pathlib import Path
from typing import Optional

from loguru import logger


class OutputManager:
    """
    Manages output file lifecycle.

    Features:
    - UUID-based unique file naming
    - Automatic directory creation
    - Path traversal protection
    - Retention-based cleanup
    - Safe file operations
    """

    def __init__(self, base_dir: str | Path):
        self.base_dir = Path(base_dir).resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def get_output_path(
        self,
        subdir: str,
        filename: str,
        prefix: str = "",
    ) -> Path:
        """
        Get a safe output path with directory creation.

        Args:
            subdir: Subdirectory name (e.g., 'heatmaps', 'masks').
            filename: Base filename.
            prefix: Optional prefix for the filename.

        Returns:
            Resolved absolute Path.

        Raises:
            ValueError: If path traversal is detected.
        """
        # Sanitize inputs
        subdir = self._sanitize_path_component(subdir)
        filename = self._sanitize_path_component(filename)

        output_dir = self.base_dir / subdir
        output_dir.mkdir(parents=True, exist_ok=True)

        full_path = (output_dir / f"{prefix}{filename}").resolve()

        # Path traversal protection
        if not str(full_path).startswith(str(self.base_dir)):
            raise ValueError(
                f"Path traversal detected: {full_path} is outside {self.base_dir}"
            )

        return full_path

    def generate_unique_prefix(self) -> str:
        """Generate a UUID-based unique prefix for output files."""
        return f"{uuid.uuid4().hex[:8]}_"

    def cleanup_old_outputs(
        self,
        max_age_hours: float = 24.0,
        max_files: int = 1000,
    ) -> int:
        """
        Remove old output files based on retention policy.

        Args:
            max_age_hours: Maximum age in hours before deletion.
            max_files: Maximum total files to keep.

        Returns:
            Number of files deleted.
        """
        deleted = 0
        now = time.time()
        max_age_seconds = max_age_hours * 3600

        all_files = []
        for path in self.base_dir.rglob("*"):
            if path.is_file() and path.name != ".gitkeep":
                all_files.append(path)

        # Delete by age
        for path in all_files:
            try:
                age = now - path.stat().st_mtime
                if age > max_age_seconds:
                    path.unlink()
                    deleted += 1
            except OSError as e:
                logger.debug(f"Could not delete {path}: {e}")

        # Delete by count (oldest first)
        remaining = [p for p in all_files if p.exists()]
        if len(remaining) > max_files:
            remaining.sort(key=lambda p: p.stat().st_mtime)
            for path in remaining[:len(remaining) - max_files]:
                try:
                    path.unlink()
                    deleted += 1
                except OSError:
                    pass

        if deleted > 0:
            logger.info(f"Cleaned up {deleted} old output files")

        return deleted

    def get_disk_usage(self) -> dict:
        """Get disk usage information for the output directory."""
        total_size = 0
        file_count = 0

        for path in self.base_dir.rglob("*"):
            if path.is_file():
                total_size += path.stat().st_size
                file_count += 1

        return {
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "file_count": file_count,
            "base_dir": str(self.base_dir),
        }

    @staticmethod
    def _sanitize_path_component(component: str) -> str:
        """
        Sanitize a path component to prevent traversal attacks.

        Removes dangerous characters and patterns.
        """
        # Remove path separators and parent directory references
        sanitized = component.replace("..", "").replace("/", "").replace("\\", "")
        # Remove null bytes
        sanitized = sanitized.replace("\x00", "")
        # Limit length
        sanitized = sanitized[:255]

        if not sanitized:
            raise ValueError(f"Invalid path component: '{component}'")

        return sanitized
