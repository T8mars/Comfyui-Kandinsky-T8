"""Install bundled workflow inputs into ComfyUI without overwriting user files."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path


LOGGER = logging.getLogger(__name__)
EXAMPLE_INPUTS = Path(__file__).resolve().parent / "example_inputs"
INPUT_SUBDIRECTORY = "kandinsky6"


def install_example_inputs(input_directory: str | Path | None = None) -> tuple[Path, ...]:
    """Copy bundled inputs into ComfyUI's input directory on first load."""
    if input_directory is None:
        try:
            import folder_paths

            input_directory = folder_paths.get_input_directory()
        except Exception as error:  # pragma: no cover - depends on ComfyUI startup
            LOGGER.warning("Unable to locate the ComfyUI input directory: %s", error)
            return ()

    if not EXAMPLE_INPUTS.is_dir():  # pragma: no cover - malformed installation
        LOGGER.warning("Bundled example-input directory is missing: %s", EXAMPLE_INPUTS)
        return ()

    target_directory = Path(input_directory) / INPUT_SUBDIRECTORY
    try:
        target_directory.mkdir(parents=True, exist_ok=True)
    except OSError as error:  # pragma: no cover - filesystem-specific failure
        LOGGER.warning("Unable to create example-input directory %s: %s", target_directory, error)
        return ()

    installed: list[Path] = []
    for source in sorted(EXAMPLE_INPUTS.iterdir()):
        if not source.is_file():
            continue

        target = target_directory / source.name
        target_was_created = False
        try:
            with source.open("rb") as source_file, target.open("xb") as target_file:
                target_was_created = True
                shutil.copyfileobj(source_file, target_file)
        except FileExistsError:
            continue
        except OSError as error:  # pragma: no cover - filesystem-specific failure
            if target_was_created:
                target.unlink(missing_ok=True)
            LOGGER.warning("Unable to install example input %s: %s", source.name, error)
            continue
        installed.append(target)

    return tuple(installed)
