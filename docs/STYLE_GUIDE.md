# AIMBOT — Style Guide

> All Python code in this repository follows these conventions.
> If a rule conflicts with an external library's style, document the exception here.

---

## 1. Language

- **All source code, comments, docstrings, and commit messages** must be written in **English**.
- User-facing `print()` messages (CLI output visible to the operator) may be in Traditional Chinese
  when the primary audience is Mandarin-speaking operators. These should eventually be extracted to
  a locale module; until then, keep them grouped at call sites and mark them `# i18n`.

---

## 2. Formatting

| Rule        | Setting                                                   |
| ----------- | --------------------------------------------------------- |
| Line length | 120 characters                                            |
| Indentation | 4 spaces (no tabs)                                        |
| Blank lines | PEP 8: 2 between top-level definitions, 1 between methods |
| Tool        | `ruff format` (configured in `pyproject.toml`)            |

Run before committing:

```sh
uv run ruff format .
uv run ruff check --fix .
```

---

## 3. Imports

Order (enforced by `ruff`'s isort-compatible section rules):

```python
# 1. future
from __future__ import annotations

# 2. stdlib
import os
import sys
from typing import Optional, List

# 3. third-party
import cv2
import numpy as np
import torch

# 4. first-party (src/ packages declared in pyproject.toml known-first-party)
from core.config import PipelineConfig
from detection.detector import YoloV7Detector
```

Rules:

- No wildcard imports (`from x import *`).
- Lazy imports (inside a function body) are permitted **only** when the import
  has a known risk of failing at startup (e.g. `serial`, `torchreid`).
  Add a `# lazy: <reason>` comment on the import line.
- Avoid circular imports. If two modules need each other, extract shared types
  to `core/`.

---

## 4. Type Annotations

- All **public** functions and methods **must** have annotated parameters and
  return type.
- Private helpers (`_foo`) should also be annotated but exceptions are allowed
  when the annotation is trivially obvious.
- Use `Optional[X]` rather than `X | None` for Python ≤ 3.9 compatibility
  (even though the project targets 3.11, the codebase uses `Optional` for
  consistency with historical code).
- Use `from __future__ import annotations` at the top of every file to enable
  PEP 563 deferred evaluation and allow forward references.
- Use `typing.List`, `typing.Tuple`, `typing.Set`, etc., **not** the built-in
  lowercase equivalents in function signatures (the built-in lowercase forms
  are fine inside `from __future__ import annotations` contexts but `typing.*`
  is preferred for clarity).

---

## 5. Docstrings (Google Style, PEP 257)

Every public class, method, and module-level function must have a docstring.

### Module-level

```python
"""One-line summary.

Longer description if necessary.  Wrap at 80 characters even though code
uses 120, so docstrings are readable in terminal pagers.
"""
```

### Classes

```python
class Foo:
    """One-line summary.

    More context.

    Attributes:
        bar: Description of instance attribute.
    """
```

### Functions / methods

```python
def encode(self, image: np.ndarray) -> np.ndarray:
    """Extract a single L2-normalised feature vector.

    Args:
        image: BGR crop, uint8, arbitrary size.

    Returns:
        1-D float32 array of length ``feature_dim``.

    Raises:
        ValueError: If ``image`` has zero-area dimensions.
    """
```

Omit `Returns:` for `-> None`. Omit `Raises:` when the function does not
raise deliberately.

---

## 6. Naming Conventions

| Symbol                   | Convention            | Example                      |
| ------------------------ | --------------------- | ---------------------------- |
| Modules                  | `snake_case`          | `byte_tracker.py`            |
| Packages                 | `snake_case`          | `pipeline/`                  |
| Classes                  | `PascalCase`          | `GimbalController`           |
| Functions / methods      | `snake_case`          | `encode_batch()`             |
| Constants (module-level) | `UPPER_SNAKE`         | `DEFAULT_DEVICE`             |
| Private attributes       | `_leading_underscore` | `self._serial`               |
| Type aliases             | `PascalCase`          | `FeatureVector = np.ndarray` |
| Local variables          | `snake_case`          | `track_id`, `pan_cmd`        |

---

## 7. Error Handling

- Use specific exception types, never bare `except:`.
- Hardware errors (serial, camera) should propagate with enough context for the
  caller to produce a useful error message.
- Do **not** swallow exceptions unless the code path is explicitly a "best
  effort" branch; mark such branches with `# best-effort` and log the exception
  at `DEBUG` level.
- Raise `ImportError` with a helpful message when an optional dependency is
  missing (see `reid/osnet.py` `try/except ImportError` pattern).

---

## 8. Logging vs. print

- Use `print()` only in `scripts/` (CLI entry points) for user-facing status
  messages.
- Inside `src/` library code, prefer `logging` at the appropriate level so
  callers can control verbosity.
- **Never** log sensitive data (serial port commands, raw frame data at high
  verbosity) at `INFO` or above in production paths.

---

## 9. Config vs. Magic Numbers

- All tuneable parameters must live in a `core/config.py` dataclass, never as
  module-level magic numbers scattered through the code.
- If you add a threshold, gain, or flag, add it as a field to the appropriate
  frozen dataclass (`RuntimeConfig`, `TrackingConfig`, `PIDConfig`,
  `ControlConfig`) with a default value and a doc comment.
- Override via `dataclasses.replace()` at the entry point; never mutate config
  objects in-place.

---

## 10. Testing Conventions

_(placeholder — tests not yet written)_

- Unit tests in `tests/` mirroring `src/` package structure.
- Use `pytest`.
- Hardware-dependent tests (serial, camera) must be marked `@pytest.mark.hardware`
  and skipped in CI via `pytest -m "not hardware"`.
