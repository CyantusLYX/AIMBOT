# AIMBOT — Refactoring Decisions Log

> Each entry records **what** was changed, **why**, any **alternatives** considered,
> and the **risk / validation** status.

---

## Table of Contents

1. [Centralized frozen-dataclass config](#1-centralized-frozen-dataclass-config)
2. [GimbalBase ABC (hardware abstraction layer)](#2-gimbalbase-abc-hardware-abstraction-layer)
3. [Lazy serial import in GimbalController](#3-lazy-serial-import-in-gimbalcontroller)
4. [PID gains — empirical values](#4-pid-gains--empirical-values)
5. [Re-ID threshold values](#5-re-id-threshold-values)
6. [numpy < 1.24 pin](#6-numpy--124-pin)
7. [protobuf < 4.21.3 pin](#7-protobuf--4213-pin)
8. [torchreid == 0.2.5 pin](#8-torchreid--025-pin)
9. [CUDA variant not in pyproject.toml](#9-cuda-variant-not-in-pyprojecttoml)
10. [Google-style docstrings](#10-google-style-docstrings)
11. [self.serial → self.\_serial rename](#11-selfserial--self_serial-rename)
12. [Optional[Set[int]] instead of set[int] in viewer.py](#12-optionalsetint-instead-of-setint-in-viewerpy)
13. [Distributed PC brain ownership in src/app](#13-distributed-pc-brain-ownership-in-srcapp)
14. [Local BoT-SORT backend as default tracker](#14-local-bot-sort-backend-as-default-tracker)

---

## 1. Centralized frozen-dataclass config

**What**: All tuneable parameters were moved from scattered module-level
`CONSTANT = value` assignments in `run_pipeline.py` into five frozen
dataclasses in `src/core/config.py`:
`RuntimeConfig`, `TrackingConfig`, `PIDConfig`, `ControlConfig`, `PipelineConfig`.

**Why**:

- Single source of truth. Searching for a threshold no longer requires grepping
  multiple files.
- Frozen (`frozen=True`) dataclasses are immutable after construction, preventing
  accidental mutation of shared state across threads.
- `dataclasses.replace()` provides an explicit, traceable "override" pattern for
  CLI argument processing.
- IDE autocomplete and mypy can type-check threshold names.

**Alternatives considered**:

- YAML/TOML config file: adds a file I/O dependency at startup; harder to type-check.
- `argparse` namespace as config: mutable, no nesting, hard to mock in tests.
- `pydantic` BaseSettings: heavier dependency, not needed at this scale.

**Risk**: Low. The config is consumed only at pipeline start.  
**Validation**: `python -m compileall src/core/config.py` passes; `run_pipeline.py`
loads defaults without error.

---

## 2. GimbalBase ABC (hardware abstraction layer)

**What**: Added `GimbalBase(ABC)` in `src/control/gimbal_controller.py` declaring
the `send(pan_speed, tilt_speed)` and `close()` abstract methods with a
`__enter__`/`__exit__` context manager on the base class. `GimbalController`
now extends `GimbalBase`.

**Why**:

- Enables drop-in replacement of `GimbalController` with a mock/stub in tests
  without touching production code.
- `dry_run=True` path was already present; the ABC formalises the contract.
- Phase 5 of the refactoring plan required "hardware abstraction layer".

**Alternatives considered**:

- `typing.Protocol` (structural subtyping): slightly lighter but less IDE-visible;
  ABC makes the relationship explicit and enforces implementation completeness.

**Risk**: Low — only affects class hierarchy, no logic change.  
**Validation**: `python -m compileall src/control/gimbal_controller.py` passes.

---

## 3. Lazy serial import in GimbalController

**What**: `import serial` was moved from module level into `GimbalController.__init__`
behind a `if not dry_run:` guard.

**Why**:

- `pyserial` is an optional runtime dependency (not needed when running in
  `dry_run=True` mode or when developing on a machine without a gimbal attached).
- A module-level `import serial` causes an `ImportError` that propagates to every
  module that imports `gimbal_controller`, blocking the entire program.

**Alternatives considered**:

- `try/except ImportError` at module level and set `serial = None`: works but
  makes `None` checks necessary everywhere the module is used.
- Separate `DryRunGimbalController` subclass with no serial dependency: cleaner
  but doubles the class count for a simple flag.

**Risk**: Negligible — the lazy path is guarded by `dry_run` which is `False` by
default (hardware mode assumed in production).  
**Validation**: `import gimbal_controller` succeeds even when `pyserial` is absent.

---

## 4. PID gains — empirical values

**What**: PID gains (`kp=0.005, ki=0.0001, kd=0.0005`) are stored in `PIDConfig`.

**Why these values**:  
These gains were determined empirically by testing against the specific gimbal
hardware used in development (see `hardware/` directory for the device spec
sheet). They are **hardware-specific** and are expected to require re-tuning if
a different gimbal or camera mounting is used.

- `kp=0.005`: Small proportional gain prevents overshoot at typical target speeds.
- `ki=0.0001`: Very small integral term; mainly compensates for static DC offset
  due to mechanical backlash.
- `kd=0.0005`: Light derivative dampening; high `kd` values caused oscillation
  on the specific stepper driver used.

**Risk**: If hardware changes, these will need re-tuning.  
**Validation**: Field-tested on target hardware; no automated test exists yet.

---

## 5. Re-ID threshold values

**What**: `TrackingConfig` contains:

- `reid_similarity = 0.60` — cosine similarity threshold to accept a Re-ID match
- `reid_distance = 0.25` — cosine _distance_ threshold (= 1 − similarity) for
  the matching cost matrix in ByteTrack
- `feature_momentum = 0.90` — EMA momentum for running feature update:
  `stored = momentum * stored + (1 − momentum) * new`

**Why these values**:

- `reid_similarity = 0.60` is a commonly used operating point on the Market-1501
  benchmark for OSNet models, balancing precision (not matching wrong person)
  vs. recall (recovering after occlusion). Values above 0.75 caused frequent
  track fragmentation in testing.
- `reid_distance = 0.25` corresponds to `1 − 0.75` — the ByteTrack integration
  uses distance rather than similarity; this caps the acceptable match cost.
- `feature_momentum = 0.90` is a standard value in online Re-ID literature.
  Lower values (e.g. 0.7) caused the stored feature to drift too quickly under
  lighting changes; higher values (0.95+) slowed recovery from pose changes.

**Risk**: Environment-specific; may need tuning for different lighting/clothing.  
**Validation**: Qualitative testing on captured video sequences.

---

## 6. numpy < 1.24 pin

**What**: `requirements.txt` and `pyproject.toml` pin `numpy<1.24`.

**Why**:

- `torchreid 0.2.5` uses `np.bool` which was deprecated in NumPy 1.20 and
  **removed** in NumPy 1.24. Installing NumPy ≥ 1.24 causes `AttributeError`
  at import time inside torchreid.
- `filterpy` (ByteTrack dependency) has a similar issue in some versions.

**Alternatives considered**:

- Monkey-patching `np.bool = bool` at startup: fragile and makes behaviour
  undefined if numpy changes other APIs.
- Upgrading torchreid: 0.3.x is available but requires API changes in OSNetEmbedder.

**Risk**: NumPy < 1.24 misses several bug-fixes and performance improvements.
When torchreid is eventually replaced or upgraded, this pin should be lifted.  
**Validation**: `uv pip install numpy<1.24` + `import torchreid` succeeds.

---

## 7. protobuf < 4.21.3 pin

**What**: `requirements.txt` and `pyproject.toml` pin `protobuf<4.21.3`.

**Why**:

- torchreid 0.2.5 depends on `tensorflow` at import time in some paths; TF 2.x
  that pairs with torchreid requires `protobuf<4.x` (protobuf 4.21.3+ changed
  the Python API).
- Even without TF, some torchreid data-loader paths call protobuf APIs directly.

**Risk**: Protobuf 3.x has no known security issues relevant to this project
(network-only risk; this project uses it only for local model metadata).  
**Validation**: `import torchreid` succeeds with the pin.

---

## 8. torchreid == 0.2.5 pin

**What**: `pyproject.toml` pins `torchreid==0.2.5` in `[project.optional-dependencies]`.

**Why**:

- `OSNetEmbedder` in `reid/osnet.py` calls `torchreid.models.build_model()` with
  the 0.2.x API. Version 0.3.x changed the model-builder signature.
- The `epoch_149.pt` model checkpoint was exported for 0.2.5 weight format.

**Risk**: Pinned to a specific minor version; will need code updates if upgraded.  
**Validation**: `OSNetEmbedder.__init__` runs without error; `encode()` produces
a 512-dimensional feature vector.

---

## 9. CUDA variant not in pyproject.toml

**What**: `torch` and `torchvision` are listed as `torch-cpu` optional dependencies
pointing to the CPU-only wheels. The CUDA-enabled wheels are **not** in
`pyproject.toml`.

**Why**:

- PyTorch's CUDA wheels are served from a custom index URL
  (`https://download.pytorch.org/whl/cu121`), which `uv` / pip cannot
  resolve via PyPI. Adding an index URL to `pyproject.toml` would force all
  contributors to use CUDA builds even on machines without a GPU.
- The correct approach is documented in `README.md`: install CUDA torch manually
  before running `uv sync`:
  ```sh
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
  # or
  uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
  ```

**Alternatives considered**:

- `[tool.uv.sources]` with `index` override: uv 0.2+ supports this but it is
  not yet stable and would couple the project to a uv-specific feature.

**Risk**: Developer must remember the extra install step. README and this document
both call it out.  
**Validation**: `import torch; torch.cuda.is_available()` returns `True` after
manual CUDA install.

---

## 10. Google-style docstrings

**What**: All public APIs use Google-style docstrings (Args/Returns/Raises sections).

**Why**:

- Google style is the most compact of the mainstream formats and parses
  correctly by `sphinx.ext.napoleon`, `mkdocstrings`, and `pdoc`.
- The ruff `D` rule set is configured in `pyproject.toml` to enforce
  `google` convention (`[tool.ruff.lint.pydocstyle] convention = "google"`).

**Alternatives considered**:

- NumPy style: verbose, common in scientific Python but not needed here.
- reStructuredText (Sphinx default): noisy syntax, harder to read inline.

---

## 11. self.serial → self.\_serial rename

**What**: `GimbalController`'s instance attribute was renamed from `self.serial`
to `self._serial`.

**Why**:

- The attribute is an implementation detail (`serial.Serial` object). Public
  callers should never access it directly; the `send()` and `close()` methods
  are the only intended interface (as per `GimbalBase`).
- The PEP 8 single-underscore convention signals "internal use" without
  enforcing it.

**Impact**: Any code that accessed `controller.serial` directly must be updated
to go through the public API. A grep of the codebase showed no external callers.

---

## 12. Optional[Set[int]] instead of set[int] in viewer.py

**What**: `secondary_target_ids: Optional[Set[int]]` uses `typing.Set` rather
than the built-in `set[int]`.

**Why**:

- Although Python 3.9+ supports `set[int]` as a generic alias, the `Set` import
  from `typing` is used consistently throughout the codebase for clarity in
  type annotations (see Style Guide §4).
- `from __future__ import annotations` defers evaluation, so `set[int]` would
  work at runtime, but the `typing.Set` form is easier for contributors
  accustomed to pre-3.9 codebases.

**Risk**: None — functionally identical.

---

## 13. Distributed PC brain ownership in src/app

**What**: The distributed host application was moved from
`scripts/gimbal_brain_pc.py` to `src/app/gimbal_brain_pc.py`.
`scripts/gimbal_brain_pc.py` is now a thin compatibility wrapper that imports
the app module and calls `main()`.

**Why**:

- The PC brain is application assembly, not an ad hoc script. It owns the
  top-level loop that connects UDP frames, detector workers, tracking,
  target selection, UI, and gimbal output.
- `src/app` already exists as the application-level namespace and is included
  in the packaged modules in `pyproject.toml`.
- Keeping the full implementation in `scripts/` hid reusable logic from
  package-level tests and encouraged duplicate helpers.
- The PC brain already depends on `src` modules, so moving it under `src/app`
  makes the dependency direction explicit instead of relying on script-local
  `sys.path` setup inside the implementation.

**Existing src code to reuse**:

| Concern | Existing module | Current use |
| ------- | --------------- | ---------------------- |
| UDP video/IMU input | `src/adapters/udp_stream.py` | Already used by the PC brain. |
| YOLO inference | `src/detection/detector.py` | Already used. |
| Async/latest-frame inference | `src/pipeline/workers.py` | Already uses `AsyncDetector.submit_latest()` and `GpuPreprocessor`. |
| Re-ID embedding scheduling | `src/pipeline/workers.py` | Already uses `ReIDHelper` when enabled. |
| Tracking backend selection | `src/tracking/tracker_adapter.py` | Already uses C++ binding fallback and Python path for Re-ID. |
| Tracking composition | `src/services/tracking_service.py` | Now reused after widening the tracker type to a backend protocol. |
| Target click/lifecycle logic | `src/control/target_controller.py` | Reuse for point selection and error computation where semantics match. |
| Control math | `src/control/pid.py` | Prefer PID or a small control service over script-local P-control. |
| ESP32 ASCII protocol | `src/control/ascii_gimbal_controller.py` | Already used by the PC brain. |

**App-local pieces that may remain app-specific**:

- `BrainUi` / pygame operator panel currently lives in
  `src/app/gimbal_brain_pc.py`. Extract it to `src/ui/brain_viewer.py` only if
  it needs independent tests or reuse.
- The auto-target policy in the PC brain currently selects the highest scoring
  live track. If this behavior differs from `TargetController`, keep it as an
  explicit app policy rather than forcing it into the generic controller.

**Alternatives considered**:

- Keep `gimbal_brain_pc.py` in `scripts/`: simplest short-term path, but it
  leaves a large executable outside the layered architecture and repeats
  helpers already present in the package.
- Merge it into `scripts/run_pipeline.py`: rejected because the local-video
  pipeline and distributed UDP brain have different input, UI, and gimbal
  command semantics.
- Move every helper into shared packages immediately: too broad. Only extract
  helpers when they have a second caller or need focused tests.

**Risk**: Medium. Moving the file changed import context, so optional imports
(`pygame`, `torchreid`, serial) and launch commands need smoke tests. Runtime
behavior should remain unchanged because the wrapper calls the same `main()`
function now hosted by `src/app/gimbal_brain_pc.py`.

**Validation**:

```sh
python -m compileall src scripts
PYTHONPATH=src python -c "from app.gimbal_brain_pc import main; print(main.__name__)"
```

The second command verifies the app module imports without running the UDP/UI
loop. Runtime smoke still requires an Android sensor node or simulator sending
UDP frames.

---

## 14. Local BoT-SORT backend as default tracker

**What**: Added a local `BoTSort` tracker backend in `src/tracking/bot_sort.py`
and made `--tracker-backend botsort` the default for both the local pipeline
and distributed PC brain. ByteTrack remains available with
`--tracker-backend bytetrack`.

**Why**:

- BoT-SORT improves identity preservation by combining motion with optional
  appearance embeddings while keeping the existing track-dict contract.
- A local backend avoids adding BoxMOT as a dependency. BoxMOT is useful for
  reference, but its AGPL-3.0 license is not appropriate as a direct dependency
  for this project.
- `TrackingService` already accepts a tracker protocol, so swapping backends
  does not require UI/control-loop changes.

**Implementation notes**:

- First version implements XYWH constant-velocity Kalman prediction,
  high/low-confidence association, optional OSNet Re-ID appearance matching,
  and lost-track buffering.
- Camera-motion compensation is intentionally deferred.
- Re-ID remains opt-in through `--enable-reid` to avoid surprise torchreid/GPU
  startup cost.

**Validation**:

```sh
.venv/bin/python -m unittest tests.test_bot_sort tests.test_tracker_adapter
.venv/bin/python scripts/run_pipeline.py --help
.venv/bin/python scripts/gimbal_brain_pc.py --help
```
