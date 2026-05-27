# AIMBOT — Tracking Improvements Report

> **Date**: 2026-05-24  
> **Scope**: BoT-SORT migration, OSNet model tuning, and Re-ID recovery after occlusion / temporary out-of-frame loss.

---

## 1. Purpose

The original tracking path was usable for simple in-frame motion, but it was
fragile under real gimbal operation:

- partial occlusion could fragment the target ID,
- target pose changes could reduce OSNet similarity,
- short out-of-frame exits almost always produced a new ID when the target
  re-entered,
- tracker tuning was spread across motion thresholds, Re-ID thresholds, and
  application-level target selection.

This report records the changes made to improve ID stability while keeping the
existing track dictionary contract used by the PC brain and local pipeline.

---

## 2. Baseline: Original ByteTrack

### 2.1 Comparison Target

The comparison target for this work is the original ByteTrack tracking path that
existed before BoT-SORT became the default backend.

In this report, "original ByteTrack" means the ByteTrack-style online tracker
used as the project baseline: Kalman motion prediction plus IoU association,
with new tracks created from high-confidence detections. It is a strong and
simple tracker when detections are frequent and the target remains spatially
near its predicted location.

The weakness is that original ByteTrack is fundamentally motion / IoU driven.
It does not natively solve identity recovery when the object disappears, changes
viewpoint, or re-enters at a very different screen location. That limitation is
especially visible in AIMBOT because the camera is mounted on a moving gimbal:
camera motion and target motion are coupled, so the same physical target can
jump across the image even when the object itself is not moving much.

### 2.2 Observed Problems

The most important field observations were:

| Situation | Baseline behavior |
| --- | --- |
| Partial occlusion | ID often stayed stable only when IoU remained high. |
| Larger pose change | ID could switch because the single stored feature did not represent multiple views. |
| Target briefly leaves frame | Re-entering target frequently received a new ID. |
| High `track-max-age` | Old tracks stayed alive too long and could compete with fresh detections. |

The short out-of-frame case was especially important for the PC brain because a
real gimbal can overshoot or briefly lose sight of the target. If the ID changes
on re-entry, the UI lock and control loop can appear to "give up" on the target.

### 2.3 ByteTrack vs. BoT-SORT Goal

The goal was not to replace ByteTrack because it was unusable. ByteTrack remains
valuable as a lightweight fallback. The goal was to add a default tracker that
handles the failure modes ByteTrack is not designed to solve.

| Capability | Original ByteTrack | Local BoT-SORT-ReID |
| --- | --- | --- |
| Main association signal | IoU / motion | IoU / motion plus optional appearance |
| Short occlusion | Good if predicted box still overlaps | Better when OSNet embedding is available |
| Out-of-frame re-entry | Often creates a new ID | Can recover lost IDs by appearance within `max_age` |
| Viewpoint change | Limited by spatial continuity | Improved by larger OSNet and appearance matching |
| Runtime cost | Lower | Higher when Re-ID is enabled |
| Best role | Fast fallback / simple scenes | Default PC-brain tracker for unstable camera motion |

---

## 3. BoT-SORT Integration

### 3.1 Implementation Choice

BoT-SORT was added as a local implementation in `src/tracking/bot_sort.py`.
BoxMOT was intentionally not added as a dependency because BoxMOT is AGPL-3.0
and would bring a large dependency surface. The implementation follows the
project-facing subset needed by AIMBOT:

- XYWH 8D constant-velocity Kalman state:
  `cx, cy, w, h, vx, vy, vw, vh`
- high / low confidence detection split,
- first association against high-confidence detections,
- second association against low-confidence detections,
- optional OSNet appearance matching,
- same output contract as ByteTrack.

The output track dictionary remains:

```python
{
    "track_id": int,
    "bbox": np.ndarray,
    "score": float,
    "class_id": int,
    "is_confirmed": bool,
    "feature": Optional[np.ndarray],
    "time_since_update": int,
}
```

This keeps `TrackingService`, PC brain target selection, and UI/control code
stable.

### 3.2 Backend Selection

Tracker creation now goes through `src/tracking/tracker_adapter.py`:

- `botsort` is the default backend,
- `bytetrack` remains available as a fallback,
- `--tracker-module` and `--require-cpp-tracker` are ByteTrack-only concerns,
- the PC brain enables Re-ID by default for real-gimbal operation; use
  `--disable-reid` for no-ReID comparison runs.

The two application entry points use the same tracker factory:

- `src/app/gimbal_brain_pc.py`
- `scripts/run_pipeline.py`

---

## 4. Re-ID Model Experiment

### 4.1 Model Change

The initial Re-ID command used:

```bash
--reid-model osnet_x0_25
```

This model is fast, but its features are weaker under pose and viewpoint
changes. In real operation, switching to:

```bash
--reid-model osnet_x0_5
```

improved partial occlusion behavior. The tradeoff is higher inference cost for
the Re-ID crop batch.

### 4.2 Candidate Count

`ReIDHelper` does not embed every detection by default. It embeds:

- the top-K detections by confidence,
- plus detections overlapping the currently locked target bbox.

For occlusion and re-entry, the target may not be the highest-confidence
detection. Raising candidates improves the chance that the returning object has
an embedding available for BoT-SORT association:

```bash
--reid-candidates 8
```

For busier scenes or frequent off-screen re-entry, this can be raised to:

```bash
--reid-candidates 12
```

### 4.3 Practical Result

With `osnet_x0_5`, `reid-candidates=8`, and tuned BoT-SORT thresholds, partial
occlusion became noticeably more stable. However, the target still often lost
its ID after briefly leaving the frame. That led to the recovery fix described
below.

---

## 5. Out-of-Frame Recovery Fix

### 5.1 Root Cause

BoT-SORT's appearance matching was still gated by spatial proximity. In code
terms, a detection feature could only help if the IoU distance also passed the
`proximity_thresh` gate.

That is reasonable for active in-frame tracks because it prevents a live track
from teleporting across the frame. It is harmful for lost tracks:

1. Target leaves the frame.
2. Track becomes `lost` but remains in memory until `max_age`.
3. Target re-enters from another side of the image.
4. The new detection has almost zero IoU with the predicted lost box.
5. Appearance matching is skipped because the proximity gate fails.
6. A new track ID is created.

The important point: the Re-ID feature could be correct, but it never got a
chance to recover the ID.

### 5.2 Code Change

The association logic now treats `lost` tracks differently:

- active `tracked` tracks still require proximity before appearance can override
  IoU,
- `lost` tracks may use appearance matching without the IoU proximity gate,
  as long as the appearance distance passes `appearance_thresh`.

In practical terms:

```text
tracked track:
  require reasonable spatial proximity + appearance match

lost track:
  allow appearance-only recovery within max_age
```

This directly targets the "briefly leaves the frame and comes back" case without
allowing every active track to jump around the image.

### 5.3 Validation

A unit test was added for the recovery case:

1. Create a confirmed track with an embedding.
2. Update with no detections so the track becomes lost.
3. Re-enter with the same embedding but at a box location with no useful IoU.
4. Assert that the original `track_id` is recovered.

Validation command:

```bash
.venv/bin/python -m unittest tests.test_byte_tracker tests.test_tracker_adapter tests.test_tracking_service tests.test_bot_sort
```

Result:

```text
Ran 15 tests
OK
```

---

## 6. Current PC-Brain Defaults

The following BoT-SORT + OSNet Re-ID settings are now the PC-brain defaults:

```bash
python scripts/gimbal_brain_pc.py
```

The default values are equivalent to:

| Argument | Default |
| --- | --- |
| `--enable-reid` | enabled |
| `--reid-model` | `osnet_x0_5` |
| `--reid-candidates` | `8` |
| `--reid-memory-frames` | `120` |
| `--track-max-age` | `120` |
| `--track-min-hits` | `2` |
| `--track-thresh` | `0.35` |
| `--new-track-thresh` | `0.55` |
| `--botsort-proximity-thresh` | `0.95` |
| `--botsort-appearance-thresh` | `0.18` |
| `--botsort-match-thresh` | `0.8` |

For no-ReID comparison runs:

```bash
python scripts/gimbal_brain_pc.py --disable-reid
```

If the target leaves the frame for longer, try:

```bash
--track-max-age 180 --reid-candidates 12
```

Avoid starting with `--track-max-age 300`. Very long track memory can leave stale
tracks alive and increase the chance of wrong recovery in scenes with similar
objects.

---

## 7. Parameter Notes

### `--track-thresh`

This is the high-confidence threshold for BoT-SORT's first association stage.
Lowering it helps partially occluded targets remain eligible for Re-ID matching:

```bash
--track-thresh 0.35
```

### `--new-track-thresh`

This controls when unmatched detections become new tracks. Keeping it higher
than `track-thresh` reduces duplicate track creation:

```bash
--new-track-thresh 0.55
```

### `--botsort-proximity-thresh`

This is an IoU-distance gate before appearance matching for active tracks. A
larger value allows more spatial movement before Re-ID is blocked:

```bash
--botsort-proximity-thresh 0.95
```

For lost tracks, the recovery fix bypasses this gate and relies on appearance
distance instead.

### `--botsort-appearance-thresh`

This is the maximum appearance distance:

```text
appearance distance = (1 - cosine_similarity) / 2
```

For example:

| Appearance threshold | Approx. minimum cosine similarity |
| --- | --- |
| `0.18` | `0.64` |
| `0.25` | `0.50` |

`0.18` is stricter and safer in multi-object scenes. `0.25` can recover more
aggressively but may increase wrong matches when objects look similar.

### `--reid-similarity` and `--reid-distance`

These arguments are still useful for older / fallback tracking paths and target
logic, but the BoT-SORT association path is primarily controlled by:

- `--botsort-appearance-thresh`,
- `--botsort-proximity-thresh`,
- `--reid-candidates`,
- `--track-max-age`.

Do not expect `--reid-similarity` alone to fix BoT-SORT ID recovery.

---

## 8. Remaining Limitation: Single Feature Memory

The current BoT-SORT track stores one feature vector per track and updates it
with exponential moving average. This is simple and fast, but it is not ideal
for large viewpoint changes.

Example:

```text
front view feature + side view feature + back view feature
```

These can be genuinely different even for the same object. Averaging them into
one vector can blur the identity representation.

The next major improvement should be a feature gallery:

- store multiple feature vectors per track,
- match using the minimum distance to any gallery feature,
- only add high-quality detections to the gallery,
- reject tiny / heavily occluded / low-confidence crops,
- cap gallery size, for example 8 to 16 features.

The matching rule would become:

```text
track_distance = min(distance(det_feature, gallery_feature_i))
```

This should improve different-angle recognition more than further threshold
tuning.

---

## 9. Deferred Work

### Camera Motion Compensation

BoT-SORT commonly pairs well with camera motion compensation. AIMBOT does not
include CMC yet. For the gimbal use case, CMC could help when the camera itself
rotates quickly, but it should be evaluated carefully because the PC brain also
receives IMU data from the Android node.

Possible future options:

- image-based global motion estimation,
- IMU-assisted compensation,
- a documented `--botsort-cmc-method none|...` flag once implemented.

### Re-ID Quality Filtering

The tracker should avoid updating identity memory from bad crops:

- partial crops at frame edges,
- very small boxes,
- obvious occlusion,
- low detection confidence,
- extreme aspect ratios.

This matters because bad features can poison the stored identity and make later
recovery worse.

---

## 10. References

- BoT-SORT paper: https://arxiv.org/abs/2206.14651
- Official BoT-SORT repository: https://github.com/NirAharon/BoT-SORT
- BoxMOT BoT-SORT documentation, used only for API / behavior comparison:
  https://mikel-brostrom.github.io/boxmot/trackers/botsort/
