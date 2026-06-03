EEG vs EEG+OpenCV — Issues, Findings, and Next Steps
=====================================================

Summary
-------
This document summarizes the issues discovered while developing and testing the Kinova controllers:
- `kinova_eeg_controller.py` (EEG-only)
- `kinova_eeg_opencv_controller.py` (EEG + OpenCV hybrid)

It lists observed differences, root causes, fixes applied, and recommended next steps.

Key Files
---------
- EEG-only controller: [EEG_Kinova_Project/kinova_eeg_controller.py](EEG_Kinova_Project/kinova_eeg_controller.py)
- Hybrid controller: [EEG_Kinova_Project/kinova_eeg_opencv_controller.py](EEG_Kinova_Project/kinova_eeg_opencv_controller.py)
- Predictor: [EEG_Kinova_Project/real_time_eeg_predictor.py](EEG_Kinova_Project/real_time_eeg_predictor.py)
- Hardware settings: [EEG_Kinova_Project/hardware_config.py](EEG_Kinova_Project/hardware_config.py)
- Example logs: [EEG_Kinova_Project/logs/kinova_eeg_opencv_controller_20260520_152444.log](logs/kinova_eeg_opencv_controller_20260520_152444.log)

Observed Behavior (EEG-only)
----------------------------
- Behaviour: EEG predictions trigger non-blocking `go_to_angles` moves to `RIGHT`/`LEFT` positions.
- Gating: Only start a move when previous move finished and the target changed (`last_target_angles` logic).
- Timeouts: position timeouts are fixed (e.g., `timeout=15` in EEG-only code).
- Result: Moves report success (`ok=True`) and measured Cartesian X deltas are printed in status logs; motion is deterministic and position-driven.

Observed Behavior (EEG + OpenCV)
--------------------------------
- Behaviour: EEG still decides direction; OpenCV originally modulated speed via an `OPENCV_WEIGHT` blending that reduced effective speed when OpenCV magnitude was small.
- Position moves: hybrid code launched non-blocking moves similar to EEG-only, but additionally suspended `send_vx` while position move in progress (`MOVEMENT_IN_PROGRESS`).
- Timeouts: earlier versions used match/mismatch timeouts; recent changes compute move `timeout` from `abs(opencv_vx)` (higher OpenCV speed → shorter timeout).
- Issues observed:
  - BASE_VX regressions: at times `BASE_VX` was set incorrectly (e.g., mistakenly 8.0 or 0.0), causing either absurd speed or effectively zero velocity.
  - Small end-effector delta: `go_to_angles` completes but measured Cartesian X change (`dx`) is typically < 0.01 m, often ~0.001–0.007 m — visually negligible.
  - Velocity suppression: because velocity commands are suspended during `go_to_angles`, the robot appears unresponsive while moves run.
  - Rapid re-triggering: without proper gating, the system could retrigger similar moves frequently, masking velocity behavior.

Root Causes Identified
----------------------
- Position target selection: hardcoded `RIGHT_ANGLES`/`LEFT_ANGLES` produce small Cartesian X displacement for the current arm configuration.
- Blending logic: OpenCV magnitude originally reduced effective speed (OPENCV_WEIGHT), which amplified small-motion effects; removing blend simplified behavior but exposed target-size issue.
- Configuration regressions: accidental edits set `BASE_VX` to wrong values; lack of safety clamps allowed problematic values to be used at runtime.
- Interaction semantics: position moves block velocity commands; frequent short position moves reduce the time the system spends in velocity-driven behavior.

Fixes Implemented
-----------------
- Restored `BASE_VX` to a sane default (`0.177`) and added clamping at use site to ensure `0.01 <= BASE_VX <= V_MAX`.
- Added `MOVE_COOLDOWN_S` and `last_move`/`last_target_angles` logic to avoid rapid duplicate moves (gating now mirrors EEG-only controller).
- Added post-move verification: logged `x` before/after each `go_to_angles`; if `dx` < `MIN_MOVE_DELTA_M` (0.01 m), retry the move once.
- Simplified blending: removed `OPENCV_WEIGHT` and made OpenCV influence the move `timeout` (faster OpenCV ⇒ smaller timeout to reach position sooner).
- Added richer logging: `base_speed`, `opencv_vx`, `opencv_speed`, `move` label, `timeout`, and `dx` after moves to aid diagnosis.

Evidence (logs)
---------------
- Example: [logs/kinova_eeg_opencv_controller_20260520_152444.log](logs/kinova_eeg_opencv_controller_20260520_152444.log)
  - Shows `[PRED] ... base_speed=0.1770 opencv_vx=-0.1011 ... move=opencv(0.101) timeout=36.5s`.
  - Shows repeated `[MOVE] done ... dx=-0.0052` and retry attempts where `dx` remains < 0.01 m.

Recommendations / Next Steps
---------------------------
1. Diagnostic logging (short term): log joint angles before/after a position move to confirm which joints changed and by how much. This will show why Cartesian X is small.
2. Larger position targets (quick fix): adjust `RIGHT_ANGLES`/`LEFT_ANGLES` to increase the joint delta so the resulting Cartesian X change is visibly larger. Prefer a small multiplicative scale factor and test incrementally.
3. Velocity-first mode (alternative): when you prefer continuous, visible movement, disable position moves and use `send_vx` solely (EEG chooses direction; OpenCV scales speed). This is simpler and more immediate to feel responsive.
4. Adaptive fallback (robust): if `dx` < threshold after move, automatically issue a larger secondary move or perform a timed velocity burst in the same direction to achieve visible displacement.
5. Safety & robustness: keep clamps on `BASE_VX`, and add max/min sanity checks on timeouts and move retries to avoid stuck loops.

Action Items I Can Implement
---------------------------
- Add joint-angle before/after logging (diagnostic) — low risk.
- Add automated larger-move fallback when `dx` small — medium risk (requires tuning).
- Replace position moves with velocity-driven control (toggle mode) — medium effort.
- Tune `RIGHT_ANGLES`/`LEFT_ANGLES` by applying a configurable multiplier and iterate with live tests — quick to try.

Status of Repo Changes
----------------------
Recent patches (branch: feature/shivansh) modified `kinova_eeg_opencv_controller.py` to:
- Restore `BASE_VX` and add clamp
- Add cooldown and last-target gating
- Add post-move `dx` verification and one retry
- Compute `timeout` from `opencv_vx`

If you'd like, I will now implement one of the recommended action items. Tell me which one and I'll proceed.

Why position-based moves (EEG-only style) are problematic with EEG + OpenCV
-----------------------------------------------------------------------
When we tried to reuse the exact EEG-only position-move approach together with OpenCV, several practical conflicts and failure modes appeared:

- Small Cartesian effect of joint targets: The hardcoded `RIGHT_ANGLES`/`LEFT_ANGLES` used for EEG-only produce only tiny end-effector X displacements for the current robot pose. `go_to_angles` reports success (joints reached) but the measured Cartesian `dx` is often < 0.01 m — visually negligible.

- Conflicting control semantics (position vs velocity): OpenCV provides a fast, frame-rate estimate of object motion that is naturally best applied to velocity commands (`send_vx`) for continuous correction. Position moves are discrete, long-running actions that lock out velocity updates while they execute; this prevents OpenCV from applying its corrective effect during moves.

- Timing mismatch and frequent retriggers: EEG predictions are periodic (e.g., 0.5–1s) and OpenCV updates at camera frame rate. Launching a position move for every EEG tick leads to frequent short or overlapping moves, cancelling the intended continuous behavior and producing the appearance of no motion.

- Blending & safety interactions: Prior blending logic (scaling a base speed by an `OPENCV_WEIGHT`) could inadvertently reduce effective speed when OpenCV magnitude is small, making motion even less visible. Mistakes in tuning or config regressions (e.g., `BASE_VX=0`) amplify this problem.

- Measurement noise and confidence gating: OpenCV and EEG probabilities are noisy; gating moves by softmax confidence can cause many aborted or repeated moves if thresholds or cooldowns are not tuned, further fragmenting motion.

Because of these factors, position-based moves ended up fighting the OpenCV-driven speed/flow and produced poor UX. For these reasons we switched to a velocity-driven paradigm for the hybrid controller: EEG decides direction, OpenCV controls the instantaneous commanded speed, and position moves are disabled (or relegated to a separate, explicitly-invoked mode).

If you prefer to keep both modes available, a safer hybrid design is:

- Add a runtime `--mode` switch (velocity vs position) so operators choose behavior explicitly.
- If using position mode with OpenCV, compute larger target offsets (scale joint deltas) and only trigger moves after robust, debounced class changes.
- Use post-move pose verification and a velocity fallback (short `send_vx` burst) when `dx` is below threshold.

These mitigations preserve the precision of position moves while allowing OpenCV to give immediate, visible feedback via velocity when needed.
