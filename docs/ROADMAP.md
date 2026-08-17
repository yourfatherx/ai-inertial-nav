# AI-Inertial-Nav — Roadmap

*AI-based inertial navigation for GPS-denied UAV flight. Public data only (EuRoC). Updated 2026-07-14.*

Legend: ✅ done · 🟩 alive/proposed · ❌ closed · ⚠️ re-scoped

---

## Where the project stands (one paragraph)

The working prototype is **Approach B + A1 + an investor demo**. B is a learned gyro-denoiser feeding an ESKF (−98 % attitude drift, generalizes across seeds and environments). A1 is OpenVINS stereo-VIO adding a camera to bound absolute position to ~5 cm. The demo replays real EuRoC flight showing conventional inertial nav drifting to hundreds of metres after GPS loss while our stack stays locked on truth. The learned-VIO research thread (A2/A3) is **closed after two honest negatives**. **The deployment track is now done except for the one item that needs hardware: D2 (real-IMU transfer), D3 (GPS-denied drift curve), and D4 (no-mag) are all complete; D1 (edge/realtime on a Jetson-class module) is the only remaining task and is blocked on the board.** There is no further work doable on the current machine — everything left requires the embedded target.

---

## Approach B — learned inertial (COMPLETE ✅)

| Phase | What | Status |
|---|---|---|
| Phase 0 | Env + EuRoC loader + raw-vs-truth plot | ✅ |
| Phase 1 | Naive open-loop integration baseline (shows drift) | ✅ |
| Phase 2 | Plain ESKF, raw IMU, no AI (the honest baseline) | ✅ |
| Phase 3 | Gyro-denoising CNN, open-loop attitude eval | ✅ |
| Phase 4 | Fuse denoised gyro into ESKF — the "money plot" (AI+filter beats filter-alone) | ✅ |
| Phase 5 | Demo polish | ✅ (see Demo track) |

**Result:** −98 % attitude drift vs raw; holds across seeds and across environments (MH↔V), cross-IMU (TUM-VI), and after fine-tune. 19k parameters.

---

## Approach A — visual-inertial (partially closed)

Adds a camera to bound *absolute position*, which gyro-only B cannot. Strategy was: wrap a proven open VIO back-end (OpenVINS/MSCKF) as the classical baseline, put the research into a **learned front-end**.

| Phase | What | Status |
|---|---|---|
| A0 | Load cam0/cam1, verify image↔IMU sync, undistort | ✅ |
| A1 | OpenVINS stereo-VIO baseline (WSL2 + Docker, headless) | ✅ **~5 cm ATE on V1_01** |
| A2 | Swap in learned front-end (SuperPoint+LightGlue), same back-end | ❌ **CLOSED — see below** |
| A3 | Fuse learned visual factor + gyro denoiser into one AI-VIO pipeline | ❌ **OFF THE TABLE** (foundation failed) |
| A4 | Stress: dropped frames, blur, longer GPS-denied runs | ⚠️ **re-scoped → D3** (see below) |

### A2 — closed permanently (two honest negatives)
- **Attempt 1 (banked):** injected both front-ends' tracks into the same mono MSCKF via `feed_measurement_simulation`. 3 of 4 runs diverged; only `Learned_clean` survived (65 cm). Root cause: the SIM feed path bypasses OpenVINS' native RANSAC/outlier rejection; KLT's long high-leverage tracks amplified residual outliers → divergence.
- **Attempt 2 (RANSAC gate, this session):** added a tracker-agnostic 2-view RANSAC gate (fair — identical rule on both front-ends), re-ran 4×. **All 4 still diverged, and the gate broke the one survivor** — `Learned_clean` regressed 65 cm → 389,303 cm. Culling outlier observations shortened the already-borderline learned tracks (meanLen 4.3→3.7) below the baseline the MSCKF needs to triangulate. The gate removed outliers *and* the constraints the filter depended on.
- **Verdict:** mono external-track injection is the wrong tool; not a tuning knob from working. **Stopped per the agreed timebox — no compromise-spiral.**
- **What survives:** the learned front-end's tracks are **3.7× more geometrically accurate under low-light** (reprojection pre-flight). A real **component-level** result — not a VIO win. A proper learned VIO would need a descriptor track-manager, which is out of scope.

### A4 — re-scoped, not run
A4 implicitly stress-tested the *full AI-VIO pipeline* (A3), which was never built. Its pieces resolve as: dropped-frames/blur-on-VIO → **dead** (no pipeline to stress); front-end degradation robustness → **already answered** by the low-light pre-flight; **longer GPS-denied runs → survives as D3** (the roadmap always noted "ties to D3"). No standalone A4 remains.

---

## Deployment track (D1–D4) — the live roadmap 🟩

Real drone deployment needs a validation track beyond the algorithm phases. This is what investors and defence buyers actually probe.

| Item | What | Status | Backs the pitch |
|---|---|---|---|
| **D1** | **Edge/realtime:** benchmark denoiser+ESKF at realtime on embedded (Jetson / flight-controller class), per-IMU-sample latency budget, INT8/quantized. Must run onboard, not the RTX 4050. | 🟩 **proposed — the only item left; needs hardware** | "runs on a small compute module" — currently unproven |
| **D3** | **Long GPS-denied drift curve:** drift-vs-denial-time, swept cut points, IMU-only after cut. Absorbs A4. | ✅ **DONE** — `results/d3_drift_curve.{png,json}`. **AI-Nav 3× tighter than INS at 60 s** (781 m → 288 m); +63–92 % across horizons. | drift-vs-time = a quotable number + a demo slide |
| D2 | Real-IMU transfer to drone-grade MEMS; per-IMU fine-tune | ✅ **DONE** — `docs/D2_real_imu.md`. Zero-shot cross-IMU **fails** (honest negative, −340 % ATE on TUM-VI); short per-IMU fine-tune **recovers** (0.808° held-out). | sensor-agnostic *with fine-tune* |
| D4 | No-magnetometer confirmation (mag jammed near motors) | ✅ **DONE (doc)** — `docs/D4_no_mag.md`. Verified in code: ESKF fuses gyro + accel-tilt only, **no mag anywhere**. | jam-proof operational assumption |

---

## Demo / pitch track ✅🟩

| Artifact | Status |
|---|---|
| `results/demo_replay.html` — Tier 1 animated 3D replay, Play + scrubber + GPS-DENIED banner | ✅ |
| `results/demo_interactive.html` — Tier 2 GPS-cut slider, red INS re-diverges live, offline | ✅ |
| `docs/DEMO_pitch.md` — narration + market one-liner | ✅ |
| Attitude-isolation panel (isolate the AI's −98 % contribution) | 🟩 optional |
| Second demo on an MH sequence (different scene) | 🟩 optional |
| One-page deck / recorded flythrough | 🟩 optional |

---

## Proposed next steps

**Only one item remains, and it needs hardware: D1 — edge/realtime benchmark.** Run
the denoiser+ESKF onboard a Jetson / flight-controller-class module at realtime,
quantized (INT8), within the per-IMU-sample latency budget. This proves the "runs on
a small module" claim on the real board. Blocked until the Jetson is in hand (a CPU
+ quantization *proxy* is possible on the laptop but is a proxy, not the claim).

Everything else is complete and reproduced in-repo:
- **D2** (real-IMU transfer) — `docs/D2_real_imu.md`
- **D3** (GPS-denied drift curve) — `scripts/d3_drift_curve.py`, `results/d3_drift_curve.{png,json}`
- **D4** (no-magnetometer) — `docs/D4_no_mag.md`
- **One-page brief** — `docs/ONEPAGER.md`

**Not on the roadmap anymore:** any further VIO front-end work (A2/A3/A4-as-written). Settled.
Optional-only: a 2nd-scene demo (needs a fresh OpenVINS run in the WSL2 container).

---

## Decisions log (load-bearing)

- **2026-07-12** — Approach A scope locked: wrap OpenVINS (MSCKF), research the learned front-end. Backend OpenVINS over VINS-Fusion (clean front-end/back-end split). Windows can't build ROS VIO → WSL2 + Docker, headless, CPU-only.
- **2026-07-13** — A1 done: OpenVINS stereo ~5 cm on V1_01. OpenVINS ships build-local Dockerfiles (no prebuilt image); ROS1 Noetic / Ubuntu 20.04.
- **2026-07-14** — A2 banked (attempt 1), then RANSAC-gate retry (attempt 2) **failed and regressed the survivor** → **A2 closed permanently, A3 off the table, A4 re-scoped into D3.** Two honest negatives; timebox respected. Prototype = B + A1 + demo.
- **2026-07-14** — **D2/D3/D4 completed in one pass.** D3 drift curve built (`scripts/d3_drift_curve.py`): AI-Nav 3× tighter than INS at 60 s GPS denial. **Bug caught & fixed during D3:** initial version seeded the filter with GT gyro bias (`bg0=GT`), which double-corrects the already-debiased denoised run and *inverted* the result (AI looked worse); the honest GPS-denied baseline is `bg0=0` for both sides (phase-4 convention) — nobody hands the filter the true bias. D2 assembled from existing cross-IMU + fine-tune runs (honest zero-shot-fails / fine-tune-recovers story). D4 verified in code (no magnetometer consumer anywhere). One-pager written. **Only D1 (needs Jetson) remains.**
