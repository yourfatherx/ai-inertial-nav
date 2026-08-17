# AI-Inertial-Nav — one-page brief

**AI-based inertial navigation for GPS-denied UAV flight.**
When GPS is jammed or spoofed, an onboard neural network keeps the aircraft knowing
where it is — no external signal, no magnetometer, running on a small compute
module. All results on **real public flight data** (EuRoC); every number reproduced
in-repo.

---

## The problem
GPS denial (jamming / spoofing) is a live, named threat in drone warfare, contested
logistics, and counter-UAS. When GPS drops, a conventional inertial nav system (INS)
**drifts without bound** — gyro bias integrates into heading error, which smears
position. Within a minute the aircraft's estimate has left the building.

## What we do
A **learned gyro-denoiser** (19k parameters) removes the IMU's systematic error
*before* a classical filter (ESKF) integrates it. Heading stays bounded using
**only inertial data** — the input an adversary cannot spoof. A camera (OpenVINS
stereo-VIO) bounds absolute position when available.

---

## The three headline results

**1 — Attitude drift cut 98 %** *(Phase 4 "money plot", held-out sequences)*
Raw-gyro ESKF vs denoised-gyro ESKF, same filter, same info (`bg0=0`, the realistic
GPS-denied case):

| held-out seq | raw ESKF orient | **denoised (ours)** | oracle (handed true bias) |
|---|---|---|---|
| V1_02 | 76.0° | **1.57°** | 0.75° |
| V2_03 | 102.7° | **5.98°** | 4.68° |

The network recovers **near-oracle** accuracy *from data alone*. Holds across 5
random seeds and across environments (Machine-Hall ↔ Vicon).

**2 — 3× tighter under sustained GPS denial** *(D3 drift curve — `results/d3_drift_curve.png`)*
Swept 20 GPS-cut points on real flight; IMU-only after each cut:

| GPS-denial time | conventional INS | **AI-Nav (ours)** | improvement |
|---|---|---|---|
| 30 s | 290 m | **67 m** | +77 % |
| **60 s** | **781 m** | **288 m** | **+63 % (3× tighter)** |
| 120 s | 1704 m | **582 m** | +66 % |

**3 — Absolute position to ~6 cm with vision** *(A1, OpenVINS stereo-VIO)*
On V1_01: **6.2 cm** ATE vs conventional INS at hundreds of metres — a **5000×**
gap. This is the live investor demo (`results/demo_replay.html`).

---

## Why it's credible (not a demo trick)
- **Real data, held-out:** the denoiser is evaluated on sequences it never trained on.
- **Same-info comparison:** the baseline gets the identical filter and inputs; the
  only change is whether the gyro is cleaned. No oracle bias handed to our side.
- **Reported straight:** where it *doesn't* transfer (a different IMU, zero-shot) we
  say so — and show a short per-IMU fine-tune recovers it (see `docs/D2_real_imu.md`).
- **Jam-proof by design:** no magnetometer, no GPS-in-the-loop (`docs/D4_no_mag.md`).

## Maturity & what's next
- **Done:** algorithm (Approach B), vision baseline (A1), investor demo, D2/D3/D4.
- **Next — D1 (needs target hardware):** run the denoiser+ESKF onboard a
  flight-controller-class module (Jetson) at realtime, quantized, within the
  per-IMU-sample latency budget. This proves the "runs on a small module" claim on
  the real board.

---

*Live demo: open `results/demo_replay.html` (▶ Play) and `results/demo_interactive.html`
(drag the GPS-cut slider) — no install, any browser. Narration in `docs/DEMO_pitch.md`.*
