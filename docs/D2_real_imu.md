# D2 — Real-IMU transfer (sensor-agnostic claim)

*Deployment track. Updated 2026-07-14. All numbers reproduced from results in-repo.*

**Question a buyer asks:** "Your denoiser was trained on the EuRoC IMU. Our drone
flies a *different*, cheaper, noisier MEMS. Does it still work?"

**Answer (honest, two parts):** Not zero-shot — a correction learned on one IMU's
systematic bias does not transfer to a different sensor's bias. **But a short
per-IMU fine-tune recovers the win.** The negative is fixable, not fatal, and the
fine-tune is cheap (warm-started, minutes on the RTX 4050).

---

## The two IMUs

| | Training IMU | Transfer IMU |
|---|---|---|
| Dataset | EuRoC | TUM-VI (room1/2/3) |
| Sensor | ADIS16448 (tactical-grade MEMS) | Bosch BMI160-class (consumer MEMS) |
| Dominant learned error | constant gyro z-bias ≈ **−4.5 °/s** | *different* bias structure |

The denoiser learned to subtract EuRoC's specific bias. On TUM-VI it applies a
correction of `[-0.016, -1.984, -4.799] °/s` — i.e. it *still tries to remove the
EuRoC z-bias*, which is the wrong correction for this sensor.

---

## Result 1 — Zero-shot transfer fails (reported straight)

Frozen EuRoC net, no adaptation, run through the same ESKF on held-out
`tumvi_room1` (`scripts/cross_imu_eval.py`, `results/cross_imu_room1.png`):

| config (both bg0=0, realistic) | orient RMSE | yaw RMSE | ATE RMSE |
|---|---|---|---|
| raw ESKF (no AI) | 35.6° | 35.5° | 335.9 m |
| **denoised ESKF (frozen EuRoC net)** | **116.0°** | **116.0°** | **1478.9 m** |

**Verdict:** −226 % orient, −340 % ATE. The denoiser makes it **worse** on a sensor
it wasn't trained for. Same picture at the open-loop attitude level: raw 0.858° vs
frozen EuRoC net 1.624° on room1.

This is the *expected* outcome and we bank it as the finding: **the learned
correction is sensor-specific.** That is what motivates D2's fine-tune step.

## Result 2 — Per-IMU fine-tune recovers the win (held-out)

Warm-start from the EuRoC checkpoint, recompute TUM-VI normalization stats, train
on `room2 + room3`, **hold out `room1`** — the exact sequence the frozen net failed
on, so recovery is a true held-out result (`scripts/finetune_tumvi.py`,
`results/finetune_tumvi.history.json`):

| config on held-out room1 | open-loop attitude RMSE |
|---|---|
| raw gyro (no AI) | 0.858° |
| frozen EuRoC net (zero-shot) | 1.624°  ← worse than raw |
| **fine-tuned net** | **0.808°**  ← beats raw |

40 epochs, warm-started (not from scratch). The fine-tune converts a −89 %
regression into a positive result on the new sensor.

---

## What this supports in the pitch

- **Claim:** "Sensor-agnostic **with a short per-IMU calibration fine-tune**." Not
  "works zero-shot on any IMU" — we don't claim that, and the data says we
  shouldn't.
- **Operational story:** each airframe/IMU gets a one-time fine-tune from its own
  logged flight data (warm-started from the base model). This is a standard,
  cheap onboarding step, not a retrain-from-scratch.
- **Why it's credible:** the transfer failure is reported at full strength, and the
  fix is demonstrated on a genuinely held-out sequence from a physically different
  sensor.

## Reproduce

```
python scripts/cross_imu_eval.py --seq room1      # -> results/cross_imu_room1.png (zero-shot fails)
python scripts/finetune_tumvi.py                  # -> results/finetune_tumvi.history.json (fine-tune recovers)
```

## Status

**Done.** Both the negative (zero-shot) and the fix (fine-tune) are computed and
in-repo. Remaining true-deployment step is D1 (run on the target embedded IMU +
compute), which needs the hardware.
