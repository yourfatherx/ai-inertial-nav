# D4 — No-magnetometer operational assumption

*Deployment track. Doc-only (this is already the design, verified in code).
Updated 2026-07-14.*

## The assumption

**The system uses no magnetometer.** Heading is held by the (denoised) gyro alone,
with accelerometer-as-gravity pinning roll/pitch. A magnetometer, if present, is
assumed **jammed or unreliable** — the realistic condition near brushless motors
(large switching currents) and in the same electronic-warfare environments that
deny GPS.

This is not a limitation we accept reluctantly; it is a **design choice that makes
the operational assumption jam-proof**. The threat model is "GPS denied by an
adversary." An adversary who can deny GPS can also spoof or swamp a magnetometer.
A nav solution that leaned on the mag for heading would inherit a second jammable
input. Ours does not.

## Verified in code (not just asserted)

A grep of `src/` and `scripts/` for magnetometer/compass/external-heading usage
finds **zero** consumers — the only matches are documentation strings stating this
very assumption. The ESKF (`src/ainav/eskf.py`) fuses exactly two things:

- **predict:** gyro (angular rate) + accel (specific force) integration.
- **update:** `update_gravity()` — accel-as-gravity, gated to near-static samples,
  correcting **roll and pitch only**.

There is no yaw/heading measurement update of any kind.

## The consequence — and why the AI matters here

With IMU only, **yaw about the gravity vector is unobservable** (the gravity tilt
update pins roll/pitch but carries no heading information). The code says so
directly (`eskf.py`: *"Yaw about gravity stays UNOBSERVABLE with IMU only"*).

That is precisely the axis a magnetometer would normally observe. Remove the mag
and any gyro **yaw-bias integrates straight into unbounded heading drift**, which
then smears horizontal position — this is the dominant error in GPS-denied INS.

**This is the exact error the gyro denoiser removes.** The learned network
subtracts the systematic gyro bias (most visibly the ~4.3 °/s EuRoC yaw bias)
*before* the filter integrates it, keeping heading bounded without ever needing a
magnetometer. So the no-mag assumption and the AI contribution are the same story:
the denoiser is what buys back the heading stability a mag would otherwise provide,
using only jam-proof inertial data.

- Quantified in the Phase-4 "money plot" (`results/phase4_*.png`): denoised vs raw
  ESKF, both `bg0=0`, on held-out sequences.
- Quantified over GPS-denial time in **D3** (`results/d3_drift_curve.png`):
  AI-Nav stays ~3× tighter than conventional INS at 60 s of denial, mag-free.

## Statement for the pitch / spec sheet

> **Heading source:** inertial only (denoised gyro). **No magnetometer dependency.**
> Operates under magnetic jamming and near high-current motors. The learned gyro
> correction substitutes for the heading reference a magnetometer would provide,
> using only inputs an adversary cannot spoof.

## Status

**Done (doc-only).** The assumption is the implemented design, verified against the
code, and quantified by Phase-4 and D3. No further work.
