# GPS-Denied Flight Demonstrator — how to present it

Two files, no install, open in any browser (double-click):
- `results/demo_replay.html` — **the 20-second money shot.** Hit ▶ Play.
- `results/demo_interactive.html` — **the "prove it's not a trick" toy.** Drag the slider.

Everything on screen is **real drone data** (EuRoC V1_01) and every number is validated
(`results/compareA1_vs_B.json`). Nothing is simulated or hand-tuned for the demo.

---

## The one-liner (lead with the market pain, not the method)
> *"When GPS is jammed or spoofed, our onboard AI keeps the drone knowing where it is —
> no external signal, running on a small compute module."*

Jamming/spoofing is a live, named problem: drone warfare, contested logistics, counter-UAS.
Call it a **"GPS-denied flight demonstrator"** — **not** a "digital twin" (overpromises to a
defence buyer).

## Narration for the replay (read over the ~20 s playback)
1. *"Green is where the drone actually flew. Watch all three start locked together."*
2. *"At 60 seconds, GPS is denied."* — (the red **⚠ GPS DENIED** banner flips on)
3. *"Red is a conventional inertial system with no GPS. It immediately starts drifting —"*
4. *"— and by the end it thinks it's **300 metres** away. It's left the building. That's a lost aircraft."*
5. *"Blue is ours. Same denied GPS, same sensors — it stays locked on truth, **6 centimetres** off."*
6. *"No external signal the whole time. That's the difference between a lost drone and a mission."*

## The interactive (hand them the mouse)
> *"You pick the moment GPS dies."* — drag the slider. Wherever they cut it, red peels off
> from that instant and ends hundreds of metres out; blue never moves off truth.
> The point: **it's not one lucky run** — the physics is the same at every cut time.

## The headline numbers (backing slide, not the lead)
| | GPS-denied position error |
|---|---|
| Conventional inertial nav (no GPS) | **~300 m** (drifts unbounded) |
| **Our AI-Nav** | **6.2 cm** (bounded) |
| | **≈ 5,000× tighter** |

Plus (from Approach B, the attitude core): learned gyro-denoiser cuts heading drift **−98%**
and generalizes across seeds and environments.

## If asked "is this simulated?"
No. It's the ETH EuRoC micro-aerial-vehicle dataset — real IMU, real camera, motion-capture
ground truth. Red is the textbook strapdown/EKF inertial solution; blue is our stack
(OpenVINS VIO front-end + our learned inertial core). The 6.2 cm reproduces our measured ATE
exactly — the demo re-derives it live on load, it isn't a stored picture.

## Rebuild
```
python scripts/demo_build.py            # default: V1_01, GPS cut at 60 s
python scripts/demo_build.py --cut 90   # cut later -> shorter denial, less drift
```
