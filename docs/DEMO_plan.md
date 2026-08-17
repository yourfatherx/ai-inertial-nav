# GPS-Denied Flight Demonstrator — build plan (Tier 0 → Tier 2)

**Goal:** an investor-facing, visceral "without us vs. with us" replay on **real EuRoC data**,
built entirely from already-validated results. Stack: **self-contained Plotly HTML** (no server, emailable).

**Hero sequence:** `V1_01_easy` — because our strongest validated number lives there.

Three trajectories, one clock:
- 🟩 **Ground truth** — where the drone actually flew (`state_groundtruth_estimate0`).
- 🟦 **AI-Nav (ours)** — `results/est_V1_01.txt` (A1 OpenVINS VIO), **6.2 cm** RMSE (validated in `compareA1_vs_B.json`). Hugs truth.
- 🟥 **Conventional INS, GPS-denied** — strapdown dead-reckoning seeded from true state at the "GPS cut" instant, then raw-IMU-integrated forward. Peels off and balloons (we've measured 317 m ESKF / 33 km naive).

All numbers trace to `results/compareA1_vs_B.json`. Nothing invented.

---

## Deliverables
1. `scripts/demo_build.py` — one script, reused loader + the **already-validated** epoch-shift/alignment from `compareA1_vs_B.py`. Builds the shared data bundle and emits both HTMLs. `--seq`, `--cut`, `--fps` flags.
2. `results/demo_replay.html` — **Tier 1**: animated 3D replay, Play button + time scrubber, "⚠ GPS DENIED" banner that flips on at the cut time, live error readout.
3. `results/demo_interactive.html` — **Tier 2**: a **GPS-cut-time slider**; moving it re-seeds and re-diverges the red INS path from that instant. Pure client-side (precomputed frames), no server.
4. `docs/DEMO_pitch.md` — 6-line narration script + the one market one-liner, so you can talk over it.

## Success criteria (verifiable)
- [ ] Blue overlaps green within a few cm end-to-end; recomputed RMSE == 6.2 cm (matches `compareA1_vs_B.json` → proves the demo isn't cheating).
- [ ] Red sits on truth **before** the cut, diverges **after** it, exceeds ~100 m by end.
- [ ] `demo_replay.html` plays standalone in a browser with no Python running (double-click test).
- [ ] `demo_interactive.html`: dragging the cut slider visibly moves where red departs, offline.
- [ ] Everything reproducible: `python scripts/demo_build.py` regenerates both files.

## Plan
1. **Data bundle** (`demo_build.py`) → verify: prints reproduced A1 RMSE == 6.2 cm.
   - Load `V1_01_easy` via `ainav.euroc.load_sequence`.
   - Load `est_V1_01.txt`, apply the validated `t - t0_abs` epoch shift, Umeyama-align to GT (reuse `compareA1_vs_B.py` logic verbatim).
   - **INS integrator** `ins_deadreckon(imu, seed_state, t_cut)`: from GT pos/vel/quat at `t_cut`, integrate quaternion (gyro) + world-frame accel (minus gravity) forward. Returns the red path for a given cut. This is the Tier 2 primitive.
2. **Tier 1 replay** (`demo_replay.html`) → verify: standalone playback, banner flips at cut.
   - Plotly `frames` over a common resampled clock (~20 fps). Three 3D line traces that grow with time; a moving marker per path; annotation toggling to "GPS DENIED" at cut; title shows live blue-vs-green error.
3. **Tier 2 interactive** (`demo_interactive.html`) → verify: offline cut-slider re-diverges red.
   - Precompute red INS paths for N cut times (e.g. every 5 s). Plotly slider swaps which red path is shown + moves the banner. All client-side, no callback server.
4. **Pitch card** (`DEMO_pitch.md`) → verify: reads in under 30 s.
5. **Refresh graphify graph** (standing rule) → verify: new `demo_build` nodes appear, health OK.

## Notes / risks
- Plotly is **MISSING** in the venv — install `plotly` (pure-python, no native build; fast). No other new deps (numpy/scipy present).
- 3D animated Plotly with many frames can get heavy; keep to ~20 fps over the ~144 s sequence and decimate traces → target < 8 MB HTML so it emails.
- Framing: call it a **"GPS-denied flight demonstrator,"** not a "digital twin" (overpromises to a defence audience). Lead with jamming/spoofing market pain, not the method.
