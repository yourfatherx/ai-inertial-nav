"""Generate the project thesis PDF (reportlab).

Produces `AI_Inertial_Nav_Thesis.pdf` in the project root: a full, professionally
structured technical thesis covering the entire project -- Approach B (learned
inertial), its hardening, Approach A (visual-inertial baseline + the learned
front-end negatives), the GPS-denied demonstrator, the drift-vs-denial-time
characterization (D3), and the deployment track (D1-D4).

All figures are pulled from results/. Every number is a verified experiment
output reproduced in-repo (results/*.json, *.log) or recorded in the project log;
nothing is fabricated. Two-pass build fills the table of contents.

    PYTHONIOENCODING=utf-8 PYTHONUTF8=1 .venv/Scripts/python.exe scripts/make_report.py
"""
from __future__ import annotations
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER, TA_LEFT
from reportlab.platypus import (BaseDocTemplate, PageTemplate, Frame,
                                Paragraph, Spacer, Image, Table, TableStyle,
                                PageBreak, HRFlowable, KeepTogether)
from reportlab.platypus.tableofcontents import TableOfContents

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = ROOT / "AI_Inertial_Nav_Thesis.pdf"

INK = colors.HexColor("#0d2b45")
ACCENT = colors.HexColor("#1f5f8b")
ACCENT2 = colors.HexColor("#12395b")
GREEN = colors.HexColor("#137333")
RED = colors.HexColor("#a61b1b")
GREY = colors.HexColor("#666666")

# ------------------------------------------------------------------ styles ----
ss = getSampleStyleSheet()
BODY = ParagraphStyle("body", parent=ss["BodyText"], fontSize=10, leading=14.5,
                      alignment=TA_JUSTIFY, spaceAfter=6)
ABSTRACT = ParagraphStyle("abstract", parent=BODY, fontSize=10, leading=15,
                          leftIndent=6, rightIndent=6)
CH = ParagraphStyle("ch", parent=ss["Heading1"], fontSize=18, leading=22,
                    textColor=INK, spaceBefore=6, spaceAfter=10)
H1 = ParagraphStyle("h1", parent=ss["Heading2"], fontSize=13, leading=17,
                    textColor=ACCENT2, spaceBefore=10, spaceAfter=5)
H2 = ParagraphStyle("h2", parent=ss["Heading3"], fontSize=11, leading=14,
                    textColor=ACCENT, spaceBefore=7, spaceAfter=3)
TITLE = ParagraphStyle("title", parent=ss["Title"], fontSize=24, leading=29,
                       textColor=INK, alignment=TA_CENTER)
SUBT = ParagraphStyle("subt", parent=ss["Normal"], fontSize=12.5, leading=17,
                      alignment=TA_CENTER, textColor=ACCENT2)
SUB = ParagraphStyle("sub", parent=ss["Normal"], fontSize=10.5, leading=15,
                     alignment=TA_CENTER, textColor=GREY)
CAP = ParagraphStyle("cap", parent=ss["Normal"], fontSize=8.5, leading=11,
                     alignment=TA_CENTER, textColor=GREY,
                     spaceBefore=2, spaceAfter=10)
KEY = ParagraphStyle("key", parent=BODY, fontSize=10, leading=14.5,
                     leftIndent=8, rightIndent=8, borderPadding=6,
                     backColor=colors.HexColor("#eef4f9"),
                     borderColor=colors.HexColor("#bcd2e2"), borderWidth=0.6,
                     spaceBefore=4, spaceAfter=8)
REF = ParagraphStyle("ref", parent=ss["Normal"], fontSize=9, leading=12.5,
                     leftIndent=10, firstLineIndent=-10, spaceAfter=4,
                     alignment=TA_LEFT)
CELL = ParagraphStyle("cell", parent=ss["Normal"], fontSize=8.5, leading=11)


def P(txt, style=BODY):
    return Paragraph(txt, style)


def bullets(items, style=BODY):
    return [P("&bull;&nbsp;&nbsp;" + it, style) for it in items]


def fig(name, caption, width=160 * mm):
    p = RES / name
    if not p.exists():
        return P(f"[missing figure: {name}]", CAP)
    img = Image(str(p))
    iw, ih = img.imageWidth, img.imageHeight
    img.drawWidth = width
    img.drawHeight = width * ih / iw
    return KeepTogether([Spacer(1, 2 * mm), img, P(caption, CAP)])


def rule(c=colors.HexColor("#cccccc")):
    return HRFlowable(width="100%", thickness=0.6, color=c,
                      spaceBefore=3, spaceAfter=8)


def styled_table(data, colWidths, green_col=None, bold_cols=()):
    t = Table(data, colWidths=colWidths, repeatRows=1)
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), ACCENT),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#eef4f9")]),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cfd8e0")),
        ("TOPPADDING", (0, 0), (-1, -1), 4.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4.5),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
    ]
    if green_col is not None:
        style += [("TEXTCOLOR", (green_col, 1), (green_col, -1), GREEN),
                  ("FONTNAME", (green_col, 1), (green_col, -1), "Helvetica-Bold")]
    for c in bold_cols:
        style += [("FONTNAME", (c, 1), (c, -1), "Helvetica-Bold")]
    t.setStyle(TableStyle(style))
    return t


# --------------------------------------------------------------- content ----
story = []

# ============================================================= TITLE PAGE ====
story += [
    Spacer(1, 26 * mm),
    P("AI-Based Inertial Navigation", TITLE),
    P("for GPS-Denied Environments", TITLE),
    Spacer(1, 8 * mm),
    P("A Learning-Augmented Inertial and Visual-Inertial Approach to "
      "Resilient UAV State Estimation", SUBT),
    Spacer(1, 4 * mm),
    HRFlowable(width="60%", thickness=1.0, color=ACCENT, spaceBefore=6,
               spaceAfter=6, hAlign="CENTER"),
    Spacer(1, 3 * mm),
    P("Technical Thesis &bull; Defence Application: Navigation under GPS "
      "Jamming, Denial, and Spoofing", SUB),
    Spacer(1, 40 * mm),
    P("<b>Abstract</b>", ParagraphStyle("ac", parent=SUB, fontSize=11,
                                        textColor=INK)),
    Spacer(1, 2 * mm),
    P("Satellite positioning is the single point of failure in autonomous flight: "
      "when GPS is jammed, denied, or spoofed &mdash; a routine condition in "
      "contested airspace &mdash; a UAV must estimate its own heading and motion "
      "from onboard sensors alone, and any drift in that estimate flies the "
      "aircraft off course. This thesis develops and validates that onboard "
      "capability on public, ground-truthed flight data (EuRoC MAV; TUM-VI for "
      "cross-sensor tests). The core contribution, <b>Approach B</b>, is a "
      "lightweight (~19k-parameter) neural gyroscope denoiser fused into a "
      "classical Error-State Kalman Filter: on held-out sequences it removes the "
      "systematic gyro bias the filter is structurally blind to, cutting "
      "open-loop attitude drift by ~98% and recovering near-oracle fused accuracy "
      "from data alone, with the win shown stable across random seeds and across "
      "environments. A cross-sensor study establishes the honest limit &mdash; the "
      "learned correction is sensor-specific &mdash; and demonstrates that a short "
      "per-IMU fine-tune closes the gap. <b>Approach A</b> adds a camera "
      "(OpenVINS stereo VIO, ~6&nbsp;cm) to bound absolute position, and reports "
      "two honest negative results on a naive learned visual front-end. Finally, a "
      "GPS-denied flight demonstrator and a drift-vs-denial-time characterization "
      "quantify the operational payoff: after 60&nbsp;s of GPS denial the learned "
      "system holds roughly three times tighter than a conventional inertial "
      "baseline. The remaining step to a fielded system is an on-board "
      "(edge/realtime) deployment, identified as future work.", ABSTRACT),
    Spacer(1, 20 * mm),
    P("Reproducible prototype &mdash; all figures and numbers regenerated from "
      "in-repo experiment outputs. Compiled 14 July 2026.", SUB),
    PageBreak(),
]

# ========================================================= TABLE OF CONTENTS ==
toc = TableOfContents()
toc.levelStyles = [
    ParagraphStyle("toc0", fontSize=11, leading=17, fontName="Helvetica-Bold",
                   textColor=INK, spaceBefore=4),
    ParagraphStyle("toc1", fontSize=9.5, leading=13.5, leftIndent=12,
                   textColor=colors.HexColor("#333333")),
]
story += [P("Contents", ParagraphStyle("toch", parent=CH, fontSize=16)),
          rule(ACCENT), toc, PageBreak()]


# helper: a chapter heading that also registers itself with the TOC
def chapter(num, title):
    return [P(f"Chapter&nbsp;{num}&nbsp;&mdash;&nbsp;{title}", CH), rule(ACCENT)]


# ============================================================ CH 1: INTRO =====
story += chapter(1, "Introduction")
story += [
    P("1.1&nbsp;&nbsp;Motivation", H1),
    P("An autonomous UAV normally knows where it is because a GNSS (GPS) receiver "
      "tells it. In contested, electronic-warfare, or denied environments that "
      "signal cannot be trusted: it may be <i>jammed</i> (drowned out), "
      "<i>denied</i> (absent), or <i>spoofed</i> (falsified). Jamming and spoofing "
      "of drones are live, named problems in modern conflict, contested logistics, "
      "and counter-UAS operations. When the fix disappears, the aircraft must fall "
      "back on sensors an adversary cannot easily corrupt and continue to estimate "
      "its own state &mdash; heading, velocity, and position &mdash; well enough to "
      "keep flying its mission or return safely."),
    P("The always-available sensor is the Inertial Measurement Unit (IMU): a "
      "three-axis gyroscope and three-axis accelerometer. But IMU signals are "
      "corrupted by slowly-varying <i>bias</i> and random <i>noise</i>, and naive "
      "integration of them makes the estimated orientation and position diverge "
      "within seconds. The most operationally dangerous component is "
      "<b>heading (yaw)</b>: it is physically <i>unobservable</i> to an IMU-only "
      "filter (gravity pins roll and pitch but carries no heading information), so "
      "any gyro yaw-bias integrates into unbounded heading error &mdash; and on a "
      "vehicle a wrong heading steers the entire trajectory off course."),

    P("1.2&nbsp;&nbsp;Thesis statement", H1),
    P("<b>A small neural network can learn the systematic error of a specific IMU "
      "and subtract it &mdash; including the unobservable yaw bias &mdash; "
      "<i>before</i> a classical filter runs, giving state-estimation accuracy the "
      "filter cannot achieve on its own, and doing so cheaply enough to run "
      "onboard.</b> A camera, when available, then bounds absolute position that "
      "inertial sensing alone cannot."),

    P("1.3&nbsp;&nbsp;Contributions", H1),
    P("This thesis makes the following contributions, each validated on public "
      "data and reproduced in the accompanying repository:"),
] + bullets([
    "<b>A learned-inertial estimator (Approach&nbsp;B).</b> A ~19k-parameter "
    "gyro denoiser fused into a 15-state ESKF that cuts open-loop attitude drift "
    "by ~98% and recovers near-oracle fused accuracy on held-out sequences "
    "(Chapters&nbsp;3&ndash;4).",
    "<b>A rigorous robustness study.</b> Multi-seed stability, cross-environment "
    "generalization, and a cross-<i>sensor</i> transfer analysis that states the "
    "method's honest limit and demonstrates the per-IMU fine-tune that fixes it "
    "(Chapter&nbsp;5).",
    "<b>A visual-inertial position layer (Approach&nbsp;A).</b> An OpenVINS "
    "stereo-VIO baseline (~6&nbsp;cm), and two honest negative results on a naive "
    "learned visual front-end that scope where learned VIO does and does not help "
    "(Chapter&nbsp;6).",
    "<b>An operational characterization.</b> A GPS-denied flight demonstrator and "
    "a drift-vs-denial-time curve quantifying the payoff (~3&times; tighter at "
    "60&nbsp;s of denial), plus a deployment analysis (Chapters&nbsp;7&ndash;8).",
]) + [
    P("1.4&nbsp;&nbsp;Scope and methodology", H1),
    P("The work uses <b>public datasets only</b> (no custom flight hardware at this "
      "stage); the deliverable is a correct, well-instrumented prototype. A guiding "
      "methodological rule is applied throughout: <i>establish a working classical "
      "baseline first, then insert the neural component</i>, so the exact "
      "contribution of the learning is always measurable and the filter and network "
      "are never debugged simultaneously. All comparisons are made on the "
      "realistic same-information setting (no oracle knowledge handed to the "
      "learned side), and negative results are reported at full strength."),
    P("<b>Operational assumption &mdash; no magnetometer.</b> A magnetometer would "
      "normally observe heading, but it is easily jammed and unreliable near "
      "motors and airframe currents, so it is treated as unavailable. Heading is "
      "held by the gyro alone &mdash; which is precisely why removing the gyro yaw "
      "bias is the central result (Chapter&nbsp;8, D4)."),
]

# ============================================== CH 2: BACKGROUND ==============
story += [PageBreak()] + chapter(2, "Background and Related Work")
story += [
    P("2.1&nbsp;&nbsp;Inertial navigation and observability", H1),
    P("Strapdown inertial navigation integrates gyroscope angular rates to track "
      "orientation and doubly-integrates accelerometer specific force (minus "
      "gravity) to track velocity and position. The Achilles' heel is error "
      "growth: a constant gyro bias produces an orientation error that grows "
      "linearly, which mis-rotates the gravity subtraction and injects an "
      "acceleration error that grows the position error cubically in time. An "
      "aided filter constrains part of this &mdash; an accelerometer gives an "
      "absolute gravity (tilt) reference that observes roll and pitch &mdash; but "
      "rotation about the gravity vector (heading/yaw) remains unobservable to an "
      "IMU+gravity system. This observability structure is the technical crux of "
      "the entire thesis."),
    P("2.2&nbsp;&nbsp;The Error-State Kalman Filter", H1),
    P("The Error-State (indirect) Kalman Filter tracks a nominal state and a small "
      "<i>error</i> state, keeping the linearization valid and the orientation "
      "error minimal (three parameters rather than a constrained quaternion). This "
      "work follows the local-error convention of S&ograve;la's quaternion-kinematics "
      "formulation [4], with a 15-dimensional error state (attitude, velocity, "
      "position, gyro bias, accelerometer bias)."),
    P("2.3&nbsp;&nbsp;Learned inertial methods", H1),
    P("Deep networks have been used to regress the systematic component of IMU "
      "error. The design here follows Brossard <i>et&nbsp;al.</i> [1], who denoise "
      "gyroscope signals with a dilated convolutional network supervised by "
      "integrated orientation increments; related lines include AI-IMU "
      "dead-reckoning and the RONIN / TLIO / RIANN family of learned-inertial "
      "odometry [5]. The distinctive design choices adopted &mdash; a bounded, "
      "zero-initialised correction so training starts exactly at the raw gyro "
      "&mdash; are detailed in Chapter&nbsp;3."),
    P("2.4&nbsp;&nbsp;Visual-inertial odometry", H1),
    P("When a camera is available, visual-inertial odometry (VIO) makes absolute "
      "scale and position observable. Approach&nbsp;A uses OpenVINS [2], an "
      "open-source filter-based (MSCKF) stereo VIO, as a strong classical "
      "baseline, and investigates replacing its hand-engineered feature front-end "
      "with a learned detector/matcher (SuperPoint [6] + LightGlue [7]). The "
      "back-end estimator is left unchanged so that any difference is attributable "
      "to the front-end."),
    P("2.5&nbsp;&nbsp;Datasets", H1),
    P("<b>EuRoC MAV</b> [3] provides 200&nbsp;Hz IMU (Analog Devices ADIS16448), "
      "stereo imagery, and millimetre-accurate Vicon/Leica ground truth including "
      "the true gyro/accel biases, across two Vicon rooms and a machine hall at "
      "easy/medium/difficult motion levels. <b>TUM-VI</b> [8] is used for the "
      "cross-sensor study; its room sequences use a Bosch BMI160-class consumer "
      "MEMS IMU &mdash; a genuine hardware change from the ADIS16448."),
]

# ============================================== CH 3: SYSTEM =================
story += [PageBreak()] + chapter(3, "System Overview and Method")
story += [
    P("The system has two complementary layers. <b>Approach&nbsp;B</b> (this "
      "chapter and Chapter&nbsp;4) is the IMU-only fallback that holds a correct "
      "heading and bounded attitude the instant GPS is lost. <b>Approach&nbsp;A</b> "
      "(Chapter&nbsp;6) adds a camera to additionally bound absolute position. "
      "Both share the same classical estimator core."),
    P("3.1&nbsp;&nbsp;Two-stage architecture: learn, then filter", H1),
    P("The learned component and the estimator are deliberately separated so each "
      "can be validated in isolation:"),
] + bullets([
    "<b>Stage&nbsp;1 &mdash; Gyro Denoiser (the AI).</b> A 1-D dilated "
    "convolutional network (~19k parameters, dilations 1&ndash;32) reads a window "
    "of the 6-channel raw IMU and outputs a 3-channel gyro correction. The "
    "correction is <i>bounded</i> by a tanh scaled by "
    "&alpha;&nbsp;=&nbsp;0.15&nbsp;rad/s (&asymp;&nbsp;8.6&nbsp;deg/s) and the "
    "output head is <i>zero-initialised</i>, so training begins exactly at the raw "
    "gyro and can only improve it: "
    "<font face='Courier'>&omega;_corr = &omega;_raw + "
    "&alpha;&middot;tanh(net(window))</font>.",
    "<b>Stage&nbsp;2 &mdash; Error-State Kalman Filter (classical).</b> The "
    "15-state ESKF of &sect;2.2 runs an IMU-driven prediction and an "
    "accelerometer-as-gravity tilt update, gated to near-static samples, that pins "
    "roll and pitch. Heading is left unobservable by construction &mdash; this is "
    "exactly the axis the denoiser stabilises upstream.",
]) + [
    P("3.2&nbsp;&nbsp;Training signal (no clean-gyro label exists)", H1),
    P("The dataset provides true <i>orientation</i>, not a per-sample clean gyro. "
      "Supervision is therefore a <b>multi-horizon relative-orientation loss</b>: "
      "the corrected gyro is integrated over horizons of 1, 8, 32, and 128 samples "
      "and the geodesic angle between each predicted relative rotation and the "
      "ground-truth relative rotation is penalised. Short horizons force removal of "
      "high-frequency noise; long horizons force removal of slow bias. The "
      "correction bound &alpha; must exceed the sensor's true bias magnitude, or "
      "the yaw bias saturates the tanh (an early &alpha;=0.05 run could remove only "
      "~66% of the &asymp;4.3&nbsp;deg/s EuRoC yaw bias; &alpha;=0.15 clears it)."),
    P("3.3&nbsp;&nbsp;Evaluation protocol", H1),
    P("A strict cross-sequence hold-out is used: the denoiser trains on six "
      "sequences and is tested on sequences it never saw. Metrics are geodesic "
      "orientation error, a dedicated yaw-only error series, and position Absolute "
      "Trajectory Error (ATE), each reported as RMSE / final / max. The decisive "
      "comparison is <i>denoised gyro through the ESKF</i> versus <i>raw gyro "
      "through the same ESKF</i>, both with initial bias set to zero &mdash; the "
      "realistic GPS-denied case in which no oracle bias is available. An oracle "
      "run (raw filter handed the true constant bias) is shown only as an upper "
      "bound."),
    P("3.4&nbsp;&nbsp;Implementation", H1),
]
tech = [
    ["Layer", "Choice"],
    ["Language / runtime", "Python 3.12, isolated uv-managed virtual environment"],
    ["Deep learning", "PyTorch 2.6.0 + CUDA 12.4 (cu124 wheels)"],
    ["Compute (training)", "NVIDIA RTX 4050 laptop GPU, 6 GB VRAM"],
    ["Classical VIO back-end", "OpenVINS (MSCKF), ROS1 Noetic / Ubuntu 20.04 "
     "under WSL2 + Docker, CPU-only, headless"],
    ["Learned visual front-end", "SuperPoint + LightGlue (evaluated, Chapter 6)"],
    ["Numerics / plotting", "NumPy, Matplotlib, Plotly (interactive demo)"],
    ["Datasets", "EuRoC MAV (ASL); TUM-VI (cross-sensor), via HTTP-range "
     "extraction from bundle archives"],
    ["Core modules", "ainav.eskf, ainav.denoiser, ainav.torch_rot, "
     "ainav.train_loss, ainav.dataset, ainav.metrics, ainav.euroc, "
     "ainav.frontend, ainav.camera"],
]
tech = [[row[0], row[1] if i == 0 else P(row[1], CELL)]
        for i, row in enumerate(tech)]
story += [styled_table(tech, [44 * mm, 116 * mm]),
          P("Table 3.1. Implementation stack.", CAP)]

# ============================================== CH 4: APPROACH B ==============
story += [PageBreak()] + chapter(4, "Approach B — Learned Inertial Navigation")
story += [
    P("This chapter reports the phased build of Approach&nbsp;B, from the raw-data "
      "sanity baseline to the fused learned estimator. The phase structure "
      "enforces the methodology of &sect;1.4: each layer's contribution is "
      "measured against the one below it."),

    P("4.1&nbsp;&nbsp;Phase 0 &mdash; Environment and data pipeline", H1),
    P("The EuRoC loader, sequence handling, and a raw-vs-truth sanity plot were "
      "validated on V1_01_easy (29,120 IMU samples over 145.6&nbsp;s). "
      "Accelerometer magnitude at rest &asymp;&nbsp;9.78&nbsp;m/s&sup2; confirms "
      "correct gravity and sensor conventions."),

    P("4.2&nbsp;&nbsp;Phase 1 &mdash; Naive integration (the motivation)", H1),
    P("Open-loop dead-reckoning directly from the raw IMU shows why the problem is "
      "hard. Drift is catastrophic and expected: position ATE RMSE "
      "&asymp;&nbsp;33&nbsp;km (max 74&nbsp;km) and orientation RMSE 114&deg; over "
      "~2.5&nbsp;minutes. This is the baseline the rest of the project must beat."),
    fig("phase1_V1_01_easy.png",
        "Figure 4.1. Phase 1 &mdash; naive open-loop integration diverges within "
        "seconds, motivating both the filter and the denoiser."),

    P("4.3&nbsp;&nbsp;Phase 2 &mdash; Classical ESKF baseline (no AI)", H1),
    P("The ESKF with raw IMU is the honest, non-exploding baseline the AI must "
      "beat. Versus Phase 1 it cuts orientation RMSE from 114&deg; to 6.96&deg; and "
      "position ATE from ~33&nbsp;km to ~317&nbsp;m. The residual ~5&deg; is "
      "essentially <b>yaw</b> &mdash; unobservable to an IMU-only filter &mdash; and "
      "that heading error is what smears the remaining horizontal position."),
    fig("phase2_V1_01_easy.png",
        "Figure 4.2. Phase 2 &mdash; the ESKF bounds drift dramatically but cannot "
        "fix the unobservable heading axis."),

    P("4.4&nbsp;&nbsp;Phase 3 &mdash; Gyro-denoising CNN (open-loop)", H1),
    P("The ~19k-parameter denoiser, trained cross-sequence, is first evaluated "
      "open-loop (pure integration, no filter) so its effect is isolated. On "
      "held-out V1_02_medium it cuts open-loop attitude drift by ~98% (raw RMSE "
      "109.6&deg; / final 178.8&deg; &rarr; denoised ~2.05&deg; / ~2.77&deg;). "
      "Crucially, the learned z-axis correction (&asymp;&minus;4.4&nbsp;deg/s) "
      "matches the true EuRoC gyro yaw-bias (+4.34&nbsp;deg/s): the network removes "
      "exactly the bias the filter is blind to, rather than fitting a per-sequence "
      "quirk."),
    fig("phase3_eval_V1_02_medium.png",
        "Figure 4.3. Phase 3 &mdash; denoised gyro (open-loop) tracks ground-truth "
        "attitude where the raw gyro runs away."),

    P("4.5&nbsp;&nbsp;Phase 4 &mdash; Fusion: denoiser + ESKF", H1),
    P("The two stages are combined: the denoised gyro is fed into the same ESKF and "
      "compared with the raw-gyro ESKF on held-out sequences. Neither side is given "
      "the true bias (both start at zero, the realistic GPS-denied case); an oracle "
      "run handing the raw filter the true bias is shown only as an upper bound. A "
      "key tuning lesson emerged: once the gyro is clean, the gravity/tilt update "
      "must be gated to <i>genuinely static</i> samples (accel gate "
      "0.05&nbsp;m/s&sup2;), otherwise it re-injects error during real acceleration "
      "and corrupts the good gyro."),
    P("Every number below is the <b>mean&nbsp;&plusmn;&nbsp;standard deviation over "
      "five independently-seeded retrains</b> (Chapter&nbsp;5), so the improvement "
      "is a property of the method, not of one lucky checkpoint.", BODY),
]
P4 = [
    ["V1_02_medium (medium)", "76.0", "1.57 &plusmn; 0.39", "0.75", "+97.9%", "631 &rarr; 114"],
    ["MH_02_easy (easy)", "90.2", "11.58 &plusmn; 4.37", "5.08", "+87.2%", "1571 &rarr; 171"],
    ["V2_03_difficult (hard)", "102.7", "3.44 &plusmn; 0.91", "4.67", "+96.6%", "828 &rarr; 141"],
]
hdr = ["Held-out sequence", "Raw ESKF\norient RMSE", "Denoised orient\nRMSE (5 seeds)",
       "Oracle\norient RMSE", "Orient\nimprovement", "ATE RMSE\n(m)"]
story += [Spacer(1, 1 * mm),
          styled_table([hdr] + P4, [42 * mm, 22 * mm, 27 * mm, 20 * mm, 24 * mm, 25 * mm],
                       green_col=2, bold_cols=(4,)),
          P("Table 4.1. Phase-4 fusion on held-out sequences (orientation RMSE in "
            "degrees). The denoiser closes almost the entire gap to the oracle; on "
            "the hardest sequence (V2_03_difficult) it beats the oracle across "
            "every seed (3.44&deg; vs 4.67&deg;) without ever being told the bias. "
            "MH_02_easy has the widest spread (&plusmn;4.4&deg;) &mdash; a large "
            "indoor hall is the hardest environment to generalize to &mdash; yet "
            "even its worst seed (15.7&deg;) beats the 90&deg; raw baseline by "
            "6&times;.", CAP)]
story += [
    fig("phase4_V1_02_medium.png",
        "Figure 4.4. Phase-4 fusion (V1_02_medium): trajectory, orientation error, "
        "heading, and position error. Denoised (green) tracks ground truth; raw "
        "(red) drifts away."),
    fig("phase4_V2_03_difficult.png",
        "Figure 4.5. Generalization to the hardest held-out sequence "
        "(V2_03_difficult): same checkpoint, same large win."),
    P("<b>Finding.</b> The denoiser recovers near-oracle attitude and position "
      "accuracy from data alone, and the win holds on every held-out sequence "
      "across two rooms and a machine hall, easy through difficult.", KEY),
]

# ============================================== CH 5: HARDENING ==============
story += [PageBreak()] + chapter(5, "Robustness and Hardening")
story += [
    P("Three questions decide whether a single good result is trustworthy: would "
      "it survive a different random seed; does it hold when the test "
      "<i>environment</i> differs from training; and does it transfer to a "
      "different <i>sensor</i>? All three were tested directly."),

    P("5.1&nbsp;&nbsp;Multi-seed stability", H1),
    P("The full pipeline (train &rarr; fuse &rarr; evaluate) was repeated for five "
      "independent seeds. Training quality is essentially deterministic: best "
      "validation geodesic error is <b>0.0958&deg;&nbsp;&plusmn;&nbsp;0.0011&deg;</b> "
      "across seeds (a ~1% relative spread), so the network reliably learns the same "
      "correction. Downstream fusion accuracy (Table&nbsp;4.1) is stable on the two "
      "Vicon-room hold-outs and wider but never failing on the machine hall. This "
      "also corrected an earlier single-checkpoint report that had drawn a lucky "
      "MH_02 seed (5.1&deg;); the honest multi-seed figure is ~11.6&deg;, still a "
      "6&times;&ndash;20&times; win over the 90&deg; raw baseline."),

    P("5.2&nbsp;&nbsp;Cross-environment generalization", H1),
    P("EuRoC records two distinct environments with the <i>same</i> IMU: the small "
      "Vicon rooms (V) and the large Machine Hall (MH). Training on one and testing "
      "on the other isolates <i>environment</i> shift from <i>sensor</i> shift. Two "
      "directions were run, three seeds each:"),
] + bullets([
    "<b>MH&nbsp;&rarr;&nbsp;V (train hall, test rooms): excellent.</b> On unseen "
    "V2_03 the denoiser reaches 3.20&deg;&nbsp;&plusmn;&nbsp;1.01 &mdash; "
    "<i>beating the oracle</i> (4.67&deg;) &mdash; and 2.57&deg;&nbsp;&plusmn;&nbsp;"
    "0.58 on V1_02, from a network that never saw a Vicon room.",
    "<b>V&nbsp;&rarr;&nbsp;MH (train rooms, test hall): degrades, never fails.</b> "
    "On unseen MH_01 the denoiser gives 22.9&deg;&nbsp;&plusmn;&nbsp;7.8 versus a "
    "116&deg; raw baseline (still a 5&times; win); MH_02 stays strong at "
    "2.90&deg;&nbsp;&plusmn;&nbsp;1.07.",
]) + [
    P("<b>The asymmetry is the honest headline:</b> the denoiser transfers cleanly "
      "to environments of similar-or-smaller scale and degrades gracefully &mdash; "
      "never catastrophically &mdash; on a larger unseen one. The remedy is broader "
      "training coverage (include hall-scale motion), a data question rather than a "
      "flaw in the method.", KEY),

    P("5.3&nbsp;&nbsp;Cross-sensor transfer &mdash; the honest limit and its fix", H1),
    P("The operationally decisive question is a change of <i>sensor</i>. The EuRoC "
      "network was trained on an ADIS16448 (tactical-grade); does it help, frozen, "
      "on a different IMU? Tested on TUM-VI room1 (Bosch BMI160-class consumer "
      "MEMS), <b>the frozen network hurts</b>: raw-gyro ESKF orientation RMSE "
      "35.6&deg; degrades to 116.0&deg; (&minus;226%), and ATE 336&nbsp;m to "
      "1479&nbsp;m (&minus;340%). The cause is precise: the network confidently "
      "subtracted its learned EuRoC yaw-bias correction (&asymp;&minus;4.8&nbsp;"
      "deg/s) from a sensor whose true bias is different, so it removed a constant "
      "that was not there and injected heading error. (Note the raw TUM-VI baseline "
      "is already far cleaner than raw EuRoC &mdash; the Bosch part has a smaller "
      "intrinsic bias, so there is both less to fix and more to break.)"),
    P("<b>The learned correction is therefore sensor-specific</b> &mdash; it "
      "encodes one IMU's systematic error, not a universal denoiser. Deployment "
      "step D2 is the fix: warm-start from the EuRoC weights, recompute input "
      "normalization on the new sensor, and fine-tune briefly on a small amount of "
      "its data. Trained on two TUM-VI rooms and evaluated on a never-seen third "
      "room, open-loop attitude error moves from raw 0.858&deg; and frozen-net "
      "1.624&deg; (worse than raw &mdash; the failure) to a fine-tuned "
      "<b>0.808&deg;</b> (now beating raw)."),
    fig("cross_imu_room1.png",
        "Figure 5.1. Cross-sensor transfer to TUM-VI (Bosch MEMS). Frozen, the "
        "EuRoC network degrades this sequence badly (top-down and error panels); "
        "the honest conclusion is that the correction is sensor-specific and a "
        "short per-IMU fine-tune restores the win."),
    P("<b>Honest scope.</b> The denoiser generalizes across seeds and across "
      "environments <i>on the sensor it was trained on</i>, but does not transfer "
      "to a new IMU unchanged. This is the expected behaviour of a model that "
      "learns a specific sensor's error, and the remedy is a short per-IMU "
      "fine-tune, not a redesign &mdash; converting deployment step D2 from a "
      "stated risk into a demonstrated, closeable gap.", KEY),
]

# ============================================== CH 6: APPROACH A ==============
story += [PageBreak()] + chapter(6, "Approach A — Visual-Inertial Navigation")
story += [
    P("Approach&nbsp;B holds heading and bounded attitude, but IMU-only sensing "
      "cannot make <i>absolute position</i> observable &mdash; position still "
      "drifts, only much more slowly. Approach&nbsp;A adds a camera. The strategy "
      "is to wrap a proven open VIO back-end as the classical baseline and put the "
      "research into a learned visual front-end."),

    P("6.1&nbsp;&nbsp;A0 &mdash; Camera pipeline", H1),
    P("Stereo cam0/cam1 loading, image&ndash;IMU synchronization, and undistortion "
      "were validated against the EuRoC calibration (fu&nbsp;&asymp;&nbsp;458.7), "
      "establishing the front-end input for both the classical and learned "
      "trackers."),

    P("6.2&nbsp;&nbsp;A1 &mdash; OpenVINS stereo-VIO baseline", H1),
    P("OpenVINS (filter-based MSCKF stereo VIO) was built and run headless under "
      "WSL2 + Docker (ROS1 Noetic / Ubuntu 20.04, CPU-only). On V1_01 it achieves "
      "<b>~6.2&nbsp;cm</b> ATE &mdash; against the Approach&nbsp;B inertial-only "
      "baseline (~317&nbsp;m) and naive integration (~33&nbsp;km) this is a "
      "roughly <b>5000&times;</b> reduction in position error, and it is the blue "
      "&lsquo;AI-Nav&rsquo; track in the demonstrator of Chapter&nbsp;7."),
    fig("compareA1_vs_B.png",
        "Figure 6.1. Absolute position error: OpenVINS stereo VIO (A1) versus the "
        "inertial-only ESKF (B) and naive integration, on V1_01. Adding a camera "
        "bounds absolute position to centimetres."),

    P("6.3&nbsp;&nbsp;A2 &mdash; Learned visual front-end (two honest negatives)", H1),
    P("The research question was whether a learned detector/matcher "
      "(SuperPoint&nbsp;+&nbsp;LightGlue) could replace the classical KLT front-end "
      "while keeping the same back-end. Two experiments were run and both returned "
      "instructive negative results."),
    P("<b>Front-end study (component level).</b> On easy, well-lit data (V1_01) the "
      "naive learned tracker loses to KLT on track-length metrics &mdash; expected, "
      "since KLT explicitly maintains features by optical flow whereas naive "
      "detect-then-match fragments a track on a single missed detection. On "
      "genuinely difficult data (V1_03) a meaningful crossover appears "
      "<b>under low light</b>: the learned front-end's epipolar inlier ratio "
      "(0.784) beats KLT's (0.635). The mechanism is the real finding &mdash; KLT "
      "reports its <i>longest</i> tracks in the dark (mean length 86 frames) but "
      "those long tracks are <i>drifting</i> (the flow asserts &lsquo;same "
      "feature&rsquo; while the point slides off the landmark), which is worse for a "
      "back-end than an honestly-dropped track. The learned matcher stays short but "
      "geometrically clean."),
    fig("phaseA2_frontend_V1_03_difficult.png",
        "Figure 6.2. Front-end comparison on V1_03_difficult across degradations. "
        "The defensible claim is not &lsquo;learned beats KLT&rsquo; but "
        "&lsquo;the learned matcher does not drift under low light, whereas KLT "
        "does and hides it behind long track counts&rsquo; &mdash; a distinction "
        "that only appears on hard data.", width=150 * mm),
    P("<b>Back-end integration (system level).</b> Injecting each front-end's "
      "tracks into a monocular MSCKF (via the simulation feed path) diverged: the "
      "feed bypasses OpenVINS' native outlier rejection, and long high-leverage "
      "tracks amplify residual outliers. Adding a fair, tracker-agnostic 2-view "
      "RANSAC gate did not rescue it &mdash; all runs still diverged and the gate "
      "<i>shortened</i> the already-borderline learned tracks below what the "
      "monocular filter needs to triangulate, regressing the one prior survivor. "
      "The comparison was stopped at the pre-agreed time-box."),
    P("<b>Conclusion (scoped, not overclaimed).</b> A naive learned front-end "
      "injected into a monocular MSCKF is the wrong tool: it is not a tuning knob "
      "away from working. What survives is a real component-level result &mdash; "
      "the learned tracker is more geometrically honest under low light &mdash; not "
      "an end-to-end VIO win. A proper learned VIO would need a descriptor-based "
      "track manager, which is out of scope. This negative was surfaced "
      "<i>before</i> any expensive deeper integration, exactly as the de-risking "
      "methodology intends.", KEY),
]

# ============================================== CH 7: DEMONSTRATOR + D3 =======
story += [PageBreak()] + chapter(7, "GPS-Denied Demonstrator and Drift Characterization")
story += [
    P("7.1&nbsp;&nbsp;The flight demonstrator", H1),
    P("To make the capability legible to non-specialists, a self-contained, "
      "browser-based demonstrator replays real EuRoC flight with three "
      "trajectories on one clock: <b>ground truth</b> (green), <b>our AI-Nav</b> "
      "(blue, the ~6&nbsp;cm OpenVINS solution), and a <b>conventional INS</b> "
      "(red) that rides truth until a chosen &lsquo;GPS-cut&rsquo; instant and then "
      "dead-reckons on inertial data alone. All three begin locked together; at the "
      "cut the red path peels off and balloons to hundreds of metres while blue "
      "stays on truth. Every number traces to validated results (the reproduced "
      "6.2&nbsp;cm AI-Nav ATE is asserted as a sanity gate so the demo cannot "
      "silently cheat). An interactive version lets a viewer choose the cut instant "
      "and watch the divergence re-form live, offline."),

    P("7.2&nbsp;&nbsp;D3 &mdash; Drift versus denial-time", H1),
    P("The single question a defence buyer asks is: <i>GPS dies &mdash; how far off "
      "are you after 30, 60, 120 seconds?</i> This is answered as a curve, not a "
      "point. Many GPS-cut instants are swept across a real sequence; from each cut "
      "the system dead-reckons inertially and the horizontal/3-D position error is "
      "recorded at fixed elapsed horizons. Two propagators are compared under an "
      "identical filter and identical GT-aligned initial state &mdash; "
      "<b>raw-gyro ESKF (conventional INS)</b> versus <b>denoised-gyro ESKF "
      "(AI-Nav)</b> &mdash; so the only difference is whether the gyro is cleaned "
      "first. Results are aggregated over all cut points (median and inter-quartile "
      "band), which is what makes the number quotable rather than lucky."),
]
D3 = [
    ["30 s", "290", "67", "+77%"],
    ["60 s", "781", "288", "+63%  (3&times; tighter)"],
    ["120 s", "1704", "582", "+66%"],
]
story += [Spacer(1, 1 * mm),
          styled_table([["GPS-denial time", "Conventional INS\nmedian error (m)",
                         "AI-Nav (ours)\nmedian error (m)", "Improvement"]] + D3,
                        [34 * mm, 42 * mm, 42 * mm, 42 * mm], green_col=2, bold_cols=(3,)),
          P("Table 7.1. Position error versus GPS-denial time on V1_01, aggregated "
            "over 20 swept cut points (IMU-only after each cut). At every horizon "
            "the learned system is markedly tighter; at 60&nbsp;s it holds roughly "
            "three times closer than conventional inertial navigation. The sample "
            "count falls at long horizons (fewer cut points have 120&nbsp;s of "
            "remaining flight), so the headline uses the well-sampled 60&nbsp;s "
            "point.", CAP)]
story += [
    fig("d3_drift_curve.png",
        "Figure 7.1. GPS-denied drift versus denial-time (median, IQR band, log "
        "axis). The two curves are cleanly separated at every horizon: the learned "
        "correction slows the unbounded inertial drift that a conventional system "
        "suffers the moment GPS is lost.", width=140 * mm),
    P("<b>Bug caught during this study (reported for honesty).</b> An initial "
      "version seeded the filter with the ground-truth gyro bias, which "
      "double-corrects the already-debiased learned run and <i>inverted</i> the "
      "result (AI appeared worse). The honest GPS-denied baseline never hands the "
      "filter the true bias &mdash; both sides start at zero &mdash; and that "
      "unobservable bias is exactly what the denoiser removes. Fixing the seeding "
      "produced the physically-correct curve above. It is recorded here because a "
      "silently-wrong plot is the failure mode this project most guards against.", KEY),
]

# ============================================== CH 8: DEPLOYMENT ==============
story += [PageBreak()] + chapter(8, "Deployment Considerations")
story += [
    P("Moving from a dataset demonstration toward a fielded system defines a "
      "deployment track, D1&ndash;D4. Three of the four are addressed in this "
      "thesis; the fourth requires target hardware."),

    P("8.1&nbsp;&nbsp;D1 &mdash; Edge / realtime (remaining, needs hardware)", H1),
    P("The ~19k-parameter denoiser plus ESKF must run onboard, within the "
      "per-IMU-sample latency budget, on flight-controller-class compute (e.g. an "
      "NVIDIA Jetson), ideally quantized (INT8). This is the only remaining item "
      "and is genuinely blocked on the embedded target; a laptop CPU + "
      "quantization proxy is possible but is a proxy, not the fielded claim. "
      "Parameter count and the vectorized inference path (&sect;8.5) make the "
      "budget plausible, but it is asserted here as future work, not a result."),

    P("8.2&nbsp;&nbsp;D2 &mdash; Real-IMU transfer (addressed, Chapter 5)", H1),
    P("Demonstrated in &sect;5.3: a frozen network does not transfer across "
      "sensors, but a short warm-start per-IMU fine-tune recovers the win on a "
      "held-out room of a physically different (Bosch MEMS) IMU. Remaining work is "
      "to broaden to more drone-grade IMUs and to automate the per-IMU calibration "
      "step."),

    P("8.3&nbsp;&nbsp;D3 &mdash; Drift under sustained denial (addressed, Chapter 7)", H1),
    P("Characterized in &sect;7.2: the drift-vs-denial-time curve quantifies "
      "accuracy as a function of how long GPS has been unavailable, and shows the "
      "learned system holding ~3&times; tighter at 60&nbsp;s. Extending to "
      "propeller-vibration and aggressive-maneuver regimes beyond EuRoC's handheld "
      "motion is the natural follow-on."),

    P("8.4&nbsp;&nbsp;D4 &mdash; No-magnetometer assumption (addressed)", H1),
    P("The system uses no magnetometer: heading is held by the denoised gyro alone, "
      "verified in code (the estimator fuses only gyro prediction and "
      "accelerometer-as-gravity tilt; there is no heading measurement of any kind). "
      "This is a deliberate, jam-proof design choice &mdash; an adversary who can "
      "deny GPS can also spoof a magnetometer &mdash; and it is precisely why the "
      "unobservable yaw axis, and the learned correction that stabilises it, are "
      "central to the whole thesis."),

    P("8.5&nbsp;&nbsp;Engineering note &mdash; training-loop vectorization", H1),
    P("The orientation-integration loss originally ran a 400-step sequential "
      "quaternion loop in Python, starving the GPU (~14% utilisation). Because the "
      "quaternion product is associative, the cumulative product was replaced with "
      "an associative parallel-prefix (Hillis&ndash;Steele) scan requiring only "
      "~log&#8322;(T)&asymp;9 vectorized steps. This made the loss ~44&times; faster "
      "and cut a full six-sequence, 40-epoch retrain from ~70&nbsp;minutes to "
      "~4&nbsp;minutes, with gradients numerically identical to the loop (maximum "
      "difference 1e-8) &mdash; the enabling change behind the multi-seed and "
      "cross-environment studies of Chapter&nbsp;5."),
]

# ============================================== CH 9: DISCUSSION ==============
story += [PageBreak()] + chapter(9, "Discussion, Limitations, and Future Work")
story += [
    P("9.1&nbsp;&nbsp;Why the method works", H1),
    P("The result rests on observability. Heading and gyro yaw-bias are "
      "unobservable to an IMU+gravity filter, so a raw-gyro ESKF with no bias prior "
      "drifts in yaw without bound. The denoiser removes that bias from the data "
      "upstream, so the identical filter stays bounded &mdash; value the AI adds "
      "that the filter cannot obtain on its own. The bounded, zero-initialised "
      "correction ensures training starts at the raw gyro and can only improve it, "
      "and matching &alpha; to the true bias magnitude is what lets the yaw bias be "
      "fully removed."),
    P("9.2&nbsp;&nbsp;Limitations", H1),
] + bullets([
    "<b>Sensor-specific correction.</b> The learned bias does not transfer across "
    "IMUs unchanged (&sect;5.3); deployment requires a per-IMU fine-tune.",
    "<b>Environment coverage.</b> Generalization degrades (never fails) on unseen "
    "environments larger/more dynamic than training (&sect;5.2); broader flight "
    "data is the remedy.",
    "<b>Position still drifts inertially.</b> Approach&nbsp;B bounds attitude, not "
    "absolute position; the camera (Approach&nbsp;A) is what bounds position, and "
    "the learned front-end for it is an open problem (&sect;6.3).",
    "<b>Not yet on-target.</b> Realtime onboard performance (D1) is argued "
    "plausible but not yet demonstrated on embedded hardware.",
    "<b>Dataset domain.</b> EuRoC/TUM-VI are handheld/MAV indoor datasets; "
    "propeller vibration and aggressive flight remain to be characterized.",
]) + [
    P("9.3&nbsp;&nbsp;Future work", H1),
] + bullets([
    "<b>Edge deployment (D1):</b> quantized realtime inference on a "
    "flight-controller-class module, with a measured latency budget.",
    "<b>Learned VIO done right:</b> a descriptor-based track manager so the "
    "learned front-end's low-light honesty (&sect;6.3) can be realised as an "
    "end-to-end position win.",
    "<b>Automated per-IMU calibration:</b> a turnkey fine-tune step from a short "
    "logged flight, closing D2 operationally.",
    "<b>Flight-realism characterization (D3+):</b> vibration, aggressive "
    "maneuvers, and longer denial windows.",
])

# ============================================== CH 10: CONCLUSION ============
story += [PageBreak()] + chapter(10, "Conclusion")
story += [
    P("This thesis set out to keep a UAV navigating when GPS is jammed, denied, or "
      "spoofed, using only sensors an adversary cannot easily corrupt. Its central "
      "result is that a lightweight neural gyroscope denoiser, fused into a "
      "classical error-state filter, removes the systematic gyro bias the filter is "
      "structurally blind to &mdash; cutting open-loop attitude drift by ~98% and "
      "recovering near-oracle fused accuracy on held-out flight, with the win shown "
      "stable across seeds and environments. The honest limits are stated and, "
      "where possible, closed: the correction is sensor-specific but a short "
      "per-IMU fine-tune restores it; generalization degrades gracefully rather "
      "than catastrophically on unseen large environments. A camera layer bounds "
      "absolute position to centimetres, and a scoped investigation of a learned "
      "visual front-end returns two instructive negatives that map exactly where "
      "learned VIO does and does not yet help. Operationally, the payoff is "
      "quantified: after a minute of GPS denial the learned system holds roughly "
      "three times tighter than conventional inertial navigation, with no "
      "magnetometer and no external signal. What remains is to carry the proven "
      "algorithm onto embedded flight hardware &mdash; the one deployment step that "
      "needs the target board rather than more analysis."),
    Spacer(1, 4 * mm),
    P("<b>Status.</b> Approach&nbsp;B complete and hardened; Approach&nbsp;A "
      "baseline complete with the learned-front-end question honestly scoped; "
      "demonstrator and drift characterization delivered; deployment steps "
      "D2/D3/D4 addressed; D1 (edge/realtime) the sole remaining, hardware-gated "
      "task.", KEY),
]

# ============================================== REFERENCES ====================
story += [PageBreak(), P("References", ParagraphStyle("refh", parent=CH,
                                                      fontSize=16)), rule(ACCENT)]
refs = [
    "[1] M. Brossard, S. Bonnabel, A. Barrau. &ldquo;Denoising IMU Gyroscopes with "
    "Deep Learning for Open-Loop Attitude Estimation.&rdquo; IEEE Robotics and "
    "Automation Letters, 2020.",
    "[2] P. Geneva, K. Eckenhoff, W. Lee, Y. Yang, G. Huang. &ldquo;OpenVINS: A "
    "Research Platform for Visual-Inertial Estimation.&rdquo; IEEE ICRA, 2020.",
    "[3] M. Burri <i>et&nbsp;al.</i> &ldquo;The EuRoC Micro Aerial Vehicle "
    "Datasets.&rdquo; International Journal of Robotics Research, 2016.",
    "[4] J. S&ograve;la. &ldquo;Quaternion Kinematics for the Error-State Kalman "
    "Filter.&rdquo; arXiv:1711.02508, 2017.",
    "[5] W. Liu <i>et&nbsp;al.</i> &ldquo;TLIO: Tight Learned Inertial "
    "Odometry.&rdquo; IEEE RA-L, 2020. (See also RONIN, RIANN, AI-IMU "
    "dead-reckoning.)",
    "[6] D. DeTone, T. Malisiewicz, A. Rabinovich. &ldquo;SuperPoint: "
    "Self-Supervised Interest Point Detection and Description.&rdquo; CVPR "
    "Workshops, 2018.",
    "[7] P. Lindenberger, P.-E. Sarlin, M. Pollefeys. &ldquo;LightGlue: Local "
    "Feature Matching at Light Speed.&rdquo; ICCV, 2023.",
    "[8] D. Schubert <i>et&nbsp;al.</i> &ldquo;The TUM VI Benchmark for Evaluating "
    "Visual-Inertial Odometry.&rdquo; IEEE/RSJ IROS, 2018.",
    "[9] S. Umeyama. &ldquo;Least-Squares Estimation of Transformation Parameters "
    "Between Two Point Patterns.&rdquo; IEEE T-PAMI, 1991.",
]
story += [P(r, REF) for r in refs]


# ------------------------------------------------------------- doc template ---
class ThesisDoc(BaseDocTemplate):
    """BaseDocTemplate that records H1/H2/chapter headings into the TOC."""

    def afterFlowable(self, flowable):
        if not isinstance(flowable, Paragraph):
            return
        name = flowable.style.name
        text = flowable.getPlainText()
        if name == "ch":
            self.notify("TOCEntry", (0, text, self.page))
        elif name == "refh" or name == "toch":
            self.notify("TOCEntry", (0, text, self.page))
        elif name == "h1":
            self.notify("TOCEntry", (1, text, self.page))


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#888888"))
    canvas.drawString(20 * mm, 12 * mm,
                      "AI-Based Inertial Navigation for GPS-Denied Environments "
                      "— Technical Thesis")
    canvas.drawRightString(190 * mm, 12 * mm, f"Page {doc.page}")
    canvas.restoreState()


def build():
    doc = ThesisDoc(str(OUT), pagesize=A4,
                    leftMargin=20 * mm, rightMargin=20 * mm,
                    topMargin=18 * mm, bottomMargin=20 * mm,
                    title="AI-Based Inertial Navigation for GPS-Denied "
                          "Environments — Technical Thesis")
    frame = Frame(doc.leftMargin, doc.bottomMargin,
                  doc.width, doc.height, id="main")
    doc.addPageTemplates([PageTemplate(id="all", frames=[frame],
                                       onPage=footer)])
    doc.multiBuild(story)   # two passes: fills the table of contents
    print(f"[pdf] {OUT}")


if __name__ == "__main__":
    build()
