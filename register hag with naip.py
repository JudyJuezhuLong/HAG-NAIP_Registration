"""Shift (and Lean) the HAG into NAIP's view geometry -- V4 ablation sets.

Model: a CHM point at height z appears in NAIP displaced by

    d(z) = t + z * a            (t in pixels, a in metres/metre)

  t : rigid shift of everything (ground, roofs, shrubs, crowns) -- NAIP vs. LiDAR
      georeferencing / orthorectification offset. Can be several metres.
  a : lean vector = tan(view angle) * unit vector away from the image nadir.
      Only tall objects move noticeably (a 2 m shrub moves <= 2*|a| m).

Both are ~constant over a 154 m chip. If t is left out, the fit absorbs the shift into
a (in closed canopy almost all pixels are 15-20 m tall, so z*a looks like a shift):
the canopy then improves but keeps a residual same-direction offset, because a lean
extends crowns away from their bases instead of moving the bases. So t and a are
fitted together:

  1. sun az/el from SDATE + chip location                            (sun.py)
  2. render the CHM as NAIP would see it: cast shadows (ground + canopy) and
     hillshade from the sun, crown columns swept from base (t) to top (t + h*a)
  3. NAIP targets ~ linear(crown, visible ground shadow, height, illumination);
     t by exhaustive integer search (cheap: shifting a rendering), a by grid search,
     alternated: t | a=0  ->  a | t  ->  t | a  ->  fine a | t  ->  fine t | a
  4. write the leaned + shifted CHM and canopy label on the NAIP grid, with an ignore
     band where the label flips across near-best lean vectors

Shadows on the ground move with t only; crown tops move with t + h*a; the sunlit /
shaded texture of crowns (hillshade) moves with t + h*a -- that texture is what keeps
closed-canopy chips identifiable where no ground shadow is visible.

V4 -- ablation export. One run can write several label sets, each fitted per chip and
written to its own sub-folder of OUT_DIR (EXPORT_SETS):

  set              fit t (shift)   fit a (lean)   HAG / mask geometry
  1_raw            no              no             LiDAR HAG as is (clipped to the NAIP grid)
  *2_shift*        yes             no             HAG and mask moved rigidly by t
  3_lean           no              yes            leaned only (t = 0; V3 renderer -- not the V2 file)
  4_shift_lean     yes             yes            moved by t and leaned (= V3)

*2_shift* is the most robust version used in the paper.

FIT_SHIFT / FIT_LEAN are the single-set switches used when EXPORT_SETS is None.
TCH_REMAP merges TCH classes on output (default 6 dark green -> 3 green).

Outputs per set folder (NAIP grid, CHIP x CHIP):
  <stem>_naip.tif          float32 4 bands 0-1, NaN = unknown
  <stem>_hag.tif           float32 metres, NaN = unknown  (named _hag_leaned.tif in lean sets)
  <stem>_canopy.tif        uint8  0 sunlit non-canopy | 1 canopy | 2 ground shadow (non-canopy)
                                  254 unknown (no data) | 255 lean-uncertain (ignore in loss)
  <stem>_tch.tif           uint8  TCH on the NAIP grid after TCH_REMAP, 0 = unknown
  pairs_<set>.csv          naip, hag, mask, tch paths (+ stem) -> input for train_multitask_pl.py
  lean_summary_<set>.csv   one row per input row (sun, t, a, scores, fractions, flags, status)
"""
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
import rasterio
from scipy.ndimage import binary_dilation, binary_erosion, gaussian_filter, grey_closing, uniform_filter

from pairs import CHIP, SkipPair, load_pair, read_pair_table
from sun import SDATE_TZ, parse_sdate, solar_az_el

# ----------------------------------------------------------------- settings
CSV_PATH = r".\pairs_with_sdate.csv"
OUT_DIR = r".\ablation_sets"

# ---- which label sets to export: (folder name, fit shift t, fit lean a). None -> one set from FIT_SHIFT/FIT_LEAN
EXPORT_SETS = [
    ("1_raw", False, False),           # (1) unshifted, unleaned HAG
    ("2_shift", True, False),          # (2) shifted, unleaned HAG
    # ("3_lean", False, True),         # (3) leaned only, t = 0          # Not stable
    # ("4_shift_lean", True, True),    # (4) shifted + leaned            # Not stable
]
FIT_SHIFT = True        # single-set mode (EXPORT_SETS = None): fit the rigid NAIP-LiDAR shift t
FIT_LEAN = True         # single-set mode: fit the lean vector a
TCH_REMAP = {6: 3}      # merge TCH classes on output: 6 dark green -> 3 green ({} = keep all)
EDGE_IGNORE_PX = 0      # sets WITHOUT lean: mark a +-N px band around canopy edges 255 (0 = off),
                        # e.g. to give them an ignore band comparable to the lean sets
WRITE_NAIP = True       # write <stem>_naip.tif into every set folder (sets stay self-contained)
check_DOY_and_SDATE = False   # check SDATE day-of-year against doy_NNN in the NAIP file name
HMIN = 2.0              # canopy height threshold (m); also the "tall" threshold for leaning
CROWN_BASE_FRAC = 0.0   # column seen from base_frac*h to h; 0 = crown reaches ground (no base gap)
FIT_CROWN_BASE = False  # True: also try base_frac 0.25 / 0.5 per chip and keep the best score
T_MAX_PX = 10           # max |t| component searched (px); 10 px = 6 m at 0.6 m  (as in your V3 run)
A_MAX = 0.5             # max |a| component searched (tan 26.6 deg)
COARSE_STEP = 0.05      # lean search: coarse grid over +-A_MAX ...  (as in your V3 run)
MID_STEP = 0.025        # ... then +-COARSE_STEP around the best ...
FINE_STEP = 0.0125      # ... then +-2*FINE_STEP
T_COARSE_STEP = 2       # shift search: every 2 px over +-T_MAX_PX, then 1 px around the best
SUBSAMPLE = 2           # score every 2nd row/col during the searches (4x faster, same support)
TOL = 0.005             # absolute score tolerance for the near-best count (printed)
TOL_GAIN = 0.25         # near-best set for the ignore band: within 25% of the gain of best
VOTE = (0.2, 0.8)       # lean-uncertain if 20-80% of near-best a disagree with the best
AMBIG_TOL = 0.002       # lean vs. pure shift indistinguishable if score gap < this
BUFFER_PX = 30          # score only within this distance of tall objects / shadows (+ T_MAX_PX)
MIN_SCORE_PX = 2000     # fewer usable pixels -> no fit, t = a = 0, flagged
QA_PNG_FIRST_N = 3000   # write a QA figure for the first N fitted chips (0 = none)

LAB_GROUND, LAB_CANOPY, LAB_SHADOW, LAB_UNKNOWN, LAB_UNCERTAIN = 0, 1, 2, 254, 255
OBJECTIVES = {                           # target columns: 0 Vrgb, 1 g-chroma, 2 b-chroma, 3 NIR, 4 NDVI
    "brightness only": (0,),
    "brightness+chromaticity": (0, 1, 2),
    "NIR-aware (used)": (0, 3, 4),
}
USE_OBJECTIVE = "NIR-aware (used)"
_KEY = 1024.0                            # z-buffer key: round(z*100)*_KEY + attribute(0..1023)


# ----------------------------------------------------------------- physics
def cast_shadow(h, az, el, pix, maxdist_m):
    """Cast shadow on the LiDAR surface (ground and canopy): p is shadowed if some q toward
    the sun at distance s has h(q) > h(p) + s*tan(el). Image coords: x = E, y = S (north-up)."""
    h0 = np.nan_to_num(h, nan=0.0)
    n0, n1 = h0.shape
    sx, sy = np.sin(np.radians(az)), -np.cos(np.radians(az))    # one step toward the sun
    tanel = np.tan(np.radians(el))
    yy, xx = np.mgrid[0:n0, 0:n1]
    sh = np.zeros(h0.shape, bool)
    for s_px in np.arange(1.0, maxdist_m / pix, 0.5):
        qx = np.rint(xx + sx * s_px).astype(int)
        qy = np.rint(yy + sy * s_px).astype(int)
        ok = (qx >= 0) & (qx < n1) & (qy >= 0) & (qy < n0)
        hq = np.zeros_like(h0)
        hq[ok] = h0[qy[ok], qx[ok]]
        sh |= hq > h0 + s_px * pix * tanel + 0.5
    return sh


def hillshade(h, az, el, pix, sigma=1.0):
    """cos(angle between surface normal and sun), clipped at 0 (no cast shadows)."""
    z = gaussian_filter(np.nan_to_num(h, nan=0.0), sigma)
    dzdy, dzdx = np.gradient(z, pix)                  # dzdy along image rows (south +)
    nE, nN, nU = -dzdx, dzdy, np.ones_like(z)         # normal in (E, N, Up)
    norm = np.sqrt(nE ** 2 + nN ** 2 + 1.0)
    ce = np.cos(np.radians(el))
    sE, sN, sU = np.sin(np.radians(az)) * ce, np.cos(np.radians(az)) * ce, np.sin(np.radians(el))
    return np.clip((nE * sE + nN * sN + nU * sU) / norm, 0, 1).astype(np.float32)


def lean_view(h, attr, ax, ay, pix, hmin=HMIN, base_frac=CROWN_BASE_FRAC, side_attr=None):
    """Forward-warp tall pixels (h > hmin) into NAIP's oblique view (no shift t here).

    Each tall pixel is a vegetation column; its point at height z is seen displaced by z*a,
    so the column is swept from its base (z = base_frac*h; default 0 -> stays on its base) to its
    top (h*a). Z-buffer: the highest visible point wins (np.maximum.at on a key that packs height
    and attribute, so the winner's attribute comes with it). Stored height is max(z, hmin+0.01):
    the lower part of a tall tree is still "tree" in the image.
    attr: per-pixel attribute (0-1) carried by the top point; side_attr: attribute of the column
    side (z < h) seen by the camera; None -> same as top.
    Returns (height, attribute); height 0 / attribute NaN where nothing tall lands."""
    shape = h.shape
    hh = np.nan_to_num(h, nan=0.0)
    ok = hh > hmin
    if not ok.any():
        return np.zeros(shape, np.float32), np.full(shape, np.nan, np.float32)
    ys, xs = np.nonzero(ok)
    hv = hh[ys, xs]
    at_top = np.clip(np.rint(np.nan_to_num(attr[ys, xs]) * 1023), 0, 1023)
    at_side = at_top if side_attr is None else np.full_like(at_top, np.clip(np.rint(side_attr * 1023), 0, 1023))
    max_px = float(hv.max() * np.hypot(ax, ay) / pix)
    fracs = np.linspace(base_frac, 1.0, max(2, int(np.ceil((1.0 - base_frac) * max_px)) + 1))
    key = np.zeros(shape, np.float64)
    floor = hmin + 0.01
    for f_ in fracs:                                 # steps <= 1 px along each column
        z = hv * f_                                  # true height -> displacement
        nx = np.rint(xs + z * ax / pix).astype(int)
        ny = np.rint(ys + z * ay / pix).astype(int)
        inb = (nx >= 0) & (nx < shape[1]) & (ny >= 0) & (ny < shape[0])
        k = np.rint(np.maximum(z, floor) * 100) * _KEY + (at_top if f_ >= 1.0 else at_side)
        np.maximum.at(key, (ny[inb], nx[inb]), k[inb])
    cov = key > 0                                    # fill isolated 1-px rounding holes
    holes = ~cov & (uniform_filter(cov.astype(np.float32), 3) * 9 - cov >= 5)
    key[holes] = grey_closing(key, size=(3, 3))[holes]
    height = (np.floor(key / _KEY) / 100.0).astype(np.float32)
    att = np.where(key > 0, (key % _KEY) / 1023.0, np.nan).astype(np.float32)
    return height, att


def lean_heights(h, ax, ay, pix, hmin=HMIN, base_frac=CROWN_BASE_FRAC):
    """Leaned surface of tall columns only (0 where nothing tall lands)."""
    return lean_view(h, np.zeros(h.shape, np.float32), ax, ay, pix, hmin, base_frac)[0]


def leaned_chm(h, ax, ay, pix, hmin=HMIN, base_frac=CROWN_BASE_FRAC):
    """Full leaned CHM (no shift): tall columns swept base->top; 0-hmin vegetation stays in place
    (moves <= hmin*|a|); 0 where a lifted crown uncovers ground; NaN where unknown and nothing landed."""
    up = lean_heights(h, ax, ay, pix, hmin, base_frac)
    low = np.where(np.isfinite(h) & (h <= hmin), h, np.where(np.isfinite(h), 0.0, np.nan))
    return np.where(up > 0, up, low).astype(np.float32)


def shift_crop(a, tx, ty, p):
    """Chip seen by NAIP when the CHM rendering (padded by p) is shifted by (tx, ty) px:
    NAIP pixel (i, j) <- rendering (p + i - ty, p + j - tx)."""
    r0, c0 = p - int(ty), p - int(tx)
    return a[..., r0:r0 + CHIP, c0:c0 + CHIP]


# ----------------------------------------------------------------- fitting
class LeanFitter:
    def __init__(self, d, az, el, base_frac=CROWN_BASE_FRAC):
        self.d, self.pix, self.p = d, d["pix"], d["pad"]
        self.base_frac = base_frac
        if self.p < T_MAX_PX + 2:
            raise ValueError("pairs.PAD must exceed T_MAX_PX")
        hp = d["hag_pad"]
        self.unknown_pad = ~np.isfinite(hp)
        self.h0 = np.nan_to_num(hp, nan=0.0).astype(np.float32)
        self.tall = self.h0 > HMIN
        hmax = float(self.h0.max())
        self.hnorm = max(hmax, HMIN + 1.0)
        maxdist = min(hmax / np.tan(np.radians(max(el, 5.0))) + 2.0, self.p * self.pix)
        self.shadow_all = cast_shadow(self.h0, az, el, self.pix, maxdist)
        self.shadow_pad = self.shadow_all & ~self.tall                   # ground shadow
        self.shadow = self.shadow_pad[self.p:self.p + CHIP, self.p:self.p + CHIP]
        self.illum = hillshade(self.h0, az, el, self.pix) * ~self.shadow_all
        self.sun_h = np.array([np.sin(np.radians(az)), -np.cos(np.radians(az))])  # toward sun, image xy
        self.cos_el, self.sin_el = np.cos(np.radians(el)), np.sin(np.radians(el))
        naip = d["naip"]
        R_, G_, B_, NIR = naip
        s = R_ + G_ + B_ + 1e-3
        self.T = np.stack([np.maximum(np.maximum(R_, G_), B_), G_ / s, B_ / s, NIR,
                           (NIR - R_) / (NIR + R_ + 1e-3)], -1)             # (CHIP, CHIP, 5)
        near_pad = binary_dilation(self.tall | self.shadow_pad, iterations=BUFFER_PX + T_MAX_PX)
        # constant scoring support for every (t, a): drop pixels that could see unknown HAG for any
        # searched shift/lean -- comparing R2 over changing pixel subsets would reward shifts that
        # simply push hard pixels out of the window
        reach = T_MAX_PX + int(np.ceil(A_MAX * hmax / self.pix)) + 1
        unk_any = binary_dilation(self.unknown_pad, iterations=reach) if self.unknown_pad.any() else self.unknown_pad
        c = slice(self.p, self.p + CHIP)
        self.R = np.isfinite(naip).all(0) & near_pad[c, c] & ~unk_any[c, c]
        sub = np.zeros((CHIP, CHIP), bool); sub[::SUBSAMPLE, ::SUBSAMPLE] = True
        self.Rs = self.R & sub
        self._Y = {True: self.T[self.Rs], False: self.T[self.R]}
        self._ss = {k: ((v - v.mean(0)) ** 2).sum(0) for k, v in self._Y.items()}
        self._cache = {}

    # rendering in CHM ground coordinates (padded); t is applied later by cropping
    def render(self, ax, ay):
        k = (round(ax, 5), round(ay, 5), self.base_frac)
        if k in self._cache:
            return self._cache[k]
        m = np.hypot(ax, ay)
        side = self.cos_el * max(0.0, -float(np.dot(np.array([ax, ay]) / m, self.sun_h))) if m > 0 else None
        hv, att = lean_view(self.h0, self.illum, ax, ay, self.pix, HMIN, self.base_frac, side)
        crown = hv > HMIN
        vis_sh = self.shadow_pad & ~crown
        height = np.where(crown, hv, np.where(self.tall, 0.0, self.h0))       # ground revealed -> 0
        illum = np.where(crown, att, np.where(self.tall, self.sin_el, self.illum))
        F = np.stack([gaussian_filter(x.astype(np.float32), 1.0)
                      for x in (crown, vis_sh, height / self.hnorm, illum)])
        out = dict(F=F, crown=crown, vis_sh=vis_sh, unknown=self.unknown_pad & ~crown)
        if len(self._cache) > 64:
            self._cache.clear()
        self._cache[k] = out
        return out

    def r2(self, rend, tx, ty, sub=True):
        Rm = self.Rs if sub else self.R
        Fc = shift_crop(rend["F"], tx, ty, self.p)
        X = np.column_stack([Fc[i][Rm] for i in range(Fc.shape[0])] + [np.ones(Rm.sum(), np.float32)])
        Y = self._Y[sub]
        beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
        res = Y - X @ beta
        return 1.0 - (res ** 2).sum(0) / np.maximum(self._ss[sub], 1e-12)          # R2 per target

    def search_t(self, ax, ay, cols, center=(0, 0), half=T_MAX_PX):
        rend = self.render(ax, ay)

        def scan(cx, cy, h, step):
            best = (-np.inf, cx, cy)
            for ty in range(cy - h, cy + h + 1, step):
                for tx in range(cx - h, cx + h + 1, step):
                    if abs(tx) <= T_MAX_PX and abs(ty) <= T_MAX_PX:
                        s = self.r2(rend, tx, ty)[list(cols)].mean()
                        if s > best[0]:
                            best = (s, tx, ty)
            return best
        step = T_COARSE_STEP if half > 2 * T_COARSE_STEP else 1
        s, tx, ty = scan(center[0], center[1], half, step)
        if step > 1:
            s, tx, ty = scan(tx, ty, step, 1)
        return tx, ty, s

    def grid_a(self, tx, ty, cx, cy, half, step):
        g = np.round(np.arange(-half, half + 1e-9, step), 4)
        axs = np.unique(np.clip(cx + g, -A_MAX, A_MAX))
        ays = np.unique(np.clip(cy + g, -A_MAX, A_MAX))
        S = np.array([[self.r2(self.render(a_x, a_y), tx, ty) for a_x in axs] for a_y in ays])
        return axs, ays, S

    def masks(self, ax, ay, tx=0, ty=0):
        rend = self.render(ax, ay)
        return shift_crop(rend["crown"], tx, ty, self.p), shift_crop(rend["vis_sh"], tx, ty, self.p)


def _best(axs, ays, S, cols):
    sc = S[..., list(cols)].mean(-1)
    i, j = np.unravel_index(np.nanargmax(sc), sc.shape)
    return axs[j], ays[i], sc


def describe(ax, ay, pix):
    m = float(np.hypot(ax, ay))
    az = float(np.degrees(np.arctan2(ax, -ay)) % 360) if m > 0 else float("nan")
    return m, az, f"tan(theta)={m:.3f} ({np.degrees(np.arctan(m)):.1f} deg), lean az {az:.0f} | 20 m crown -> {20 * m / pix:.1f} px"


# ----------------------------------------------------------------- outputs
def write_tif(path, arr, d, dtype, nodata):
    prof = dict(driver="GTiff", height=CHIP, width=CHIP, count=1, dtype=dtype, crs=d["crs"],
                transform=d["transform"], nodata=nodata, compress="deflate")
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(arr.astype(dtype), 1)


def write_naip_tif(path, d):
    prof = dict(driver="GTiff", height=CHIP, width=CHIP, count=d["naip"].shape[0], dtype="float32",
                crs=d["crs"], transform=d["transform"], nodata=np.nan, compress="deflate")
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(d["naip"].astype("float32"))


def remap_tch(tch, mapping=None):
    mapping = TCH_REMAP if mapping is None else mapping
    out = tch.copy()
    for src, dst in mapping.items():
        out[tch == src] = dst
    return out


def qa_png(path, d, f, t0, t_best, a_best, coarse):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rgb = np.clip(np.nan_to_num(np.moveaxis(d["naip"][:3], 0, -1)) * 1.4, 0, 1)
    panels = [((0.0, 0.0), (0, 0), "raw CHM"),
              ((0.0, 0.0), t0, f"shift only t=({t0[0]},{t0[1]}) px"),
              (a_best, t_best, f"t=({t_best[0]},{t_best[1]}) px + lean a=({a_best[0]:+.3f},{a_best[1]:+.3f})")]
    axs, ays, sc = coarse
    fig, ax = plt.subplots(1, 4, figsize=(21, 5.4))
    for k, (a, t, title) in enumerate(panels):
        c, v = f.masks(*a, *t)
        ax[k].imshow(rgb)
        ax[k].contour(c.astype(float), [0.5], colors="cyan", linewidths=0.7)
        ax[k].contour(v.astype(float), [0.5], colors="yellow", linewidths=0.7)
        ax[k].set_title(f"{title}\ncyan crowns | yellow visible ground shadow", fontsize=9)
        ax[k].set_xticks([]); ax[k].set_yticks([])
    im = ax[3].imshow(sc, extent=[axs[0], axs[-1], ays[-1], ays[0]])
    ax[3].plot(*a_best, "r*", ms=12); ax[3].plot(0, 0, "wo")
    ax[3].set_xlabel("a_x (east +)"); ax[3].set_ylabel("a_y (south +)")
    ax[3].set_title(f"score over a at t=({t0[0]},{t0[1]}) (coarse)", fontsize=9)
    plt.colorbar(im, ax=ax[3], fraction=0.046)
    plt.tight_layout(); plt.savefig(path, dpi=85); plt.close(fig)


# ----------------------------------------------------------------- fits (one per geometry, cached per chip)
def fit_shift_only(f, cols):
    """Best rigid shift t with no lean (set 2)."""
    tx, ty, s = f.search_t(0.0, 0.0, cols)
    return (tx, ty), float(s)


def fit_lean(f, cols, d, fit_shift, shift_only=None, verbose=True):
    """Lean a, with (fit_shift=True, set 4 = V3) or without (t = 0, set 3) the rigid shift t."""
    flags, info = [], {}
    s_raw = float(f.r2(f.render(0.0, 0.0), 0, 0)[cols].mean())
    t0 = shift_only[0] if (fit_shift and shift_only) else (0, 0)
    axs, ays, S = f.grid_a(*t0, 0.0, 0.0, A_MAX, COARSE_STEP)            # coarse lean at t0
    if verbose:
        for name, oc in OBJECTIVES.items():
            bx, by, sc = _best(axs, ays, S, oc)
            s0 = f.r2(f.render(0.0, 0.0), *t0)[list(oc)].mean()
            print(f"  {name:26s}: no-lean {s0:.4f} | best a=({bx:+.3f},{by:+.3f}) score {sc.max():.4f} | "
                  f"{describe(bx, by, d['pix'])[2]}")
    bx, by, sc_c = _best(axs, ays, S, cols)
    coarse = (axs, ays, sc_c)
    t1 = f.search_t(bx, by, cols)[:2] if fit_shift else (0, 0)            # t at the coarse lean
    mx, my, Sm = f.grid_a(*t1, bx, by, COARSE_STEP, MID_STEP)
    a_mid = _best(mx, my, Sm, cols)[:2]
    fx, fy, Sf = f.grid_a(*t1, *a_mid, 2 * FINE_STEP, FINE_STEP)
    a_best = _best(fx, fy, Sf, cols)[:2]
    if FIT_CROWN_BASE:
        best = (Sf[..., cols].mean(-1).max(), f.base_frac, a_best, fx, fy, Sf)
        for bf in (0.25, 0.5):
            f.base_frac = bf
            gx, gy, Sg = f.grid_a(*t1, *a_best, COARSE_STEP, MID_STEP)
            if Sg[..., cols].mean(-1).max() > best[0]:
                best = (Sg[..., cols].mean(-1).max(), bf, _best(gx, gy, Sg, cols)[:2], gx, gy, Sg)
        _, f.base_frac, a_best, fx, fy, Sf = best
    t_best = f.search_t(*a_best, cols, center=t1, half=3)[:2] if fit_shift else (0, 0)
    if tuple(t_best) != tuple(t1):                                       # lean again at the refined t
        fx, fy, Sf = f.grid_a(*t_best, *a_best, 2 * FINE_STEP, FINE_STEP)
        a_best = _best(fx, fy, Sf, cols)[:2]
    sc_f = Sf[..., cols].mean(-1)
    s_best = float(sc_f.max())
    s_t_only = float(f.r2(f.render(0.0, 0.0), *t_best)[cols].mean())
    info.update(score_raw=s_raw, score_best=s_best, near_best_cells=int((sc_c > sc_c.max() - TOL).sum()))
    if max(abs(a_best[0]), abs(a_best[1])) >= A_MAX - 1e-9:
        flags.append("lean_on_search_boundary")
    if fit_shift and max(abs(t_best[0]), abs(t_best[1])) >= T_MAX_PX:
        flags.append("shift_on_search_boundary")
    if s_best - s_raw < 0.01:
        flags.append("weak_gain_over_raw")
    if fit_shift and np.hypot(*a_best) > 0:                              # can a pure shift explain it?
        hmed = float(np.median(f.h0[f.tall])) if f.tall.any() else 0.0
        eq = (int(round(t_best[0] + hmed * a_best[0] / d["pix"])), int(round(t_best[1] + hmed * a_best[1] / d["pix"])))
        s_eq = f.search_t(0.0, 0.0, cols, center=eq, half=2)[2]
        info["score_equiv_pure_shift"] = float(s_eq)
        if s_best - s_eq < AMBIG_TOL:
            flags.append("lean_vs_shift_ambiguous")
    tol = max(1e-4, TOL_GAIN * (s_best - s_t_only))
    yy, xx = np.nonzero(sc_f > s_best - tol)
    near = [(fx[j], fy[i]) for i, j in zip(yy, xx)]
    if len(near) > 25:
        near = [near[k] for k in np.linspace(0, len(near) - 1, 25).astype(int)]
    return dict(t=tuple(int(v) for v in t_best), a=(float(a_best[0]), float(a_best[1])), near=near,
                flags=flags, info=info, coarse=coarse, t0=t0)


# ----------------------------------------------------------------- write one set
def write_set(set_dir, stem, d, f, t, a, near, lean, flags):
    os.makedirs(set_dir, exist_ok=True)
    p = d["pad"]
    naip_bad = ~np.isfinite(d["naip"]).all(0)
    if lean:
        hag = shift_crop(leaned_chm(d["hag_pad"], *a, d["pix"], HMIN, f.base_frac), *t, p).copy()
        crown, vis_sh = f.masks(*a, *t)
        vote = np.mean([f.masks(ax_, ay_, *t)[0] for ax_, ay_ in near], axis=0)
        uncertain = np.where(crown, vote < VOTE[1], vote > VOTE[0])
        if uncertain.mean() > 0.5 * max(crown.mean(), 1e-6):
            flags = flags + ["wide_lean_uncertainty"]
    else:                                           # raw / shifted LiDAR HAG, exactly as resampled
        hag = shift_crop(d["hag_pad"], *t, p).copy()
        crown = np.nan_to_num(hag, nan=0.0) > HMIN
        vis_sh = shift_crop(f.shadow_pad, *t, p) & ~crown
        if EDGE_IGNORE_PX > 0:
            uncertain = (binary_dilation(crown, iterations=EDGE_IGNORE_PX)
                         & ~binary_erosion(crown, iterations=EDGE_IGNORE_PX))
        else:
            uncertain = np.zeros_like(crown)
    lab = np.full((CHIP, CHIP), LAB_GROUND, np.uint8)
    lab[vis_sh] = LAB_SHADOW
    lab[crown] = LAB_CANOPY
    lab[uncertain] = LAB_UNCERTAIN
    lab[naip_bad | (~np.isfinite(hag) & ~crown)] = LAB_UNKNOWN
    hag[naip_bad] = np.nan

    paths = dict(naip=os.path.join(set_dir, f"{stem}_naip.tif"),
                 hag=os.path.join(set_dir, f"{stem}_hag_leaned.tif" if lean else f"{stem}_hag.tif"),
                 mask=os.path.join(set_dir, f"{stem}_canopy.tif"),
                 tch=os.path.join(set_dir, f"{stem}_tch.tif") if d["tch"] is not None else "")
    if WRITE_NAIP:
        write_naip_tif(paths["naip"], d)
    write_tif(paths["hag"], hag, d, "float32", np.nan)
    write_tif(paths["mask"], lab, d, "uint8", None)
    if d["tch"] is not None:
        tch = remap_tch(d["tch"])
        tch[naip_bad] = 0
        write_tif(paths["tch"], tch, d, "uint8", 0)
    fr = lambda v: float((lab == v).mean())
    tan_t, lean_az, _ = describe(*a, d["pix"])
    rec = dict(tx_px=int(t[0]), ty_px=int(t[1]), shift_m=float(np.hypot(*t) * d["pix"]),
               ax=float(a[0]), ay=float(a[1]), tan_theta=tan_t, lean_az=lean_az, base_frac=f.base_frac,
               canopy_frac=fr(LAB_CANOPY), shadow_frac=fr(LAB_SHADOW), uncertain_frac=fr(LAB_UNCERTAIN),
               unknown_frac=fr(LAB_UNKNOWN), flags=";".join(flags), status="ok",
               out_naip=paths["naip"] if WRITE_NAIP else "", out_hag=paths["hag"], out_mask=paths["mask"],
               out_tch=paths["tch"])
    return rec


def active_sets():
    if EXPORT_SETS is None:
        name = {(False, False): "1_raw", (True, False): "2_shift", (False, True): "3_lean",
                (True, True): "4_shift_lean"}[(FIT_SHIFT, FIT_LEAN)]
        return [(name, FIT_SHIFT, FIT_LEAN)]
    return list(EXPORT_SETS)


# ----------------------------------------------------------------- one row -> every set
def process_row(row, out_dir, qa=False, verbose=True):
    stem = os.path.splitext(os.path.basename(str(row["naip"])))[0]
    base = dict(src_naip=row["naip"], src_hag=row["hag"], src_label=row["label"], SDATE=row["SDATE"], stem=stem)
    d = load_pair(row["naip"], row["hag"], row["label"])
    if verbose:
        tch_shape = d["tch"].shape if d["tch"] is not None else None
        print(f"Loaded pair: NAIP {d['naip'].shape}, HAG {d['hag'].shape}, TCH {tch_shape}")
    ts = parse_sdate(row["SDATE"], SDATE_TZ)
    if check_DOY_and_SDATE:
        m_doy = re.search(r"doy_(\d{1,3})", stem)
        if m_doy and int(m_doy.group(1)) != ts.tz_convert(SDATE_TZ).dayofyear:
            raise SkipPair(f"SDATE day-of-year {ts.tz_convert(SDATE_TZ).dayofyear} != file name doy_{m_doy.group(1)}")
    az, el = solar_az_el(ts, d["lat"], d["lon"])
    base.update(sdate_utc=str(ts), lat=d["lat"], lon=d["lon"], pix_m=d["pix"], sun_az=az, sun_el=el,
                naip_unknown_frac=float((~np.isfinite(d["naip"]).all(0)).mean()),
                hag_unknown_frac=float((~np.isfinite(d["hag"])).mean()))
    if verbose:
        print(f"Sun: {ts:%Y-%m-%d %H:%M} UTC, lat {d['lat']:.3f}, lon {d['lon']:.3f}: az {az:.1f}, el {el:.1f}")
    if el < 10:
        raise SkipPair(f"sun elevation {el:.1f} deg < 10: shadows too long/unreliable")

    f = LeanFitter(d, az, el)
    cols = list(OBJECTIVES[USE_OBJECTIVE])
    enough = f.R.sum() >= MIN_SCORE_PX
    base["score_px"] = int(f.R.sum())
    cache, recs = {}, {}
    for name, fit_s, fit_l in active_sets():
        flags, info, near = [], {}, [(0.0, 0.0)]
        t, a = (0, 0), (0.0, 0.0)
        if not enough:
            flags.append("too_few_tall_or_shadow_px")
        elif fit_s and "shift" not in cache:
            cache["shift"] = fit_shift_only(f, cols)
        if enough and not fit_l:
            if fit_s:
                t, s_t = cache["shift"]
                info = dict(score_raw=float(f.r2(f.render(0.0, 0.0), 0, 0)[cols].mean()), score_best=s_t)
                if max(abs(t[0]), abs(t[1])) >= T_MAX_PX:
                    flags.append("shift_on_search_boundary")
        elif enough and fit_l:
            key = ("lean", fit_s)
            if key not in cache:
                if verbose:
                    print(f"[{name}] fitting lean{' + shift' if fit_s else ' (t = 0)'}")
                cache[key] = fit_lean(f, cols, d, fit_s, cache.get("shift"), verbose)
            res = cache[key]
            t, a, near, info = res["t"], res["a"], res["near"], res["info"]
            flags += res["flags"]
            if qa:
                qa_png(os.path.join(out_dir, name, f"{stem}_lean_qa.png"), d, f, res["t0"], t, a, res["coarse"])
        rec = write_set(os.path.join(out_dir, name), stem, d, f, t, a, near, lean=fit_l, flags=flags)
        rec.update(base, set=name, **info)
        recs[name] = rec
        if verbose:
            print(f"[{name:13s}] t=({t[0]:+d},{t[1]:+d}) px  a=({a[0]:+.4f},{a[1]:+.4f}) | canopy "
                  f"{100 * rec['canopy_frac']:.1f}% shadow {100 * rec['shadow_frac']:.1f}% "
                  f"uncertain {100 * rec['uncertain_frac']:.1f}% unknown {100 * rec['unknown_frac']:.1f}%"
                  + (f" | FLAGS: {rec['flags']}" if rec["flags"] else ""))
    return recs


def _safe_row(args):
    k, n, row, out_dir, qa, verbose = args
    t0 = time.time()
    if verbose:
        print(f"\n[{k + 1}/{n}] {row['naip']}")
    try:
        recs = process_row(row, out_dir, qa=qa, verbose=verbose)
    except (SkipPair, ValueError, rasterio.errors.RasterioError) as e:
        if verbose:
            print(f"SKIP: {e}")
        recs = {name: dict(src_naip=row.get("naip"), src_hag=row.get("hag"), src_label=row.get("label"),
                           SDATE=row.get("SDATE"), set=name, status=f"skip: {e}") for name, _, _ in active_sets()}
    for r in recs.values():
        r["seconds"] = round(time.time() - t0, 1)
    return recs


def _save(out_dir, all_recs):
    for name, _, _ in active_sets():
        df = pd.DataFrame([r[name] for r in all_recs if name in r])
        if df.empty:
            continue
        df.to_csv(os.path.join(out_dir, name, f"lean_summary_{name}.csv"), index=False)
        ok = df[df["status"].eq("ok")]
        if len(ok):
            ok.rename(columns={"out_naip": "naip", "out_hag": "hag", "out_mask": "mask", "out_tch": "tch"})[
                ["naip", "hag", "mask", "tch", "stem", "src_naip", "SDATE"]].to_csv(
                os.path.join(out_dir, name, f"pairs_{name}.csv"), index=False)


def run(csv_path=CSV_PATH, out_dir=OUT_DIR, n_jobs=1):
    """Writes every set in active_sets() under out_dir/<set>/. n_jobs > 1 = parallel processes."""
    df = read_pair_table(csv_path)
    for name, _, _ in active_sets():
        os.makedirs(os.path.join(out_dir, name), exist_ok=True)
    print("export sets:", ", ".join(f"{n} (shift={s}, lean={l})" for n, s, l in active_sets()),
          "| TCH remap:", TCH_REMAP or "none")
    jobs = [(k, len(df), row, out_dir, k < QA_PNG_FIRST_N, n_jobs == 1) for k, row in df.iterrows()]
    all_recs = []
    if n_jobs == 1:
        for j in jobs:
            all_recs.append(_safe_row(j))
            _save(out_dir, all_recs)                                   # checkpoint after every row
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            for i, recs in enumerate(ex.map(_safe_row, jobs, chunksize=4)):
                all_recs.append(recs)
                if (i + 1) % 50 == 0 or i + 1 == len(jobs):
                    _save(out_dir, all_recs)
                    print(f"{i + 1}/{len(jobs)} done")
    _save(out_dir, all_recs)
    for name, _, _ in active_sets():
        st = [r[name]["status"] for r in all_recs if name in r]
        n_ok = sum(s == "ok" for s in st)
        print(f"{name}: {n_ok} ok, {len(st) - n_ok} skipped -> {os.path.join(out_dir, name)}")


# ----------------------------------------------------------------- utilities for existing output folders
def remap_tch_folder(folder, mapping=None):
    """Apply TCH_REMAP in place to every *_tch.tif in a folder (e.g. your V2 / V3 outputs),
    so all ablation sets use the same TCH classes. Safe to run twice."""
    import glob
    mapping = TCH_REMAP if mapping is None else mapping
    files = sorted(glob.glob(os.path.join(folder, "*_tch.tif")))
    n_px = 0
    for fp in files:
        with rasterio.open(fp) as s:
            a, prof = s.read(1), s.profile.copy()
        b = remap_tch(a, mapping)
        n_px += int((a != b).sum())
        with rasterio.open(fp, "w", **prof) as dst:
            dst.write(b, 1)
    print(f"{folder}: {len(files)} TCH files remapped {mapping}, {n_px} pixels changed")


def pairs_csv_from_folder(folder, name=None):
    """Write pairs_<name>.csv (naip, hag, mask, tch, stem) for an existing output folder such as
    your V2 (set 3) or V3 (set 4) runs, matching <stem>_naip / _hag(_leaned) / _canopy / _tch.tif."""
    import glob
    name = name or os.path.basename(os.path.normpath(folder))
    rows = []
    for fp in sorted(glob.glob(os.path.join(folder, "*_naip.tif"))):
        stem = os.path.basename(fp)[:-len("_naip.tif")]
        hag = next((h for h in (os.path.join(folder, f"{stem}_hag_leaned.tif"), os.path.join(folder, f"{stem}_hag.tif"))
                    if os.path.exists(h)), "")
        msk, tch = os.path.join(folder, f"{stem}_canopy.tif"), os.path.join(folder, f"{stem}_tch.tif")
        if hag and os.path.exists(msk) and os.path.exists(tch):
            rows.append(dict(naip=fp, hag=hag, mask=msk, tch=tch, stem=stem))
    out = os.path.join(folder, f"pairs_{name}.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"{out}: {len(rows)} complete chips")
    return out


if __name__ == "__main__":
    # python lean_labels.py [pairs.csv out_dir [n_jobs]]   export the sets in EXPORT_SETS
    # python lean_labels.py remap_tch <folder>             apply TCH_REMAP to an existing folder's *_tch.tif
    # python lean_labels.py make_pairs <folder>            write pairs_<folder>.csv for an existing folder
    import sys
    a = sys.argv[1:]
    if a and a[0] == "remap_tch":
        for fo in a[1:]:
            remap_tch_folder(fo)
    elif a and a[0] == "make_pairs":
        for fo in a[1:]:
            pairs_csv_from_folder(fo)
    else:
        run(a[0] if len(a) > 0 else CSV_PATH, a[1] if len(a) > 1 else OUT_DIR, int(a[2]) if len(a) > 2 else 1)
