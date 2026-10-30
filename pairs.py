"""Read NAIP / HAG / TCH triplets listed in a CSV and put them on one grid.

Grid = a CHIP x CHIP window of the NAIP raster (NAIP's CRS, pixel size and
extent). HAG and TCH are reprojected onto that grid, so they may come in any
CRS / resolution / extent. HAG is also read with a PAD-pixel margin around the
chip, because trees just outside the chip cast shadows into it and lean into it.

Unknown handling (NaN or the file's nodata value):
  NAIP  -> NaN (a pixel is unknown if any band is NaN, or all bands == nodata)
  HAG   -> NaN
  TCH   -> 0   (legend: 0 = unknown)
"""
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import Affine
from rasterio.warp import Resampling, reproject, transform as warp_transform
from rasterio.windows import Window

CHIP = 256
PAD = 96                    # margin (px) read around the chip for HAG (~58 m at 0.6 m)
TCH_UNKNOWN = 0             # TCH legend: 0 unknown, 1 red, 2 yellow, 3 green, 4 gray, 5 shadow, 6 dark green
HAG_NODATA = None           # extra HAG nodata value to honour, or None (NaN and file nodata always honoured)
HAG_RESAMPLING = Resampling.bilinear   # continuous heights; use .nearest to keep crown edges hard
CSV_COLUMNS = ("naip", "hag", "label", "SDATE")


class SkipPair(Exception):
    """A row that cannot be used; the message is the reason."""


def _blank(v):
    return v is None or (isinstance(v, float) and np.isnan(v)) or str(v).strip() == ""


def read_pair_table(csv_path):
    df = pd.read_csv(csv_path)
    missing = [c for c in CSV_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path}: missing column(s) {missing}; found {list(df.columns)}")
    return df


def _pixel_size_m(crs, transform, lat):
    """Pixel size in metres (NAIP is usually UTM; geographic CRS handled approximately)."""
    if crs.is_geographic:
        return abs(transform.a) * 111_320.0 * np.cos(np.radians(lat))
    try:
        _, factor = crs.linear_units_factor
    except Exception:
        factor = 1.0
    return abs(transform.a) * factor


# def _read_naip(src, win):
#     arr = src.read(window=win).astype(np.float32)
#     if arr.shape[0] < 4:
#         raise SkipPair(f"NAIP has {arr.shape[0]} band(s); need 4 (R, G, B, NIR)")
#     arr = arr[:4]
#     bad = ~np.isfinite(arr).all(0)
#     if src.nodata is not None and np.isfinite(src.nodata):
#         bad |= (arr == src.nodata).all(0)
#     dt = np.dtype(src.dtypes[0])
#     if np.issubdtype(dt, np.integer):
#         scale = float(np.iinfo(dt).max)
#     else:
#         scale = 255.0 if np.nanmax(arr) > 1.5 else 1.0
#     arr = arr / scale
#     arr[:, bad] = np.nan
#     return arr                                   # (4, CHIP, CHIP), 0-1, NaN = unknown

def _read_naip(src, win):
    # arr = src.read(window=win).astype(np.float32)
    # 20261003_modification: do not skip NAIP images smaller than the chip size.
    # Read the complete requested window, padding portions outside the raster.
    arr = src.read(window=win, boundless=True, fill_value=0).astype(np.float32)
    if arr.shape[0] < 4:
        raise SkipPair(f"NAIP has {arr.shape[0]} band(s); need 4 (R, G, B, NIR)")
    arr = arr[:4]
    bad = ~np.isfinite(arr).all(0)
    if src.nodata is not None and np.isfinite(src.nodata):
        bad |= (arr == src.nodata).all(0)
    dt = np.dtype(src.dtypes[0])
    if np.issubdtype(dt, np.integer):
        scale = float(np.iinfo(dt).max)
    else:
        scale = 255.0 if np.nanmax(arr) > 1.5 else 1.0
    arr = arr / scale
    # arr[:, bad] = np.nan
    arr[:, bad] = 0.0
    return arr                                   # (4, CHIP, CHIP), 0-1, 0 = unknown




def _reproject_band(path, dst_transform, dst_crs, shape, resampling, extra_nodata=None):
    dst = np.full(shape, np.nan, dtype=np.float32)
    with rasterio.open(path) as s:
        src = s.read(1).astype(np.float32)
        for nd in (s.nodata, extra_nodata):
            if nd is not None and np.isfinite(nd):
                src[src == nd] = np.nan
        reproject(src, dst, src_transform=s.transform, src_crs=s.crs,
                  dst_transform=dst_transform, dst_crs=dst_crs,
                  src_nodata=np.nan, dst_nodata=np.nan, resampling=resampling)
    return dst


def load_pair(naip_path, hag_path, tch_path=None, chip=CHIP, pad=PAD):
    """Returns a dict with
       naip      (4, chip, chip) float32, 0-1, NaN = unknown
       hag_pad   (chip+2*pad, chip+2*pad) float32 metres, NaN = unknown
       hag       (chip, chip) view of hag_pad
       tch       (chip, chip) uint8, 0 = unknown   (or None)
       transform, crs, pix (m), lat, lon (chip centre), pad
    """
    if _blank(naip_path):
        raise SkipPair("NAIP path missing")
    if _blank(hag_path):
        raise SkipPair("HAG path missing")
    with rasterio.open(naip_path) as src:
        t = src.transform
        if t.b != 0 or t.d != 0 or t.e >= 0:
            raise SkipPair("NAIP is not north-up; rotated grids are not supported")
        if src.width < chip or src.height < chip:
            # 20261003_modification: do not skip NAIP images smaller than the chip size.
            # raise SkipPair(f"NAIP smaller than {chip}x{chip} ({src.width}x{src.height})")
            # Extend the naip size to (CHIP, CHIP), fill the missing pixels with all 0 values instead.
            win = Window((src.width - chip) // 2 if src.width > chip else 0,
                         (src.height - chip) // 2 if src.height > chip else 0, chip, chip)
        else:
            win = Window((src.width - chip) // 2, (src.height - chip) // 2, chip, chip)
        chip_t = src.window_transform(win)
        crs = src.crs
        naip = _read_naip(src, win)


    cx, cy = chip_t * (chip / 2, chip / 2)
    lon, lat = (v[0] for v in warp_transform(crs, "EPSG:4326", [cx], [cy]))
    pix = _pixel_size_m(crs, chip_t, lat)

    pad_t = chip_t * Affine.translation(-pad, -pad)
    n_pad = chip + 2 * pad
    hag_pad = _reproject_band(hag_path, pad_t, crs, (n_pad, n_pad), HAG_RESAMPLING, HAG_NODATA)
    hag_pad[np.isfinite(hag_pad) & (hag_pad < 0)] = 0.0          # small negative HAG = ground

    tch = None
    if not _blank(tch_path):
        t_f = _reproject_band(tch_path, chip_t, crs, (chip, chip), Resampling.nearest)
        tch = np.where(np.isfinite(t_f), t_f, TCH_UNKNOWN).astype(np.uint8)

    return dict(naip=naip, hag_pad=hag_pad, hag=hag_pad[pad:pad + chip, pad:pad + chip],
                tch=tch, transform=chip_t, crs=crs, pix=float(pix),
                lat=float(lat), lon=float(lon), pad=pad)
