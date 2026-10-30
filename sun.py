"""Sun azimuth / elevation from the NAIP survey time (CSV column SDATE).

NOAA solar-position equations (the spreadsheet algorithm; ~0.01 deg vs. pvlib
for 1950-2050), no extra dependency. Azimuth: degrees clockwise from north.
Elevation: degrees above horizon, with NOAA's atmospheric-refraction correction.
"""
import numpy as np
import pandas as pd

SDATE_TZ = "America/Chicago"   # NAIP times are local unless SDATE carries its own offset.
                               # (The 2019-08-20 15:34 example only matches the NE-pointing
                               #  shadows if 15:34 is CDT, not UTC.)


def parse_sdate(value, tz=SDATE_TZ):
    """SDATE -> tz-aware UTC Timestamp. Accepts '2019-08-20 15:34', '8/20/2019 15:34',
    ISO strings with or without an offset. Raises ValueError if no time of day."""
    if value is None or (isinstance(value, float) and np.isnan(value)) or str(value).strip() == "":
        raise ValueError("SDATE missing")
    s = str(value).strip()
    ts = pd.Timestamp(s)
    if ":" not in s and ts.hour == 0 and ts.minute == 0:
        raise ValueError(f"SDATE has a date but no time of day: {s!r} (sun position needs the time)")
    if ts.tzinfo is None:
        ts = ts.tz_localize(tz)
    return ts.tz_convert("UTC")


def solar_az_el(ts_utc, lat, lon):
    """NOAA solar position. ts_utc: tz-aware Timestamp; lat, lon in degrees (lon east +)."""
    t = pd.Timestamp(ts_utc).tz_convert("UTC").tz_localize(None)
    jc = (t.to_julian_date() - 2451545.0) / 36525.0
    L0 = (280.46646 + jc * (36000.76983 + jc * 0.0003032)) % 360.0
    M = 357.52911 + jc * (35999.05029 - 0.0001537 * jc)
    e = 0.016708634 - jc * (0.000042037 + 0.0000001267 * jc)
    Mr = np.radians(M)
    C = (np.sin(Mr) * (1.914602 - jc * (0.004817 + 0.000014 * jc))
         + np.sin(2 * Mr) * (0.019993 - 0.000101 * jc) + np.sin(3 * Mr) * 0.000289)
    omega = np.radians(125.04 - 1934.136 * jc)
    app_long = L0 + C - 0.00569 - 0.00478 * np.sin(omega)
    obliq = 23 + (26 + (21.448 - jc * (46.815 + jc * (0.00059 - jc * 0.001813))) / 60) / 60
    obliq_c = np.radians(obliq + 0.00256 * np.cos(omega))
    decl = np.arcsin(np.sin(obliq_c) * np.sin(np.radians(app_long)))
    y = np.tan(obliq_c / 2) ** 2
    L0r = np.radians(L0)
    eq_time = 4 * np.degrees(y * np.sin(2 * L0r) - 2 * e * np.sin(Mr)
                             + 4 * e * y * np.sin(Mr) * np.cos(2 * L0r)
                             - 0.5 * y * y * np.sin(4 * L0r) - 1.25 * e * e * np.sin(2 * Mr))
    minutes_utc = t.hour * 60 + t.minute + t.second / 60.0
    tst = (minutes_utc + eq_time + 4 * lon) % 1440.0
    ha = tst / 4.0 - 180.0 if tst / 4.0 >= 0 else tst / 4.0 + 180.0
    latr, har = np.radians(lat), np.radians(ha)
    cos_zen = np.clip(np.sin(latr) * np.sin(decl) + np.cos(latr) * np.cos(decl) * np.cos(har), -1, 1)
    zen = np.arccos(cos_zen)
    el = 90.0 - np.degrees(zen)
    # NOAA refraction correction
    te = np.tan(np.radians(el))
    if el > 85:
        refr = 0.0
    elif el > 5:
        refr = 58.1 / te - 0.07 / te ** 3 + 0.000086 / te ** 5
    elif el > -0.575:
        refr = 1735 + el * (-518.2 + el * (103.4 + el * (-12.79 + el * 0.711)))
    else:
        refr = -20.772 / te
    el += refr / 3600.0
    cos_az = np.clip((np.sin(latr) * np.cos(zen) - np.sin(decl)) / (np.cos(latr) * np.sin(zen)), -1, 1)
    acos_az = np.degrees(np.arccos(cos_az))
    az = (acos_az + 180.0) % 360.0 if ha > 0 else (540.0 - acos_az) % 360.0
    return float(az), float(el)
