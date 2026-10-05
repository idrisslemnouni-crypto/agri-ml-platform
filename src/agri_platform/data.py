"""NOAA USCRN daily observations with frozen source hashes and calendar alignment."""

import hashlib
import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd


def read_sources(root: Path) -> tuple[pd.DataFrame, dict]:
    raw = root / "data/raw"
    raw.mkdir(parents=True, exist_ok=True)
    names = (root / "data/source-headers.txt").read_text().splitlines()[1].split()
    frames = []
    for item in json.loads((root / "data/source-manifest.json").read_text()):
        path = raw / item["file"]
        if not path.exists():
            temporary = path.with_suffix(".part")
            with urllib.request.urlopen(item["url"], timeout=60) as response:
                temporary.write_bytes(response.read())
            temporary.replace(path)
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError("NOAA source was revised; review provenance before changing the hash")
        frame = pd.read_csv(path, sep=r"\s+", names=names, header=None)
        if len(frame.columns) != 28:
            raise ValueError("Unexpected NOAA schema")
        frame["station"] = item["station"]
        frames.append(frame)
    full = pd.concat(frames, ignore_index=True)
    full["date"] = pd.to_datetime(full.LST_DATE.astype(str), format="%Y%m%d")
    if full.duplicated(["station", "date"]).any():
        raise ValueError("Duplicate station/day")
    numeric = [n for n in names[5:] if n in full.select_dtypes("number").columns]
    full[numeric] = full[numeric].mask(full[numeric] <= -99)
    moisture = [n for n in names if n.startswith("SOIL_MOISTURE")]
    for name in moisture:
        full[name] = full[name].where(full[name].between(0, 1))
    for name in ["RH_DAILY_MIN", "RH_DAILY_MAX", "RH_DAILY_AVG"]:
        full[name] = full[name].where(full[name].between(0, 100))
    for name in ["P_DAILY_CALC", "SOLARAD_DAILY"]:
        full[name] = full[name].where(full[name] >= 0)
    evidence = {
        "observed_rows": len(full),
        "stations": sorted(full.station.unique()),
        "missing_fraction": {
            n: float(full[n].isna().mean())
            for n in [
                "SOIL_MOISTURE_5_DAILY",
                "SOIL_MOISTURE_10_DAILY",
                "P_DAILY_CALC",
                "T_DAILY_AVG",
            ]
        },
        "years": sorted(full.date.dt.year.unique().tolist()),
    }
    # Reindex each calendar before shifts: a seven-row shift must mean seven days.
    aligned = []
    calendar = pd.date_range(full.date.min(), full.date.max(), freq="D")
    for station, group in full.groupby("station"):
        group = group.set_index("date").sort_index().reindex(calendar)
        group.index.name = "date"
        group["station"] = station
        aligned.append(group.reset_index())
    return pd.concat(aligned, ignore_index=True), evidence


def features(frame: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    if not isinstance(horizon, int) or horizon < 1:
        raise ValueError("Positive integer forecast horizon required")
    groups = []
    for _station, group in frame.groupby("station"):
        group = group.sort_values("date").copy()
        if (
            group.date.duplicated().any()
            or not group.date.diff().dropna().eq(pd.Timedelta(days=1)).all()
        ):
            raise ValueError("Features require a unique uninterrupted daily calendar")
        group["origin"] = group.date
        group["target_date"] = group.date + pd.Timedelta(days=horizon)
        group["target"] = group.SOIL_MOISTURE_10_DAILY.shift(-horizon)
        for name in [
            "SOIL_MOISTURE_5_DAILY",
            "SOIL_MOISTURE_10_DAILY",
            "P_DAILY_CALC",
            "T_DAILY_AVG",
            "RH_DAILY_AVG",
            "SOLARAD_DAILY",
        ]:
            group[name + "_mean7"] = group[name].rolling(7, min_periods=7).mean()
        group["soil10_lag1"] = group.SOIL_MOISTURE_10_DAILY.shift(1)
        group["soil10_lag7"] = group.SOIL_MOISTURE_10_DAILY.shift(7)
        angle = 2 * np.pi * (group.date.dt.dayofyear - 1) / 365.25
        group["doy_sin"], group["doy_cos"] = np.sin(angle), np.cos(angle)
        groups.append(group)
    return pd.concat(groups, ignore_index=True)


FEATURES = [
    "SOIL_MOISTURE_5_DAILY",
    "SOIL_MOISTURE_10_DAILY",
    "P_DAILY_CALC",
    "T_DAILY_AVG",
    "RH_DAILY_AVG",
    "SOLARAD_DAILY",
    "soil10_lag1",
    "soil10_lag7",
    "doy_sin",
    "doy_cos",
] + [
    n + "_mean7"
    for n in [
        "SOIL_MOISTURE_5_DAILY",
        "SOIL_MOISTURE_10_DAILY",
        "P_DAILY_CALC",
        "T_DAILY_AVG",
        "RH_DAILY_AVG",
        "SOLARAD_DAILY",
    ]
]
