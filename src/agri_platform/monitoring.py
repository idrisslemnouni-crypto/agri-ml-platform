"""Local telemetry and diagnostic population stability, not accuracy monitoring."""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def distributions(frame, names):
    result = {}
    for name in names:
        values = frame[name].dropna().to_numpy()
        if len(values) < 30:
            continue
        boundaries = np.unique(np.quantile(values, np.linspace(0.1, 0.9, 9)))
        edges = np.r_[-np.inf, boundaries, np.inf]
        counts = np.histogram(values, bins=edges)[0]
        result[name] = {
            "boundaries": boundaries.tolist(),
            "proportions": (counts / counts.sum()).tolist(),
        }
    return result


def psi(values, reference):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 30:
        return None
    edges = np.r_[-np.inf, reference["boundaries"], np.inf]
    current = np.histogram(values, bins=edges)[0].astype(float)
    current = np.maximum(current / current.sum(), 1e-6)
    baseline = np.maximum(reference["proportions"], 1e-6)
    current /= current.sum()
    baseline /= baseline.sum()
    return float(np.sum((current - baseline) * np.log(current / baseline)))


def connect(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path, timeout=10)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute(
        "CREATE TABLE IF NOT EXISTS predictions (id INTEGER PRIMARY KEY, timestamp TEXT, release_id TEXT, probability REAL, elapsed_ms REAL, source TEXT, features TEXT)"
    )
    return con


def record(path, release_id, probability, elapsed_ms, source, features):
    with connect(path) as con:
        con.execute(
            "INSERT INTO predictions(timestamp,release_id,probability,elapsed_ms,source,features) VALUES (?,?,?,?,?,?,?)",
            (
                datetime.now(timezone.utc).isoformat(),
                release_id,
                probability,
                elapsed_ms,
                source,
                json.dumps(features, allow_nan=False),
            ),
        )


def summarize(path, reference, release_id):
    with connect(path) as con:
        total = con.execute(
            "SELECT COUNT(*) FROM predictions WHERE release_id=?", (release_id,)
        ).fetchone()[0]
        rows = con.execute(
            "SELECT probability,elapsed_ms,source,features FROM predictions WHERE release_id=? ORDER BY id DESC LIMIT 300",
            (release_id,),
        ).fetchall()
    counts = {}
    for row in rows:
        counts[row[2]] = counts.get(row[2], 0) + 1
    if not rows:
        return {
            "total_accepted": total,
            "window_n": 0,
            "source_counts": counts,
            "feature_psi": {},
            "scope": "No accepted requests; no drift evidence",
        }
    features = [json.loads(r[3]) for r in rows]
    drift = {
        name: psi([f.get(name) if f.get(name) is not None else np.nan for f in features], ref)
        for name, ref in reference.items()
    }
    latency = np.array([r[1] for r in rows])
    return {
        "total_accepted": total,
        "window_n": len(rows),
        "source_counts": counts,
        "latency_p50_ms": float(np.median(latency)),
        "latency_p95_ms": float(np.quantile(latency, 0.95)),
        "feature_psi": drift,
        "heuristic_shift_alerts": [
            name for name, value in drift.items() if value is not None and value > 0.2
        ],
        "scope": "PSI is a population-shift diagnostic; no observed live target or accuracy-drift measurement",
    }
