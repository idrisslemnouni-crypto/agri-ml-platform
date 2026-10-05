"""Next-day measured-soil proxy classification with spatial and temporal holdouts."""

import hashlib
import json
import logging
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from agri_platform.data import FEATURES, features, read_sources


def evaluate(y, probability, threshold):
    prediction = probability >= threshold
    return {
        "positive_f1": float(f1_score(y, prediction, zero_division=0)),
        "precision": float(precision_score(y, prediction, zero_division=0)),
        "recall": float(recall_score(y, prediction, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "pr_auc": float(average_precision_score(y, probability)),
        "prevalence": float(np.mean(y)),
        "n": len(y),
        "positive_n": int(np.sum(y)),
        "false_alarms": int(((prediction == 1) & (y == 0)).sum()),
    }


def split_data(table, config):
    known = table.station.isin(config["train_stations"])
    train = table[known & (table.target_date <= config["train_target_end"])].copy()
    val = table[known & (table.target_date.dt.year == config["validation_target_year"])].copy()
    test = table[
        (table.station == config["holdout_station"])
        & (table.target_date >= config["test_target_start"])
    ].copy()
    if any(t.empty for t in [train, val, test]):
        raise ValueError("Empty split")
    if not train.target_date.max() < val.target_date.min() < test.target_date.min():
        raise ValueError("Time overlap")
    if set(train.station) & set(test.station):
        raise ValueError("Holdout station leaked into training")
    return train, val, test


def run(root: Path):
    config = json.loads((root / "configs/default.json").read_text())
    raw, evidence = read_sources(root)
    table = features(raw, 1).dropna(subset=["target", "SOIL_MOISTURE_10_DAILY"])
    table["label"] = (table.target < config["proxy_threshold_m3_m3"]).astype(int)
    train, val, test = split_data(table, config)
    candidates = {
        "logistic": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=42),
        ),
        "random_forest": make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(
                n_estimators=160,
                max_depth=10,
                min_samples_leaf=5,
                class_weight="balanced",
                n_jobs=2,
                random_state=42,
            ),
        ),
    }
    validation, test_metrics, thresholds = {}, {}, {}
    for name, model in candidates.items():
        model.fit(train[FEATURES], train.label)
        p = model.predict_proba(val[FEATURES])[:, 1]
        thresholds[name] = max(
            config["probability_thresholds"],
            key=lambda t: evaluate(val.label.to_numpy(), p, t)["positive_f1"],
        )
        validation[name] = evaluate(val.label.to_numpy(), p, thresholds[name])
    baseline_val = (
        (val.SOIL_MOISTURE_10_DAILY < config["proxy_threshold_m3_m3"]).astype(float).to_numpy()
    )
    validation["persistence"] = evaluate(val.label.to_numpy(), baseline_val, 0.5)
    chosen = max(validation, key=lambda n: validation[n]["positive_f1"])
    # Selection is frozen before calculating held-out probabilities.
    baseline = (
        (test.SOIL_MOISTURE_10_DAILY < config["proxy_threshold_m3_m3"]).astype(float).to_numpy()
    )
    test_probabilities = {"persistence": baseline}
    for name, model in candidates.items():
        test_probabilities[name] = model.predict_proba(test[FEATURES])[:, 1]
    for name, p in test_probabilities.items():
        test_metrics[name] = evaluate(test.label.to_numpy(), p, thresholds.get(name, 0.5))
    reports = root / "reports"
    (reports / "figures").mkdir(parents=True, exist_ok=True)
    output = {
        "config": config,
        "data": evidence,
        "retained_rows": len(table),
        "selection_metric": "validation positive-class F1",
        "selected_on_validation": chosen,
        "thresholds_from_validation": thresholds,
        "validation": validation,
        "test": test_metrics,
        "units": "Soil moisture m3/m3; label is an exploratory threshold proxy, not observed irrigation or crop stress",
        "split_sizes": {"train": len(train), "validation": len(val), "test": len(test)},
        "feature_names": FEATURES,
    }
    artifact = {
        "model": candidates.get(chosen),
        "kind": chosen,
        "probability_threshold": thresholds.get(chosen, 0.5),
        "soil_proxy_threshold": config["proxy_threshold_m3_m3"],
        "features": FEATURES,
    }
    (root / "models").mkdir(exist_ok=True)
    joblib.dump(artifact, root / "models/selected.joblib")
    output["model_sha256"] = hashlib.sha256(
        (root / "models/selected.joblib").read_bytes()
    ).hexdigest()
    (reports / "metrics.json").write_text(json.dumps(output, indent=2))
    prediction = test[["station", "origin", "target_date", "target", "label"]].copy()
    for name, p in test_probabilities.items():
        prediction[name + "_probability"] = p
    prediction["selected_prediction"] = (
        test_probabilities[chosen] >= artifact["probability_threshold"]
    ).astype(int)
    prediction.to_csv(reports / "test-predictions.csv", index=False)
    sample = {
        name: float(test[name].iloc[0]) if np.isfinite(test[name].iloc[0]) else None
        for name in FEATURES
    }
    (root / "configs/example-input.json").write_text(json.dumps(sample, indent=2))
    fig, ax = plt.subplots(figsize=(6, 5), layout="constrained")
    ConfusionMatrixDisplay.from_predictions(
        test.label,
        prediction.selected_prediction,
        display_labels=["Not below proxy", "Below proxy"],
        cmap="Blues",
        ax=ax,
        colorbar=False,
    )
    ax.set_title("Unseen Illinois station Â· 2023â€“2024 Â· no irrigation labels")
    fig.savefig(reports / "figures/confusion-matrix.png", dpi=150)
    plt.close(fig)
    selected_model = candidates.get(chosen)
    if selected_model is not None:
        importance = permutation_importance(
            selected_model,
            test[FEATURES],
            test.label,
            n_repeats=5,
            random_state=42,
            scoring="average_precision",
            n_jobs=2,
        )
        order = np.argsort(importance.importances_mean)[-8:]
        fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
        ax.barh(
            np.array(FEATURES)[order],
            importance.importances_mean[order],
            xerr=importance.importances_std[order],
            color="#276b57",
        )
        ax.set_xlabel("Decrease in held-out average precision after permutation")
        ax.set_title("Correlated predictors limit interpretation; not causation")
        fig.savefig(reports / "figures/permutation-importance.png", dpi=150)
        plt.close(fig)
    logging.info("Selected %s, test F1 %.4f", chosen, test_metrics[chosen]["positive_f1"])
    return output


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = run(Path.cwd())
    print(
        json.dumps({"selected": result["selected_on_validation"], "test": result["test"]}, indent=2)
    )
