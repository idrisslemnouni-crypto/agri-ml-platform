# Local verification — 5 October 2026

The fixed source experiment, validation gate, actual MLflow run and checksum-bound release executed. Five meaningful tests passed: absent-release status, physical input contract, PSI sample requirements, changed-model rejection, and actual service/release/telemetry. The local browser loaded the model, submitted one real example and displayed the accepted request count. The screenshot is from the working application. Forty separate historical replay requests produced the stored monitoring report.

A clean Git clone in a fresh Python 3.12 platform environment independently downloaded all 27 public NOAA files and reran prepare. Full model metrics match exactly and per-day probabilities match within atol=rtol=1e-10. A distinct finished MLflow run was generated. Five clone tests, pip check and a clean 40-request API replay passed. Runtime-dependent latency and run IDs are expected to differ. No source cache or model was copied into the clone.

Ruff lint/format and the executed notebook passed. Docker is unavailable locally; remote build and smoke results are recorded in [GitHub Actions](https://github.com/idrisslemnouni-crypto/agri-ml-platform/actions). There is no public deployment, production load benchmark or independent irrigation validation.

## Source-isolated monitoring — 6 October 2026

Eight local tests passed, including the real checksum-bound release and API inference. A 30-request balanced manual population retains identical counts, latency and zero PSI after 305 shifted replay requests; the replay window is capped at 300 after source filtering. Empty populations, other release IDs and unsupported API source values are covered. Ruff and notebook checks passed. Existing experiment files and model bytes are unchanged. Local Docker execution remains unverified; inspect the exact-commit CI for remote container evidence.
