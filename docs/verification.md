# Local verification — 5 October 2026

The fixed source experiment, validation gate, actual MLflow run and checksum-bound release executed. Five meaningful tests passed: absent-release status, physical input contract, PSI sample requirements, changed-model rejection, and actual service/release/telemetry. The local browser loaded the model, submitted one real example and displayed the accepted request count. The screenshot is from the working application. Forty separate historical replay requests produced the stored monitoring report.

A clean Git clone in a fresh Python 3.12 platform environment independently downloaded all 27 public NOAA files and reran prepare. Full model metrics match exactly and per-day probabilities match within atol=rtol=1e-10. A distinct finished MLflow run was generated. Five clone tests, pip check and a clean 40-request API replay passed. Runtime-dependent latency and run IDs are expected to differ. No source cache or model was copied into the clone.

Ruff lint/format and the executed notebook passed. Docker is unavailable locally; its build and smoke CI is supplied but pending actual GitHub execution. There is no public deployment, production load benchmark or independent irrigation validation.
