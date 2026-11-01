# Configuration

- `datasets/ohr_split.json` is the immutable OHR-Bench split.
- `environment/` contains pinned dependency constraints.
- `model_revisions.json` locks model repositories and immutable revisions.
- `pilots/` holds one config per frozen sample. `scripts/data/build_pilot.py` builds each sample once into `results/pilots/<pilot_id>/`. Do not edit a config whose sample exists. Create a new `pilot_id` instead.
- `split_checksums.json` locks the OHR split and QA source hashes.
- `gate_threshold.json` is populated only from an approved validation B0 run.
