Keep the published RealICU directory structure:

```text
realicu/
    trajectories/events.parquet
    trajectories/icu_stay.parquet
    gold/labels/<stay_id>.json
    scale/labels/<stay_id>.json
```

Each run explicitly chooses `data.root` and `data.label_set` in its configuration.
Set `data.stay_ids` to null for all published stays or provide a list of stay IDs.
Evaluation uses `labels_dir` and does not need the trajectory files.

Clinical data are not bundled with this software. See the root README for the
installation and usage. Only the published trajectory and annotation format is supported.
