# AutoML smoke datasets

`synthetic.py` generates deterministic CSV data when each smoke case starts. Each of the
regression, binary classification, multiclass classification, and timeseries datasets
contains 150 unique rows. Tabular rows have a unique `row_id`; timeseries rows have
unique monthly timestamps for one `series_id`.

The fixtures upload the generated CSV to the test DSPA's MinIO using a temporary
ConfigMap and uploader Pod. No external dataset bucket or AWS credentials are needed
for AutoML smoke or upgrade tests. Upgrade tests keep their namespace, DSPA, runs,
and baseline ConfigMaps across the upgrade; post-upgrade checks remove the namespace.
