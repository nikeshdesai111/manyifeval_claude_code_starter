"""PRELIMINARY S3 TRAIN report (2026-11-24 course submission). Interim workflow only.

Snapshot of the S3 TEST Objective 1 evaluation code (mife.eval_s3, copied 2026-11-24 ~22:20 ET) re-pointed at the
50 S3 TRAIN groups and the V2 grouped 5-fold out-of-fold (OOF) predictions in results/v2_s3/predictions/. Nothing here
reads TEST prompts, labels, features or predictions; mife.eval_s3 and the final TEST plan are untouched.
Outputs: results/preliminary_train_report/. Run: PYTHONPATH=src .venv/bin/python -m mife.prelim_train.run_all
"""
