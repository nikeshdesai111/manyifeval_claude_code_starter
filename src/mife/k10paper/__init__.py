"""K10 (k = 1..10) re-run of the paper's exhibits on the 50 S3 TRAIN groups (grouped-CV out-of-fold).

Inputs: pilots/high_k/data (frozen K10 design, generations, verifier labels and Betty CUDA features; PRESPEC.md).
Code here is a snapshot of mife.probing.v2 (fitting) and mife.prelim_train (evaluation) re-pointed at K10 TRAIN data.
TEST prompts (22 S3 TEST groups) are never read: every loader filters split == "train" first and asserts it.
Outputs: data/k10_paper/ (feature caches) and results/k10_paper/ (selections, OOF predictions, tables, figures, paper).
"""
