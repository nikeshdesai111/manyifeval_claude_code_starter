"""S3 TEST evaluation, Objective 1 (authorised by the project owner on 2026-11-24, see results/eval_s3/AUTHORIZATION.json).

Modules:
  authorization  the unblinding record every TEST reader checks
  extract        one forward pass per S3 TEST prompt (A1max + selected hidden rows), original frozen backend
  verify         official verifier on the expansion responses (original responses reuse run `main`)
  features       TEST feature blocks (S1, EP, ER, H, A1) in the V2 layouts
  predict        frozen V2 S3 refits applied to TEST
  eval1          Evaluation 1, Objective 1 (Table 2, A1-A4, per-type CSV)
  eval2          Evaluation 2, Objective 1 (position, strict append, companion, task context; Table 4)
  report         markdown report with embedded figures
"""
