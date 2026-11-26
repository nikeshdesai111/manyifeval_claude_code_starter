# K10 split check (pool metadata only)

Split: S3, 22 TEST / 50 TRAIN groups; K10 pools have [10] rules. Source: pilots/high_k/data/inputs/pools.jsonl and prompts.parquet (split, tasks); task families from the frozen manifest. No generations, labels or features were read.

```json
{
 "groups": {
  "test": 22,
  "train": 50
 },
 "rules_per_pool": [
  10
 ],
 "split_matches_s3_assignment": true,
 "a_type_test_min_max": [
  6,
  22
 ],
 "a_type_train_min_max": [
  15,
  50
 ],
 "b_n_compatible_pairs": 98,
 "b_pairs_in_any_pool": 98,
 "b_test_min": 2,
 "b_train_min": 1,
 "b_pairs_test_lt2": [],
 "b_pairs_train_lt1": [],
 "b_pairs_test_ge3": 89,
 "b_conflicting_pairs_inside_a_pool": 0,
 "c_exact_rules": {
  "distinct": 291,
  "test": 103,
  "train": 213,
  "in_both_splits": 25
 },
 "c_exact_pairs": {
  "distinct": 2217,
  "in_test": 742,
  "in_train": 1592,
  "in_both_splits": 117,
  "max_groups_per_exact_pair": 72
 },
 "d_family_L1_gap": 0.0253,
 "d_type_L1_gap": 0.0215,
 "d_score_10rule": 0.0467,
 "d_score_6rule_s3": 0.0871,
 "e_tasks_in_both_splits": 0,
 "e_related_units_split": []
}
```

## (a) Types

| Type | TEST groups | TRAIN groups |
|---|---|---|
| change_case:capital_word_frequency | 6 | 17 |
| change_case:english_capital | 8 | 18 |
| change_case:english_lowercase | 8 | 15 |
| detectable_content:number_placeholders | 22 | 50 |
| detectable_format:number_bullet_lists | 22 | 50 |
| detectable_format:title | 22 | 50 |
| keywords:existence | 11 | 25 |
| keywords:forbidden_words | 22 | 50 |
| keywords:frequency | 11 | 25 |
| keywords:letter_frequency | 22 | 50 |
| length_constraints:number_paragraphs | 8 | 17 |
| length_constraints:number_sentences | 8 | 15 |
| length_constraints:number_words | 6 | 18 |
| punctuation:no_comma | 22 | 50 |
| startend:quotation | 22 | 50 |

Per-pair and exact-rule rows: split_check_k10.csv.
