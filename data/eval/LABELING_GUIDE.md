# AI Resume Analyzer — Evaluation Benchmark Labeling Guide

This document defines the formal annotation rubric and scoring methodology for the 105-pair evaluation dataset partitioned into development (`data/eval/dev.json`, $N=50$) and held-out test (`data/eval/test.json`, $N=55$) sets.

---

## 1. Objective & Design Principles

The benchmark evaluates how accurately different matching architectures (Keyword Jaccard, Sparse TF-IDF, Pure Dense Embeddings, and the Dual-Signal Hybrid Matcher) assess candidate suitability against real-world job requisitions.

To eliminate data leakage:
- **`dev.json`** ($N=50$): Used strictly for metric calibration, weight tuning, and threshold selection.
- **`test.json`** ($N=55$): Held-out gold standard evaluated solely for final performance reporting (MAE, Spearman $\rho$, Precision@K, NDCG@K).

---

## 2. Multi-Component Ground-Truth Scoring Rubric

Each pair is assigned a calibrated continuous suitability score between **0.0 and 100.0**, calculated via three orthogonal criteria:

### A. Required & Preferred Skill Overlap ($0 - 50$ points)
- **Mandatory Required Skills (Max 35 points)**:
  - 100% of core required skills present: 35 points
  - 75 - 99% of core skills present: 27 - 34 points
  - 50 - 74% of core skills present: 18 - 26 points
  - 25 - 49% of core skills present: 9 - 17 points
  - < 25% of core skills present: 0 - 8 points
- **Preferred / Bonus Skills (Max 15 points)**:
  - 100% of preferred skills present: 15 points
  - Partial preferred skills present: proportional points (e.g. 5 - 12 points)
  - 0 preferred skills present: 0 points

### B. Semantic & Domain Context Alignment ($0 - 40$ points)
- **Direct Domain Fit (30 - 40 points)**: Candidate has demonstrated track record in the exact industry subfield, architecture patterns, and domain problem space.
- **Adjacent / Transferable Fit (15 - 29 points)**: Candidate has strong software engineering fundamentals and related technical experience, but in an adjacent subdiscipline.
- **Irrelevant / Distractor Domain (0 - 14 points)**: Candidate experience is fundamentally disconnected from the target role.

### C. Career Tenure Fit ($0 - 10$ points)
- Meets or exceeds required years of experience: **10 points**
- 70% to 99% of required tenure: **6 - 8 points**
- 50% to 69% of required tenure: **3 - 5 points**
- < 50% of required tenure: **0 - 2 points**

$$\text{Ground Truth Score} = \text{Skill Overlap (50)} + \text{Domain Fit (40)} + \text{Tenure Fit (10)}$$

---

## 3. Qualitative Verdict Thresholds

| Match Category | Score Range | Description |
|----------------|-------------|-------------|
| **High Match** | $75.0 - 100.0$ | Meets all mandatory skills, strong domain track record, and adequate tenure. |
| **Moderate Match** | $45.0 - 74.9$ | Solid core foundation; minor skill gaps or slightly less tenure than ideal. |
| **Low / Distractor Match** | $0.0 - 44.9$ | Substantial gaps in core prerequisites, cross-domain mismatches, or inadequate tenure. |

---

## 4. Evaluation Metrics

1. **Mean Absolute Error (MAE)**: Measures absolute percentage calibration error against gold labels (lower is better):
   $$\text{MAE} = \frac{1}{N} \sum_{i=1}^N |y_i - \hat{y}_i|$$
2. **Spearman Rank Correlation ($\rho$)**: Measures monotonicity of candidate ranking across diverse applicant pools (higher is better):
   $$\rho = 1 - \frac{6 \sum d_i^2}{N(N^2 - 1)}$$
3. **Precision@K (P@3, P@5)**: Proportion of top-$K$ recommended candidates that belong to the true **High Match** category ($\ge 75\%$).
4. **Normalized Discounted Cumulative Gain (NDCG@K)**: Measures ranking quality by penalizing relevant candidates placed lower in the rankings.
