# Linguistic Markers of Dependency Formation in Longitudinal Human-AI Conversations

[Read paper](https://github.com/orvelte/human-AI-dependency-prediction/blob/main/Linguistic%20Markers%20of%20Dependency%20Formation%20in%20Longitudinal%20Human.pdf)

A computational NLP study identifying early-session linguistic features that predict dependency-adjacent interaction patterns in later sessions of human-AI chat. Using the WildChat-1M corpus, extracted lexical, pragmatic, and session-level features from users' first three sessions and tested the ability to classify users who exhibit high-dependency behaviour in sessions 10 + beyond.

---

## Research Question

Can linguistic features extracted from early user sessions (sessions 1–3) predict dependency-adjacent interaction patterns in late sessions (session 10+) in longitudinal human-AI chat logs?

---

## Theoretical Framing

Dependency-adjacent patterns are framed as the behavioural surface of the **goal-directed → habitual transition** — the same striatal dopamine-mediated process documented in substance use disorders (Dickinson, 1985; Everitt & Robbins, 2005). This grounds feature selection in a mechanistic process model rather than pure text classification.

Key operationalised behaviours:
- Return interval (time between sessions)
- Session length (user turns per session)
- First-person plural language ("we", "us", "our")
- Self-disclosure and relational register
- Resistance to AI refusals (rephrasing rather than disengaging)
- Vocabulary narrowing (type–token ratio)

---

## Dataset

**Source:** WildChat-1M (Zhao et al., 2024) — publicly available, no ethics approval required  
**Acquisition:** Streamed from Hugging Face, filtered to longitudinal users during download  
**Cohort:** Users with ≥ 10 conversations retained  
**Final dataset:** 12,577 users · 505,397 conversations · Apr 2023 – Apr 2024

---

## Results

### Dependency label
A user is classified as **high-dependency** in late sessions (10+) if they meet ≥ 2 of 4 pre-specified criteria (return interval, session length, refusal resistance, FPP rate). **24.1% of users (n = 3,014)** meet this threshold.

### Univariate baselines

| Feature | AUC | Cohen's d | BH-significant |
|---|---|---|---|
| Session length (log turns) | 0.659 | +0.591 | Yes |
| Refusal resistance rate | 0.540 | +0.372 | Yes |
| First-person plural rate | 0.575 | +0.272 | Yes |
| Vocabulary richness (TTR) | 0.562 | −0.241 | Yes |
| Return interval (log hours) | 0.545 | −0.221 | Yes |
| Filler / phatic rate | 0.599 | +0.173 | Yes |
| Hedging rate | 0.561 | +0.194 | Yes |
| Question–statement ratio | 0.571 | +0.216 | Yes |
| Self-disclosure rate | 0.510 | +0.054 | Yes |
| Sentiment slope | 0.507 | | No |
| Mean message length | 0.475 | | No |

9 of 12 features are BH-corrected significant. Effect sizes confirm theoretically predicted directions: high-dependency users show more relational language, faster returns, and narrower vocabulary in early sessions.

### Main model (L1-regularised logistic regression)

| Metric | Value |
|---|---|
| Mean AUC (10-fold CV) | **0.712 ± 0.014** |
| Fold range | 0.688 – 0.737 |
| Balanced accuracy | 0.661 |
| Precision / Recall | 0.402 / 0.611 |

Cross-validation is **user-level stratified** — no user appears in both train and test sets.

### Robustness

- **Permutation test** (1,000 shuffles): p < 0.001 — result is not spurious
- **Negative control** (punctuation rate): rank 14/15, coefficient −0.018 — model is not learning noise
- **Sensitivity** (label threshold ≥1–≥3): top-3 feature ranking stable; AUC range 0.690–0.748

---

## Feature Engineering

All features are extracted from user turns only (assistant turns excluded). Extraction uses a single-pass token scan for speed (~6 min for 500k conversations).

**Lexical:** First-person plural rate · Hedging rate · Self-disclosure marker rate · Filler/phatic rate  
**Pragmatic:** Refusal resistance rate · Gratitude/deference rate · Question-to-statement ratio  
**Session-level:** Type–token ratio · Mean message length · Return interval · Sentiment slope

Log transforms are pre-applied to session length and return interval (right-skewed distributions).

---

## Reproducing the Analysis

```bash
# Install dependencies
pip install -r requirements.txt

# Feature extraction (~6 min on a standard laptop)
python src/extract_features.py

# Run notebooks in order
jupyter notebook
# → 01_data_exploration.ipynb
# → 02_feature_engineering.ipynb
# → 03_modelling.ipynb
# → 04_robustness_checks.ipynb
```

The dataset (`data/interim/wildchat_longitudinal.parquet`) must be acquired separately via the WildChat-1M Hugging Face dataset, filtered to users with ≥ 10 conversations.

---

## File Structure

```
├── .pre-analysis-plan.md         # Hypotheses written before any modelling
├── requirements.txt
│
├── data/
│   ├── interim/                  # Filtered parquet (not committed)
│   └── processed/                # Feature matrix (not committed)
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_modelling.ipynb
│   └── 04_robustness_checks.ipynb
│
├── src/
│   ├── extract_features.py       # Standalone fast feature extraction script
│   ├── data/
│   │   ├── loader.py
│   │   ├── filter.py
│   │   └── session_builder.py
│   ├── features/
│   │   ├── lexical.py
│   │   ├── pragmatic.py
│   │   ├── session_metrics.py
│   │   └── pipeline.py
│   ├── modelling/
│   │   ├── baseline.py
│   │   ├── main_model.py
│   │   └── evaluation.py
│   └── utils/
│       ├── text_utils.py
│       └── stats_utils.py
│
├── outputs/
│   ├── figures/                  # All publication figures (PNG + PDF)
│   └── tables/                   # baseline_results.csv, feature_importances.csv
│
└── tests/
    ├── test_features.py
    └── test_pipeline.py
```

---

## Scientific Integrity

- Analysis plan and hypotheses were **pre-specified** before any modelling (see `.pre-analysis-plan.md`)
- FDR correction (Benjamini–Hochberg) applied across all feature-level tests
- Effect sizes (Cohen's d) reported alongside all significance tests
- Permutation test and negative control are required stopping points before any claims

---

## References

- Dickinson, A. (1985). Actions and habits. *Philosophical Transactions of the Royal Society B*, 308, 67–78.
- Everitt, B. J., & Robbins, T. W. (2005). Neural systems of reinforcement for drug addiction. *Nature Neuroscience*, 8(11), 1481–1489.
- Zhao, W. X., et al. (2024). WildChat: 1M ChatGPT interaction logs in the wild. *ICLR 2024*.
