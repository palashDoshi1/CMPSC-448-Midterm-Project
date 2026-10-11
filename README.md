
### RQ3 — Cross-Domain Generalization (Extra Credit)

We investigated whether CNN and LSTM classifiers trained on five source categories could identify the generating LLM in a previously unseen category.

**Held-out category:** `general_writing`

The five source categories were:

- `cs_software`
- `finance_banking`
- `math_logic`
- `sports`
- `stocks_trading`

For this experiment, fresh classifiers were trained without any general-writing responses in training or validation. Experiment-specific vocabularies were constructed using source-domain training responses only.

Both architectures were evaluated on all **90 general-writing responses**, with 30 responses per LLM family.

#### CNN — Source vs. Unseen Domain

| Evaluation Set | Responses | Accuracy | Macro-F1 |
|---|---:|---:|---:|
| Source-domain test | 66 | 89.39% | 0.8932 |
| Unseen general writing | 90 | 72.22% | 0.7161 |

The CNN's accuracy decreased by **17.17 percentage points** on the unseen writing category.

The CNN selected its best RQ3 checkpoint at epoch 6 using source-domain validation macro-F1.

#### CNN vs. LSTM — Unseen-Domain Results

| Model | Responses | Accuracy | Macro-F1 |
|---|---:|---:|---:|
| **CNN** | 90 | **72.22%** | **0.7161** |
| LSTM | 90 | 56.67% | 0.5429 |

The CNN outperformed the LSTM by **15.56 percentage points** in accuracy and approximately **0.1732** in macro-F1.

The LSTM exhibited a strong tendency to predict Gemini. On the unseen domain, Gemini recall was 0.93, compared with 0.43 for GPT and 0.33 for Claude.

The LSTM achieved 51.52% accuracy on its 66-response source-domain test set and 56.67% on the unseen writing domain. Thus, unlike the CNN, the LSTM did not show an accuracy decrease between these two evaluation sets.

These results suggest that the CNN generalized more effectively to the held-out general-writing category under the evaluated configurations. However, this experiment covers only one unseen domain and does not establish generalization across all possible task categories.

### RQ4 — Writing-Style Analysis (Extra Credit)

We conducted a descriptive writing-style analysis of all **540 responses**, with 180 responses per LLM family.

The analysis examined response length, lexical diversity, punctuation and symbol usage, headings, lists, code fences, and bold formatting.

#### Writing-Style Measurements

| Measurement | GPT | Claude | Gemini |
|---|---:|---:|---:|
| Mean response length (words) | 459.27 | 629.71 | 328.78 |
| Mean lexical diversity (MATTR-50) | 0.8592 | 0.8259 | 0.8625 |
| Punctuation/symbols per 100 words | 26.53 | 26.00 | 41.25 |
| Responses with heading markers (%) | 30.56 | 0.56 | 39.44 |
| Responses with list markers (%) | 19.44 | 61.67 | 33.33 |
| Responses with code fences (%) | 5.00 | 0.00 | 23.89 |
| Responses with bold markers (%) | 25.56 | 19.44 | 46.67 |

#### Main Observations

**Claude**
- Produced the longest responses on average, approximately 630 words.
- Used list formatting most frequently, in 61.67% of responses.
- Had the lowest mean MATTR-50 lexical-diversity score.

**Gemini**
- Produced the shortest responses on average, approximately 329 words.
- Used heading markers most frequently (39.44%).
- Used code fences most frequently (23.89%).
- Had the highest punctuation/symbol count per 100 words.

**GPT**
- Had an intermediate average response length of approximately 459 words.
- Used heading markers in 30.56% of responses.
- Had mean lexical diversity close to Gemini.

Lexical diversity was measured using the moving-average type-token ratio with 50-word windows (MATTR-50), which reduces the length sensitivity of ordinary type-token ratio.

The analysis reveals measurable differences in the writing styles of the three LLM families within our dataset.

However, these descriptive patterns do not prove which features the CNN or LSTM used internally. No feature-ablation or statistical-significance experiments were performed.

## Repository Structure

```text
CMPSC-448-Midterm-Project/
├── README.md
├── CNN.ipynb
├── LSTM.ipynb
└── data/
    ├── dataset.csv
    ├── train.csv
    ├── validation.csv
    ├── test.csv
    └── vocab.json
```

### Notebook Contents

**CNN.ipynb**
- RQ1: Output-only CNN classification
- RQ2: Input-only, output-only, and combined experiments
- RQ3: CNN cross-domain generalization
- RQ4: Descriptive writing-style analysis

**LSTM.ipynb**
- RQ1: Output-only LSTM classification
- RQ2: Input-only, output-only, and combined experiments
- RQ3: LSTM cross-domain generalization

The notebooks generate classification metrics, confusion matrices, training histories, and experimental outputs.

## Key Findings

1. **CNN outperformed LSTM in RQ1:** 82.72% vs. 71.60% test accuracy.

2. **Output-only classification performed best in RQ2** for both architectures.

3. **Input-only classification achieved 33.33% accuracy**, consistent with the three-class chance baseline.

4. **CNN accuracy decreased from 89.39% to 72.22%** when evaluated on unseen general-writing responses.

5. **CNN outperformed LSTM on the unseen domain**, achieving 72.22% accuracy compared with 56.67%.

6. **Writing-style analysis revealed substantial descriptive differences**, particularly in response length, Markdown formatting, and punctuation.

7. **The findings are dataset-specific** and do not establish reliable authorship attribution across every LLM version or task domain.
