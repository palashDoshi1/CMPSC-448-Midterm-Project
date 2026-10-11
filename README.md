# Who Wrote It? Identifying Large Language Models from Their Responses

**CMPSC 448 — Machine Learning and Artificial Intelligence**  
**Midterm Project | Fall 2026**  
**The Pennsylvania State University**

**Team Members:** Palash Doshi (Team Leader) and Siddharth Pendyala  
**Submission Date:** October 11, 2026  
**GitHub Repository:** https://github.com/palashDoshi1/CMPSC-448-Midterm-Project

## Project Overview

This project investigates whether neural-network classifiers can identify which large language model (LLM) generated a text response. We compare two architectures:

- **Convolutional Neural Network (CNN)**
- **Long Short-Term Memory (LSTM)**

We classify responses from three LLM families: **GPT, Claude, and Gemini**. During dataset collection, we used **GPT-6 Instant**, **Claude Sonnet 5.5**, and **Gemini 1.5 Pro**.

The project is organized into **three Jupyter notebooks**: a CNN notebook for RQ1–RQ3, an LSTM notebook for RQ1–RQ3, and a separate notebook for **RQ4 writing-style analysis**, developed jointly by both team members. RQ4 uses descriptive statistics rather than training an additional classifier.

## Research Questions

### RQ1 — LLM Identification

Can CNN and LSTM classifiers identify which model generated a response using only the response text? Which architecture performs better?

### RQ2 — Input vs. Output vs. Combined

Does classification improve when using the original prompt (**input-only**), the generated response (**output-only**), or the prompt and response together (**combined input/output**)?

### RQ3 — Cross-Domain Generalization (Extra Credit)

Can models trained on responses from five subject categories identify LLMs in the sixth, previously unseen category? We hold out **general writing** from training and validation.

### RQ4 — Writing-Style Analysis (Extra Credit)

What measurable characteristics distinguish GPT, Claude, and Gemini responses? We analyze response length, lexical diversity, punctuation, heading markers, list markers, code fences, and bold formatting in a **standalone RQ4 notebook**.

## Dataset

We constructed an original, balanced dataset containing **180 prompts and 540 generated responses**. Each prompt was submitted to all three LLMs, resulting in **180 responses per model family**.

### Dataset Categories

| Category | Prompts | GPT | Claude | Gemini | Total responses |
|---|---:|---:|---:|---:|---:|
| Computer Science and Software | 30 | 30 | 30 | 30 | 90 |
| Finance and Banking | 30 | 30 | 30 | 30 | 90 |
| Mathematics and Logic | 30 | 30 | 30 | 30 | 90 |
| Sports | 30 | 30 | 30 | 30 | 90 |
| Stocks and Trading | 30 | 30 | 30 | 30 | 90 |
| General Writing | 30 | 30 | 30 | 30 | 90 |
| **Total** | **180** | **180** | **180** | **180** | **540** |

Each record contains five fields:

| Field | Description |
|---|---|
| `prompt_id` | Identifier shared by the three responses to one prompt |
| `category` | Subject category |
| `model` | Generating family: GPT, Claude, or Gemini |
| `input` | Original user prompt |
| `output` | Generated response |

The `model` field is the target label for supervised classification.

### Train, Validation, and Test Splits

For RQ1 and RQ2, we used **prompt-grouped splitting** so that all three responses to a prompt belong to the same partition. This prevents the same original prompt from appearing in different train, validation, and test sets.

| Split | Prompts | Responses | Percentage |
|---|---:|---:|---:|
| Training | 126 | 378 | 70% |
| Validation | 27 | 81 | 15% |
| Testing | 27 | 81 | 15% |
| **Total** | **180** | **540** | **100%** |

The primary CNN and LSTM experiments used the same partitions and **random seed 42**.

## Text Preprocessing

Both neural-network classifiers converted text into numerical token sequences using the following settings:

- Case-sensitive tokenization that separates words and punctuation
- Vocabulary constructed from training data, limited to **10,000 tokens**
- Padding token `<PAD>` at index **0** and unknown token `<UNK>` at index **1**
- Maximum sequence length of **512 tokens**, with padding and truncation
- Class labels: **GPT = 0**, **Claude = 1**, **Gemini = 2**

For RQ2, we tested three representations:

1. **Input-only:** original prompt
2. **Output-only:** generated response
3. **Combined input/output:** prompt, `[SEP]` token, and response

The combined representation allocated **up to 128 prompt tokens + 1 separator token + up to 383 response tokens**, for a total maximum of 512 tokens.

For RQ3, we trained new models and constructed separate vocabularies using **source-domain training data only**, excluding general writing.

## Model Architectures and Training

### Convolutional Neural Network (CNN)

**Primary implementation:** Palash Doshi

The TextCNN uses token embeddings, three parallel one-dimensional convolutional branches, padding-aware global max pooling, dropout, and a fully connected three-class output layer.

| Component | Configuration |
|---|---|
| Embedding dimension | 128 |
| Convolutional kernel sizes | 3, 5, 7 |
| Filters per kernel size | 128 |
| Concatenated feature dimension | 384 |
| Pooling | Padding-aware global max pooling |
| Dropout | 0.3 |
| Output classes | 3 |

For RQ1, the best CNN checkpoint was selected by **validation loss** at **epoch 9**.

### Long Short-Term Memory (LSTM)

**Primary implementation:** Siddharth Pendyala

The LSTM uses an embedding layer, a single unidirectional LSTM with packed padded sequences, dropout after the LSTM representation, and a three-class output layer.

| Component | Configuration |
|---|---|
| Embedding dimension | 128 |
| LSTM layers | 1 |
| Hidden dimension | 128 |
| Directionality | Unidirectional |
| Sequence handling | Packed padded sequences |
| Classification features | Final hidden state |
| Dropout | 0.3 |
| Output classes | 3 |

For RQ1, the best LSTM checkpoint was selected by **validation macro-F1** at **epoch 5**.

### Shared Training Configuration

| Hyperparameter | CNN | LSTM |
|---|---|---|
| Framework | PyTorch | PyTorch |
| Optimizer | Adam | Adam |
| Learning rate | 0.001 | 0.001 |
| Loss function | Cross-entropy | Cross-entropy |
| Batch size | 16 | 16 |
| Maximum epochs | 20 | 20 |
| Early stopping patience | 4 | 4 |
| Maximum sequence length | 512 | 512 |
| Random seed | 42 | 42 |
| RQ1 checkpoint criterion | Validation loss | Validation macro-F1 |

We evaluated the models using **accuracy**, **precision**, **recall**, **macro-F1**, and **confusion matrices**. Training and validation curves were used to examine learning behavior.

## Experimental Results

### RQ1 — LLM Identification Using Responses

Both models were evaluated on the same held-out set of **81 responses**, with 27 responses per LLM family.

| Model | Test accuracy | Macro-F1 | Correct predictions |
|---|---:|---:|---:|
| **CNN** | **82.72%** | **0.8293** | **67/81** |
| LSTM | 71.60% | 0.7141 | 58/81 |
| Chance baseline | 33.33% | — | — |

The CNN outperformed the LSTM by approximately **11.11 percentage points** in test accuracy. Both architectures performed above the three-class chance baseline. This comparison applies to our dataset and training configurations; it does not establish that CNNs are universally superior to LSTMs.

### RQ2 — Input-Only, Output-Only, and Combined Input

| Input configuration | CNN accuracy | CNN macro-F1 | LSTM accuracy | LSTM macro-F1 |
|---|---:|---:|---:|---:|
| Input-only | 33.33% | 0.3078 | 33.33% | 0.3281 |
| **Output-only** | **82.72%** | **0.8293** | **71.60%** | **0.7141** |
| Combined input/output | 77.78% | 0.7730 | 53.09% | 0.5294 |

**Output-only classification performed best for both architectures.** Input-only accuracy was at the 33.33% chance baseline, consistent with submitting the same prompts to all three LLMs. Combining prompts with responses did not improve performance.

One possible contributing factor is that the combined representation retains at most **383 response tokens**, compared with up to **512 tokens** in the output-only configuration. We did not perform a separate ablation experiment to isolate this explanation.

### RQ3 — Cross-Domain Generalization

For RQ3, we trained new CNN and LSTM models on responses from five source categories: computer science and software, finance and banking, mathematics and logic, sports, and stocks and trading. **General writing was excluded from training and validation.**

| Partition | Responses |
|---|---:|
| Source-domain training | 315 |
| Source-domain validation | 69 |
| Source-domain testing | 66 |
| Unseen general-writing testing | 90 |

Both architectures used the same source-domain partitions and were evaluated on all **90 general-writing responses** (30 per LLM family).

| Evaluation | CNN accuracy | CNN macro-F1 | LSTM accuracy | LSTM macro-F1 |
|---|---:|---:|---:|---:|
| Source-domain test | 89.39% | 0.8932 | 51.52% | 0.4890 |
| **Unseen general writing** | **72.22%** | **0.7161** | 56.67% | 0.5429 |
| Accuracy change (source to unseen) | **−17.17 pp** | — | **+5.15 pp** | — |

The CNN correctly classified **65 of 90** unseen writing responses, compared with **51 of 90** for the LSTM. The CNN performed better overall, although its accuracy decreased relative to its source-domain evaluation. The LSTM showed a small source-to-target increase from a substantially lower starting accuracy.

These source-to-target comparisons are **within RQ3**. RQ1 and RQ3 used different training configurations, so differences between those experiments should not be attributed solely to domain shift.

### RQ4 — Writing-Style Analysis (Separate Notebook)

Both team members jointly developed a standalone RQ4 notebook to examine linguistic and structural characteristics across **all 540 responses**. This analysis is descriptive and does not train a classifier.

| Measurement | GPT | Claude | Gemini |
|---|---:|---:|---:|
| Mean response length (words) | 459.27 | 629.71 | 328.78 |
| Mean lexical diversity (MATTR-50) | 0.8592 | 0.8259 | 0.8625 |
| Punctuation/symbols per 100 words | 26.53 | 26.00 | 41.25 |
| Responses with heading markers | 30.56% | 0.56% | 39.44% |
| Responses with list markers | 19.44% | 61.67% | 33.33% |
| Responses with code fences | 5.00% | 0.00% | 23.89% |
| Responses with bold markers | 25.56% | 19.44% | 46.67% |

**MATTR-50** is the moving-average type-token ratio calculated with 50-word windows.

Claude produced the longest responses and most frequently used list markers. Gemini produced the shortest responses and showed the highest frequencies of heading markers, code fences, bold markers, and punctuation/symbols per 100 words. GPT generally fell between Claude and Gemini in mean response length.

These measurements show descriptive differences but **do not establish which features influenced the CNN or LSTM predictions**. We did not conduct feature-ablation or statistical-significance experiments for RQ4.

### Overall Results

| Experiment | CNN accuracy | LSTM accuracy |
|---|---:|---:|
| RQ1: Output-only | **82.72%** | 71.60% |
| RQ2: Input-only | 33.33% | 33.33% |
| RQ2: Combined input/output | **77.78%** | 53.09% |
| RQ3: Unseen general writing | **72.22%** | 56.67% |

## Repository Structure

The project is organized around three Jupyter notebooks and the shared dataset. The intended structure **after the standalone RQ4 notebook is committed** is:

```text
CMPSC-448-Midterm-Project/
├── README.md
├── CNN.ipynb
├── LSTM.ipynb
├── RQ4.ipynb
└── data/
```

`RQ4.ipynb` is a suggested filename. If the standalone notebook has a different name, update the tree and running instructions to match the actual file in the repository. The `data/` folder contains the dataset and related project files; use the actual paths referenced by the notebooks.

### CNN Notebook — RQ1, RQ2, and RQ3

`CNN.ipynb` contains CNN preprocessing, training, checkpoint selection, evaluation, confusion matrices, and experiments for RQ1–RQ3.

### LSTM Notebook — RQ1, RQ2, and RQ3

`LSTM.ipynb` contains LSTM preprocessing, training, checkpoint selection, evaluation, confusion matrices, and experiments for RQ1–RQ3.

### RQ4 Notebook — Writing-Style Analysis

`RQ4.ipynb` contains the separately organized, jointly developed analysis of response length, MATTR-50, punctuation, and formatting characteristics. It produces descriptive summaries and visualizations corresponding to **Table 9 and Figures 13–14** of the report.

## How to Reproduce the Experiments

The project was developed using **Python, PyTorch, and Google Colab**. Core libraries include **PyTorch, pandas, NumPy, scikit-learn, and Matplotlib**. Open the notebooks in Colab and ensure the dataset is available at each notebook's expected file path before running cells in sequence.

### Running the CNN Notebook

1. Open `CNN.ipynb` in Google Colab.
2. Select an appropriate runtime and make the shared dataset accessible.
3. Execute the notebook cells in sequence.
4. Review RQ1 output-only results and RQ2 input-only/combined results.
5. Run RQ3 and review the source-domain and unseen-domain evaluations.

### Running the LSTM Notebook

1. Open `LSTM.ipynb` in Google Colab.
2. Select an appropriate runtime and make the shared dataset accessible.
3. Execute the notebook cells in sequence.
4. Review RQ1 and RQ2 classification results.
5. Run RQ3 and review the source-domain and unseen-domain evaluations.

### Running the RQ4 Notebook

1. Open the standalone RQ4 notebook in Google Colab (`RQ4.ipynb`, or its actual filename).
2. Make the complete dataset of **540 responses** available at the expected path.
3. Import or install the analysis libraries used in the notebook.
4. Execute the analysis cells in sequence.
5. Review the writing-style statistics and compare the generated charts with **Table 9 and Figures 13–14** of the report.

RQ4 **does not require training CNN or LSTM classifiers**. Exact neural-network results may vary slightly with runtime conditions and training nondeterminism.

## Limitations

- **Dataset size:** 540 responses from 180 prompts is small for neural text classification.
- **Model coverage:** Results apply to the three LLM families and versions used during collection, not all LLMs.
- **Subject coverage:** The dataset covers six selected categories rather than the full range of possible prompts.
- **Evaluation variability:** The primary experiments used a single prompt-grouped split and an 81-response test set.
- **Architecture comparison:** CNN and LSTM differed in architecture and RQ1 checkpoint-selection criteria.
- **Cross-domain scope:** Only general writing was held out; leave-one-category-out evaluation was not performed.
- **Writing-style attribution:** RQ4 is descriptive and does not establish which characteristics caused classifier predictions.
- **Authorship certainty:** Predictions reflect statistical patterns and do not constitute definitive proof of authorship.

## Future Work

Future extensions could include larger datasets, additional LLMs and versions, transformer-based classifiers, repeated splits and random seeds, leave-one-category-out testing, controlled studies of response length and truncation, and feature-ablation experiments connecting writing-style statistics to classifier predictions.

## Team Contributions

### Palash Doshi — Team Leader

- CNN architecture design, implementation, training, and evaluation
- CNN experiments for RQ1, RQ2, and RQ3
- **Joint development of the standalone RQ4 writing-style analysis notebook**
- Experimental interpretation, project coordination, and documentation

### Siddharth Pendyala

- LSTM architecture design, implementation, training, and evaluation
- LSTM experiments for RQ1, RQ2, and RQ3
- LSTM metrics and visualizations
- **Joint development of the standalone RQ4 writing-style analysis notebook**
- Experimental interpretation and documentation

Both members contributed to research planning, dataset preparation, experimental discussions, and the final report.

## Project Report

The accompanying **CMPSC 448 midterm project report** documents the motivation, development timeline, dataset design, preprocessing, CNN and LSTM architectures, experiments for RQ1–RQ3, separately implemented RQ4 analysis, figures, limitations, conclusions, and references.

## Conclusion

Our experiments show that CNN and LSTM classifiers can distinguish GPT, Claude, and Gemini responses above the three-class chance baseline under the tested conditions. The CNN achieved **82.72%** accuracy on the primary output-only task versus **71.60%** for the LSTM. For unseen general writing, the CNN achieved **72.22%**, while the LSTM achieved **56.67%**.

Generated responses provided stronger identifying information than the original prompts or combined prompt-response text. The independent RQ4 analysis also identified measurable differences in response length, lexical diversity, punctuation, and formatting. These findings support LLM authorship classification within our experimental setting while highlighting limitations in dataset size, domain generalization, and interpretability.

## Course Information

**Course:** CMPSC 448 — Machine Learning and Artificial Intelligence  
**Institution:** The Pennsylvania State University  
**Project:** Midterm Project  
**Semester:** Fall 2026  
**Submission Date:** October 11, 2026
