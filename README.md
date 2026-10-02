# MultilayerNERModel — Nested Named Entity Recognition with Multilayer BERT

Code for **"Nested Named Entity Recognition using Multilayer BERT-based Model"**, Notebook for the BioASQ Lab at **CLEF 2024**.

A multilayer BERT architecture for **nested named entity recognition (NNER)**, where entity mentions overlap or are fully contained within longer mentions. Evaluated on the **BioASQ-BioNNE 2024** shared task across all three tracks: English, Russian, and Bilingual.

**Best English result: 67.30% F1 / 56.36% macro F1** (PubMedBERT base).

---

## Motivation

Biomedical text is densely nested. In the sentence *"GCSs caused a decrease in the serum level of soluble interleukin-2 receptor (sCD25) in both groups"*, the chemical mention *interleukin* sits inside a longer chemical span, which in turn sits inside a FINDING span covering the whole decrease event — six levels of nesting in one sentence.

Conventional flat NER assigns one label per token and must therefore discard all but one of these spans. This work keeps every level: nested annotations are decomposed by depth, a separate classification head is trained per level on shared contextual embeddings, and predictions are merged back into nested spans. Each sub-task stays in standard BIO format, so the base encoder is swappable — which is how the same architecture was applied to English, Russian, and bilingual tracks without redesign.

---

## Method

![Methodology](Assets/methodology.svg)

**Base model.** Pretrained **PubMedBERT** provides contextualized embeddings. BERT and BioBERT were also evaluated as drop-in alternatives; any encoder matching the dataset's domain and language can be substituted.

**Classification layers.** Six linear classification heads sit on top of the encoder, one per nesting level (the corpus annotations reach six levels deep). Each head maps PubMedBERT hidden states to **17 output classes** — `B-` and `I-` for each of the eight entity types, plus `O`.

**Dictionary augmentation.** Eight dictionaries built from the **UMLS Metathesaurus 2024AA** (`MRCONSO.RRF`), one per entity type, selected by UMLS Semantic Group. Dictionary matches on the test data are merged with model predictions.

**Preprocessing.** Abstracts are split into sentences with annotations remapped to sentence scope, then encoded with six levels of BIO tagging. Sentences are shuffled and split 80:20 for hyperparameter tuning; the full set is then used for final training.

---

## Repository Structure

Notebooks run in numerical order.

| Notebook | Purpose |
|---|---|
| `1. prepare Nested NER dataset.ipynb` | Load BioASQ-BioNNE abstracts and annotations; map annotations to sentences |
| `2. convert dataset and split.ipynb` | Six-level nested BIO tagging; 80:20 train/validation split |
| `3. Data Analysis.ipynb` | Entity type and per-layer tag distribution statistics |
| `4. LayeredNNER.ipynb` | Train and evaluate the MultilayerNERModel |
| `5. dictionary.ipynb` | Build UMLS dictionaries; dictionary matching and merge with model output |

---

## Dataset

**BioASQ-BioNNE 2024** ([task overview](https://bioasq.org/)), derived from the NEREL-BIO corpus. Eight biomedical entity classes, nesting up to six levels.

| Track | Train | Dev | Test |
|---|---|---|---|
| English | 54 abstracts | 50 | 500 |
| Russian | 716 abstracts | 50 | 500 |

Entity distribution (English train / dev):

| Entity type | Train | Dev |
|---|---|---|
| DISO | 1,200 | 1,012 |
| ANATOMY | 911 | 897 |
| CHEM | 579 | 575 |
| FINDING | 456 | 348 |
| PHYS | 397 | 379 |
| LABPROC | 190 | 154 |
| INJURY_POISONING | 90 | 20 |
| DEVICE | 20 | 28 |
| **Total** | **3,843** | **3,413** |

`DISO` and `ANATOMY` dominate; `DEVICE` is heavily underrepresented. Test annotations were not released to participants.

---

## Results

### English track

| Model | Precision (%) | Recall (%) | F1 (%) | Macro F1 (%) |
|---|---|---|---|---|
| BERT | 53.94 | 59.32 | 56.50 | 44.49 |
| BERT + UMLS | 46.39 | 64.40 | 53.93 | 46.15 |
| BioBERT | 64.01 | 66.28 | 65.12 | 54.49 |
| BioBERT + UMLS | 53.58 | 70.97 | 61.06 | 53.58 |
| **PubMedBERT** | **66.45** | 68.18 | **67.30** | **56.36** |
| PubMedBERT + UMLS | 55.10 | **72.55** | 62.63 | 55.46 |

### Russian and Bilingual tracks

| Track | Base encoder | Precision (%) | Recall (%) | F1 (%) | Macro F1 (%) |
|---|---|---|---|---|---|
| Russian | SBERT-Large-NLU-RU | 68.59 | 65.34 | 66.93 | 60.07 |
| Bilingual | BERT-Base-Multilingual-uncased | 60.27 | 57.50 | 58.89 | 50.53 |

### What the ablations show

**Domain pretraining dominates.** PubMedBERT beats general BERT by **10.8 F1 points** (67.30 vs 56.50) on identical splits and hyperparameters. BioBERT lands between them. The gain is architecture-independent — it comes entirely from the pretraining corpus.

**UMLS dictionaries trade precision for recall, and the trade is unfavorable.** Adding dictionaries raises recall in every configuration (PubMedBERT 68.18 → 72.55; BioBERT 66.28 → 70.97) but costs more precision than it gains (PubMedBERT 66.45 → 55.10), so F1 falls in all three pairs. Macro F1 behaves differently from F1 for the BERT baseline — dictionaries help there (44.49 → 46.15) because they surface rare classes the model alone misses. Useful when coverage matters more than exactness; not a default.

**Deep layers are nearly empty.** Per-layer tag distributions show FINDING, DISO and ANATOMY present at all nesting levels, while level 6 contains almost nothing but `O`. Future work could prune or weight layers by expected occupancy rather than training all six uniformly.

---

## Setup

```bash
git clone https://github.com/hurlab/Nested-NER-BERT.git
cd Nested-NER-BERT
pip install -r requirements.txt
```
**Requirements**

- Python 3.9+
- PyTorch with CUDA
- Hugging Face Transformers
- UMLS Metathesaurus 2024AA access ([UMLS license required](https://uts.nlm.nih.gov/uts/)) for `5. dictionary.ipynb`

---

## Training Configuration

| Hyperparameter | Value |
|---|---|
| Batch size | 64 |
| Learning rate | 1e-4 |
| Epochs | 40 |
| Max sequence length | 512 |
| Optimizer | Adam |

Training ran on **6 × NVIDIA Tesla V100 (32 GB HBM2 each)** using PyTorch `DataParallel` for multi-GPU throughput. Settings were held constant across all six model configurations so the comparison isolates the encoder and the dictionary component.

---

## Usage

```
1. prepare Nested NER dataset.ipynb     # corpus → sentence-aligned annotations
2. convert dataset and split.ipynb      # → six-level BIO tags, 80:20 split
3. Data Analysis.ipynb                  # optional: distribution statistics
4. LayeredNNER.ipynb                    # train + evaluate MultilayerNERModel
5. dictionary.ipynb                     # UMLS dictionaries + merge
```

To swap the base encoder, change the pretrained checkpoint in `4. LayeredNNER.ipynb` — the layer stack and tagging scheme are encoder-agnostic.

---

## Citation

```bibtex
@inproceedings{rehana2024nested,
  title     = {Nested Named Entity Recognition using Multilayer {BERT}-based Model},
  author    = {Rehana, Hasin and Bansal, Benu and {\c{C}}am, Nur Bengisu and
               Zheng, Jie and He, Yongqun and {\"O}zg{\"u}r, Arzucan and Hur, Junguk},
  booktitle = {CLEF 2024 Working Notes},
  series    = {CEUR Workshop Proceedings},
  volume    = {3740},
  pages     = {197--206},
  year      = {2024},
  url       = {https://ceur-ws.org/Vol-3740/paper-18.pdf}
}
```

---

## Acknowledgments

Supported by the US National Institute of Allergy and Infectious Disease (U24AI171008 to Y. He and J. Hur). GEBIP Award of the Turkish Academy of Sciences (to A. Özgür) gratefully acknowledged.

## Contact

Hasin Rehana — hasin.rehana@und.edu · [Hur Lab](https://github.com/hurlab), University of North Dakota
