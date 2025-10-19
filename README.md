# 🇮🇩 IndoBERT Sentiment Analysis (Version 1)

Fine-tuning **IndoBERT** (`indobenchmark/indobert-base-p1`) for **Indonesian sentiment classification** using tweet data.

This repository contains two Google Colab–ready notebooks:

| Notebook | Purpose |
|-----------|----------|
| [`notebooks/IndoBERT_V1.ipynb`](notebooks/IndoBERT_V1.ipynb) | End-to-end pipeline — from preprocessing and label encoding to fine-tuning, evaluation, and artifact export. |
| [`notebooks/IndoBERT_Inference_V1.ipynb`](notebooks/IndoBERT_Inference_V1.ipynb) | Inference-only notebook — load trained model, predict new texts, or run batch CSV predictions. |

---

## 🧭 Project Overview

**Objective:**  
Train a robust IndoBERT model to classify Indonesian texts (e.g., tweets) into `positif`, `netral`, or `negatif` sentiments.

**Workflow Summary**
1. **Preprocessing** – load raw CSV (`tweet.csv`), clean whitespace, encode sentiment labels.  
2. **Dataset Split** – 80 / 10 / 10 stratified split for train, validation, and test.  
3. **Tokenization** – IndoBERT tokenizer with `max_length=128`.  
4. **Fine-Tuning** – Transformers `Trainer`, optimizing for **macro-F1**.  
5. **Evaluation** – accuracy, precision, recall, macro-F1, confusion matrix.  
6. **Export** – best model checkpoint, tokenizer, label maps, metrics, and reports.  
7. **Inference** – predict single texts or whole CSV files.

---

## 📂 Folder Structure

```
Sentiment_IndoBERT/
│
├── notebooks/
│   ├── IndoBERT_V1.ipynb               # Full fine-tuning & evaluation
│   └── IndoBERT_Inference_V1.ipynb     # Inference-only version
│
├── outputs/                            # (optional) commit selected small files only
│   ├── label_maps.json
│   ├── classification_report.txt
│   ├── test_metrics.json
│   ├── confusion_matrix_test.png
│   └── best_model/                     # (large) model weights + tokenizer — usually not committed
│
├── data/                               # (optional) dataset folder
│   └── tweet.csv
│
├── README.md
├── requirements.txt
└── .gitignore
```

> **Note:** The notebooks save artifacts to your Google Drive at  
> `/content/drive/MyDrive/Proyek/Sentiment_IndoBERT/`

---

## ⚙️ Paths (Colab Defaults)

| Variable | Path |
|-----------|------|
| **DATA_PATH** | `/content/drive/MyDrive/Proyek/Sentiment_IndoBERT/Data/tweet.csv` |
| **OUTPUT_DIR** | `/content/drive/MyDrive/Proyek/Sentiment_IndoBERT` |
| **MODEL_DIR (for inference)** | `/content/drive/MyDrive/Proyek/Sentiment_IndoBERT/best_model` |

> Ensure your `tweet.csv` has columns:  
> `tweet` → text input, and `sentimen` → label (`positif`, `netral`, `negatif`).

---

## 🚀 How to Run (Training)

1. Open **[`notebooks/IndoBERT_V1.ipynb`](notebooks/IndoBERT_V1.ipynb)** in **Google Colab**.
2. Run **Process 1** (Clean Install). If prompted, **restart runtime** after installs.
3. Run **Process 2–9** in order (mount Drive, config, load/encode, split, tokenize, train, evaluate, export).
4. Review the outputs in Drive:
   - `/best_model/` (model + tokenizer)
   - `label_maps.json`, `test_metrics.json`, `classification_report.txt`
   - `confusion_matrix_test.png`

**Example classification report** (illustrative):

```
              precision    recall  f1-score   support
    negatif      0.613     0.706     0.656       119
     netral      0.640     0.603     0.621       121
    positif      0.625     0.569     0.596       123
   macro avg    0.626     0.626     0.624       363
   accuracy                          0.625       363
```

---

## 🔍 How to Run (Inference Only)

1. Open **[`notebooks/IndoBERT_Inference_V1.ipynb`](notebooks/IndoBERT_Inference_V1.ipynb)**.
2. Confirm your model exists at: `/content/drive/MyDrive/Proyek/Sentiment_IndoBERT/best_model`.
3. Run **Process 1–3** to install dependencies, mount Drive, and load model.
4. Predict directly in Python:

```python
predict_texts([
    "Pelayanannya sangat memuaskan, terima kasih!",
    "Biasa saja sih, tidak terlalu istimewa.",
    "Sangat buruk, saya kecewa."
])
# -> ['positif', 'netral', 'negatif']
```

5. Predict from a CSV:

```python
CSV_PATH = "/content/drive/MyDrive/Proyek/Sentiment_IndoBERT/Data/tweet.csv"
TEXT_COL = "tweet"
```
Outputs a file like `predictions_YYYYMMDD_HHMMSS.csv` to your Drive project folder.

---

## 🧠 Model & Training Configuration

| Parameter | Value |
|------------|--------|
| Model | `indobenchmark/indobert-base-p1` |
| Tokenizer | IndoBERT FastTokenizer |
| Epochs | 3 |
| Batch Size | 16 |
| Max Length | 128 |
| Learning Rate | 2e-5 |
| Optimizer | AdamW |
| Evaluation Metric | **Macro-F1** |
| Mixed Precision | FP16 / BF16 (auto-select) |

---

## 📊 Outputs and Artifacts

- **best_model/** – Saved weights + tokenizer  
- **label_maps.json** – `label2id` & `id2label` mappings  
- **classification_report.txt** – per-class Precision / Recall / F1  
- **test_metrics.json** – overall metrics summary  
- **confusion_matrix_test.png** – visual evaluation  
- **predictions_*.csv** – batch inference results  

---

## 💡 Tips & Troubleshooting

- **Dependency conflicts in Colab**: The training notebook includes a robust **Process 1** that removes RAPIDS/dask/gcsfs, pins `pyarrow==19.0.0`, installs NLP libs, and runs `pip check`.
- **Longer texts**: Increase `MAX_LEN` to `192` (slower but may improve recall).
- **More training**: Raise `EPOCHS` to 4–5 (monitor validation macro-F1).
- **GPU**: Use T4/A100 on Colab for faster training.
- **Large files**: Don’t commit `best_model/` to GitHub; use `.gitignore`.

---

## 🧾 Citation

If you use this project, please cite:

```
@misc{Sentiment_IndoBERT_2025,
  author = {moonandart (Aris)},
  title  = {IndoBERT Sentiment Analysis},
  year   = {2025},
  url    = {https://github.com/moonandart/Sentiment_IndoBERT}
}
```

---

## 🧩 Dependencies

See [`requirements.txt`](requirements.txt). For Colab, the training notebook’s **Process 1** installs:
- `transformers`, `datasets<3.0.0`, `accelerate`
- `scikit-learn`, `matplotlib`, `sentencepiece`
- `pyarrow==19.0.0` (pinned)
- `jedi` (for IPython completeness)

---

## 🏁 Author

**Aris (moonandart)**  
AI & NLP enthusiast — exploring sentiment analysis, BERT fine-tuning, and contextual AI tools.  
🔗 https://github.com/moonandart
