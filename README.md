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
├── README.md                           # This file
└── (optional outputs on Drive)
    ├── best_model/                     # Saved fine-tuned model + tokenizer
    ├── label_maps.json
    ├── classification_report.txt
    ├── test_metrics.json
    ├── confusion_matrix_test.png
    └── predictions_*.csv
```

---

## ⚙️ Paths (Colab Defaults)

| Variable | Path |
|-----------|------|
| **DATA_PATH** | `/content/drive/MyDrive/Proyek/Sentiment_IndoBERT/Data/tweet.csv` |
| **OUTPUT_DIR** | `/content/drive/MyDrive/Proyek/Sentiment_IndoBERT` |
| **MODEL_DIR (for inference)** | `/content/drive/MyDrive/Proyek/Sentiment_IndoBERT/best_model` |

> Make sure your `tweet.csv` file has two columns:  
> `tweet` → text input, and `sentimen` → label (`positif`, `netral`, `negatif`).

---

## 🚀 How to Run (Training)

1. Open **[`notebooks/IndoBERT_V1.ipynb`](notebooks/IndoBERT_V1.ipynb)** in **Google Colab**.
2. Run all cells in order:
   - Mount Google Drive  
   - Confirm dataset path  
   - Train IndoBERT (≈ 3 epochs by default)
3. Wait until the notebook prints final metrics:
   - **Accuracy**, **Precision**, **Recall**, **Macro-F1**
4. Check Drive folder for outputs:
   - `/best_model/`, `classification_report.txt`, `test_metrics.json`, `confusion_matrix_test.png`

Example classification report snippet:

```
              precision    recall  f1-score   support
    negatif      0.613     0.706     0.656       119
     netral      0.640     0.603     0.621       121
    positif     0.625     0.569     0.596       123
   macro avg    0.626     0.626     0.624       363
   accuracy                          0.625       363
```

---

## 🔍 How to Run (Inference Only)

1. Open **[`notebooks/IndoBERT_Inference_V1.ipynb`](notebooks/IndoBERT_Inference_V1.ipynb)**.
2. Make sure your fine-tuned model exists in:  
   `/content/drive/MyDrive/Proyek/Sentiment_IndoBERT/best_model`
3. Run **Process 1–3** to install dependencies, mount Drive, and load model.
4. Use the helper function:

```python
predict_texts([
    "Pelayanannya sangat memuaskan, terima kasih!",
    "Biasa saja sih, tidak terlalu istimewa.",
    "Sangat buruk, saya kecewa."
])
```

Expected output:

```python
['positif', 'netral', 'negatif']
```

5. To predict from a CSV:

```python
CSV_PATH = "/content/drive/MyDrive/Proyek/Sentiment_IndoBERT/Data/tweet.csv"
TEXT_COL = "tweet"
```

→ Saves `predictions_YYYYMMDD_HHMMSS.csv` to your project folder.

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
- **classification_report.txt** – Precision / Recall / F1 per class  
- **test_metrics.json** – Numeric metrics summary  
- **confusion_matrix_test.png** – Visual evaluation  
- **predictions_*.csv** – Batch inference results  

---

## 💡 Tips

- To improve performance:
  - Increase `EPOCHS` to 4–5 and monitor overfitting.
  - Adjust `LR` between `2e-5` → `3e-5`.
  - Use `MAX_LEN = 192` for longer texts.
- Use GPU runtime (`T4` / `A100`) on Colab.
- The Trainer automatically keeps the **best checkpoint** by macro-F1.

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

- Python ≥ 3.9  
- `transformers`  
- `datasets`  
- `accelerate`  
- `scikit-learn`  
- `matplotlib`  
- `sentencepiece`

Install (Colab):

```bash
!pip install -q transformers datasets accelerate scikit-learn matplotlib sentencepiece
```

---

## 🏁 Author

**Aris (moonandart)**  
AI & NLP enthusiast — exploring sentiment analysis, BERT fine-tuning, and contextual AI tools.  
🔗 [GitHub](https://github.com/moonandart)
