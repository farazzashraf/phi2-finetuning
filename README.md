# 🚀 Translation Fine-Tuning & Evaluation on LLMs

## 🎯 Learning Objective

This project was focused on **learning how to fine-tune large language models (LLMs)** like `Phi-2`, using efficient techniques such as **LoRA** and **QLoRA**. The goal was to explore the end-to-end process of adapting LLMs for translation tasks and evaluating them with standard NLP metrics.

⚠️ **Note:** This was **not a full-scale or production-quality fine-tuning**. The goal was to explore the process — using a small dataset for experimentation — rather than achieve high evaluation scores.

---

## 📌 Project Highlights

- ✅ Learned how to fine-tune a small LLM (`Phi-2`) on a bilingual dataset (English ↔ Malayalam)
- ✅ Applied parameter-efficient fine-tuning using **LoRA/QLoRA**
- ✅ Measured translation quality using **BLEU score**
- ✅ Practiced reproducible ML workflows with Hugging Face libraries

---

## 📊 Dataset

We used the [COCO English-Malayalam Translation Corpus](https://github.com/narVidhai/COCO-English-Malayalam-Translation-Corpus) for this experiment.

- **Source Language**: English  
- **Target Language**: Malayalam  
- **Tokenizer**: Hugging Face `AutoTokenizer`  
- **Size**: Small-scale for fast experimentation

---

## ⚙️ Fine-Tuning Setup

- **Base Model**: `microsoft/phi-2`
- **Fine-Tuning Method**: `QLoRA` via `peft`
- **Trainer**: `transformers.Trainer`
- **Platform**: Kaggle / Local GPU (16GB)

---

## 📈 Evaluation

| Metric     | Tool         | Description                      |
|------------|--------------|----------------------------------|
| BLEU       | `sacrebleu`  | Measures translation accuracy via n-gram overlap |

📌 **Disclaimer**: The BLEU scores weren’t strong — again, this wasn't about performance, but about learning how evaluation works.

---

## 🧠 What I Learned

- LoRA and QLoRA make fine-tuning accessible even with limited hardware
- Hugging Face ecosystem (Transformers, Datasets, PEFT) simplifies LLM experimentation
- Evaluation with BLEU helped understand quality differences in model outputs
- Tokenization, preprocessing, and adapter merging are crucial steps in custom LLM workflows

---

> 🧪 This project was just a **learning sandbox** to understand how to fine-tune LLMs. It helped me practice the full loop — from data prep to adapter training and evaluation — in a low-risk, fast way.


