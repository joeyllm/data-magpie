# Data Magpie

This repo is for **data preparation and cleaning** for Joey LLM training.

---

## 🎯 Focus for This Sprint

- Use **one parquet file** from the FineWeb dataset (any file is fine for now).  
- Clean the data and tokenize it.  
- Build a **dataset class in PyTorch** to prepare the data for training.  
- Output should be ready to plug into the training pipeline in **models-magpie**.  

---

## 📦 Deliverable

A script that:  
1. Loads and cleans a parquet file from FineWeb.  
2. Tokenizes the data.  
3. Defines a **PyTorch dataset/dataloader** that can be used in training.  

---

## 🔜 Coming Next Sprint

- Swap in the **full FineWeb dataset** once it is available.  
- Optimize cleaning/tokenization for **larger scale data**.  

---

## 📝 Notes

- Keep it simple: one parquet file, cleaned and tokenized.  
- If needed, look up examples of cleaning + tokenizing text for LLM training.  
- The important part: **working code that outputs a dataset ready for training**.  
