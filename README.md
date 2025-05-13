# FinTabTransformer

**FinTabTransformer** is a model designed for **bond credit default prediction**, which integrates **financial data**, **ESG (Environmental, Social, and Governance)** scores, and **sentiment information**.

Using our custom dataset **FinSentim-ESG**, FinTabTransformer achieves high prediction performance (**R² = 0.9381**), demonstrating the effectiveness of incorporating **non-financial data** into credit risk modeling.

---

## 🔧 Features

- ✅ Combines tabular financial features with ESG and sentiment embeddings  
- ✅ Achieves state-of-the-art performance on bond default prediction  
- ✅ Simple pipeline, easy to customize and extend  

---

## 📁 Dataset

The model is trained on **FinSentim-ESG**, a newly curated dataset that includes:

- Financial indicators (structured tabular data)  
- ESG ratings and category tags  
- Sentiment signals from public news and corporate disclosures  

> Note: Due to data privacy, the dataset is not publicly released. Please contact the authors for academic access.

---

## 🚀 Getting Started

### 1. Install Dependencies

Ensure you are using **Python 3.8+**. Install required packages:

```bash
pip install -r requirements.txt
