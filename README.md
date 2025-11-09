# ⚡ AI-Based Smart Energy Consumption Predictor

This project predicts national electricity demand using a locally trained machine learning model.  
It’s built with **Streamlit** for the interface and **scikit-learn** for model training — no API or internet dependency.

---

## 🔍 How It Works

The app takes the following inputs:
- Year  
- Population  
- GDP (in USD)  
- Primary Energy Consumption (TWh)  
- Fossil Fuel Consumption (TWh)  
- Renewables Consumption (TWh)  
- Greenhouse Gas Emissions (MtCO₂e)

After entering these values, it predicts the **Electricity Demand (TWh)**.

**Example Output:**

---

## 🧠 Tech Used

- Python  
- Streamlit  
- scikit-learn  
- pandas, numpy  

---

## 🏗️ Folder Structure
sample/
├── backend/
├── frontend/
├── data/
├── models/
├── train_model.py
└── README.md

---

## ▶️ Run Locally

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run frontend/app.py



