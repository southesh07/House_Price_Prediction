# 🏡 House Price Predictor

A simple Machine Learning project that predicts house sale prices using Ridge Regression, built with Python, Scikit-learn, and Streamlit.

---

## 📁 Project Structure

```
house_price_prediction/
├── app.py                       # ML pipeline (load → train → save)
├── streamlit_app.py             # Streamlit web application
├── model_bundle.pkl             # Saved model + scaler + feature cols
├── house_prices_practice.csv    # Dataset (300 rows × 9 columns)
├── requirements.txt             # Python dependencies
└── README.md
```

---

## 📊 Dataset

| Column | Description |
|---|---|
| `OverallQual` | Overall quality (1–10) |
| `GrLivArea` | Above-ground living area (sq ft) |
| `GarageCars` | Garage capacity (# cars) |
| `TotalBsmtSF` | Basement area (sq ft) |
| `YearBuilt` | Construction year |
| `FullBath` | Full bathrooms above grade |
| `BedroomAbvGr` | Bedrooms above grade |
| `LotArea` | Lot size (sq ft) |
| **`SalePrice`** | **Target — sale price (USD)** |

---

## 🚀 How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the model
python app.py

# 3. Launch the web app
streamlit run streamlit_app.py
```

---

## 📦 Tech Stack

| Tool | Purpose |
|---|---|
| Pandas / NumPy | Data loading & manipulation |
| Scikit-learn | Preprocessing, Ridge model, evaluation |
| Pickle | Model save/load |
| Streamlit | Web application |
