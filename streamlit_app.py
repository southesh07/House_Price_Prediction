import streamlit as st
import pickle
import numpy as np

st.set_page_config(page_title="House Price Predictor", page_icon="🏡", layout="centered")

# Load model
@st.cache_resource
def load_model():
    return pickle.load(open("model_bundle.pkl", "rb"))

bundle       = load_model()
model        = bundle["model"]
scaler       = bundle["scaler"]
feature_cols = bundle["feature_cols"]

# Header
st.title("🏡 House Price Predictor")
st.write("Fill in the property details below and click **Estimate Price**.")
st.divider()

# Inputs
col1, col2 = st.columns(2)
overall_qual  = col1.selectbox("Overall Quality (1–10)", range(1, 11), index=5)
year_built    = col2.selectbox("Year Built", range(2024, 1871, -1), index=24)

col3, col4 = st.columns(2)
gr_liv_area   = col3.number_input("Living Area (sq ft)", 300, 6000, 1500, 50)
total_bsmt_sf = col4.number_input("Basement Area (sq ft)", 0, 4000, 800, 50)

col5, col6 = st.columns(2)
lot_area      = col5.number_input("Lot Area (sq ft)", 1000, 215000, 9000, 500)
garage_cars   = col6.selectbox("Garage Cars", [0, 1, 2, 3, 4], index=2)

col7, col8 = st.columns(2)
full_bath     = col7.selectbox("Full Bathrooms", [0, 1, 2, 3, 4], index=2)
bedroom_abvgr = col8.selectbox("Bedrooms", range(0, 9), index=3)

st.divider()

# Predict
if st.button("🔍 Estimate Price", use_container_width=True):
    house_age = 2025 - year_built
    total_sf  = gr_liv_area + total_bsmt_sf
    qual_area = overall_qual * gr_liv_area

    row = {
        "OverallQual": overall_qual, "GrLivArea": gr_liv_area,
        "GarageCars": garage_cars,   "TotalBsmtSF": total_bsmt_sf,
        "YearBuilt": year_built,     "FullBath": full_bath,
        "BedroomAbvGr": bedroom_abvgr, "LotArea": lot_area,
        "HouseAge": house_age,       "TotalSF": total_sf,
        "QualArea": qual_area,
    }

    X      = np.array([[row[c] for c in feature_cols]])
    price  = float(model.predict(scaler.transform(X))[0])
    price  = max(price, 50_000)

    st.success(f"### Estimated Sale Price: **${price:,.0f}**")
    st.caption(f"Range: ${price * 0.92:,.0f} – ${price * 1.08:,.0f}")
