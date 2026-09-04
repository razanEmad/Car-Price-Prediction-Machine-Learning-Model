import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Page Config
st.set_page_config(
    page_title="Car Price Estimator",
    page_icon="🚗",
    layout="wide"
)

# Custom Styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .prediction-card {
        background: linear-gradient(135deg, #F8FAFC 0%, #F1F5F9 100%);
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 26px;
        text-align: center;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    .price-value {
        font-size: 2.6rem;
        font-weight: 800;
        color: #0F766E;
        line-height: 1.2;
    }
    .usd-value {
        font-size: 1.4rem;
        font-weight: 700;
        color: #2563EB;
        margin-top: 8px;
    }
    .exchange-rate {
        font-size: 0.82rem;
        color: #94A3B8;
        margin-top: 6px;
    }
    .price-label {
        font-size: 0.9rem;
        color: #475569;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 8px;
    }
    </style>
""", unsafe_allow_html=True)

# Directly run the exact ML_project.ipynb code in memory
@st.cache_resource
def load_ml_model_from_notebook_code():
    # Code directly from ML_project.ipynb
    df_original = pd.read_csv("processes2_intentionally_uncleaned.csv")
    df = df_original.copy()
    df = df.drop_duplicates()
    df = df.dropna(subset=['selling_price'])
    if 'Mileage Unit' in df.columns:
        df = df[df['Mileage Unit'] == 'kmpl']
        df = df.drop(columns=['Mileage Unit'])

    df = df.replace(['nan', 'NAN'], np.nan)
    df['name'] = df['name'].str.lower().str.strip()
    df = df.dropna(subset=['name'])

    # Imputations from notebook
    df['km_driven'] = df['km_driven'].fillna(df['km_driven'].median())
    df['seats'] = df['seats'].fillna(df['seats'].median())
    df['max_power (in bph)'] = df['max_power (in bph)'].fillna(df['max_power (in bph)'].median())
    df['Mileage'] = df['Mileage'].fillna(df['Mileage'].median())
    df['Engine (CC)'] = df['Engine (CC)'].fillna(df['Engine (CC)'].median())
    df['year'] = df['year'].fillna(df['year'].median())

    df['fuel'] = df['fuel'].fillna(df['fuel'].mode()[0])
    df['seller_type'] = df['seller_type'].fillna(df['seller_type'].mode()[0])
    df['transmission'] = df['transmission'].fillna(df['transmission'].mode()[0])
    df['owner'] = df['owner'].fillna(df['owner'].mode()[0])

    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)

    x = df.drop(columns=['selling_price'])
    y = df['selling_price']

    ordinal_encoder = OrdinalEncoder()
    x['owner'] = ordinal_encoder.fit_transform(x[['owner']])

    x_encoded = pd.get_dummies(x, columns=['name', 'fuel', 'seller_type', 'transmission'], drop_first=True)
    bool_cols = x_encoded.select_dtypes(include='bool').columns
    x_encoded[bool_cols] = x_encoded[bool_cols].astype(int)

    x_train, x_test, y_train, y_test = train_test_split(x_encoded, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    num_cols = ['year', 'km_driven', 'seats', 'max_power (in bph)', 'Mileage', 'Engine (CC)']
    x_train[num_cols] = scaler.fit_transform(x_train[num_cols])
    x_test[num_cols] = scaler.transform(x_test[num_cols])

    model = LinearRegression()
    model.fit(x_train, y_train)

    return {
        'model': model,
        'scaler': scaler,
        'ordinal_encoder': ordinal_encoder,
        'feature_cols': x_encoded.columns.tolist(),
        'num_cols': num_cols,
        'unique_names': sorted(df['name'].unique().tolist()),
        'unique_fuels': sorted(df['fuel'].unique().tolist()),
        'unique_seller_types': sorted(df['seller_type'].unique().tolist()),
        'unique_transmissions': sorted(df['transmission'].unique().tolist()),
        'owner_categories': ordinal_encoder.categories_[0].tolist(),
        'df_summary': {
            'year_median': int(df['year'].median()),
            'km_median': int(df['km_driven'].median()),
            'seats_options': sorted([int(s) for s in df['seats'].unique()]),
            'mileage_median': float(df['Mileage'].median()),
            'engine_median': int(df['Engine (CC)'].median()),
            'power_median': float(df['max_power (in bph)'].median()),
        }
    }

ml = load_ml_model_from_notebook_code()

# Header
st.markdown('<div class="main-header">Car Valuation Tool</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Calculate an estimated market selling price based on vehicle specs and condition.</div>', unsafe_allow_html=True)

col_input, col_output = st.columns([3, 2], gap="large")

with col_input:
    st.subheader("Vehicle Specifications")

    col1, col2 = st.columns(2)
    with col1:
        brand = st.selectbox("Car Brand", options=ml['unique_names'], index=ml['unique_names'].index("maruti") if "maruti" in ml['unique_names'] else 0)
        year = st.slider("Manufacturing Year", min_value=1990, max_value=2023, value=ml['df_summary']['year_median'])
        km_driven = st.number_input("Kilometers Driven", min_value=0, max_value=1000000, value=ml['df_summary']['km_median'], step=5000)
        owner = st.selectbox("Ownership Status", options=ml['owner_categories'], index=0)

    with col2:
        fuel = st.selectbox("Fuel Type", options=ml['unique_fuels'], index=0)
        seller_type = st.selectbox("Seller Type", options=ml['unique_seller_types'], index=0)
        transmission = st.selectbox("Transmission", options=ml['unique_transmissions'], index=0)
        seats = st.selectbox("Seating Capacity", options=ml['df_summary']['seats_options'], index=ml['df_summary']['seats_options'].index(5) if 5 in ml['df_summary']['seats_options'] else 0)

    st.markdown("---")
    st.subheader("Engine & Power Specs")
    col3, col4, col5 = st.columns(3)
    with col3:
        engine_cc = st.number_input("Engine Capacity (CC)", min_value=500, max_value=7000, value=ml['df_summary']['engine_median'], step=100)
    with col4:
        max_power = st.number_input("Max Power (bph)", min_value=20.0, max_value=600.0, value=ml['df_summary']['power_median'], step=5.0)
    with col5:
        mileage = st.number_input("Mileage (kmpl)", min_value=5.0, max_value=50.0, value=ml['df_summary']['mileage_median'], step=1.0)

with col_output:
    st.subheader("Valuation Summary")

    input_data = {
        'name': brand,
        'year': year,
        'km_driven': km_driven,
        'fuel': fuel,
        'seller_type': seller_type,
        'transmission': transmission,
        'owner': owner,
        'seats': seats,
        'max_power (in bph)': max_power,
        'Mileage': mileage,
        'Engine (CC)': engine_cc
    }

    df_in = pd.DataFrame([input_data])
    df_in['owner'] = ml['ordinal_encoder'].transform(df_in[['owner']])

    dummy_data = {c: 0 for c in ml['feature_cols']}
    for c in ml['num_cols'] + ['owner']:
        dummy_data[c] = df_in[c].iloc[0]

    for cat_col, prefix in [('name', 'name_'), ('fuel', 'fuel_'), ('seller_type', 'seller_type_'), ('transmission', 'transmission_')]:
        val = df_in[cat_col].iloc[0]
        d_col = f"{prefix}{val}"
        if d_col in dummy_data:
            dummy_data[d_col] = 1

    df_prepared = pd.DataFrame([dummy_data])[ml['feature_cols']]
    df_prepared[ml['num_cols']] = ml['scaler'].transform(df_prepared[ml['num_cols']])

    predicted_price = ml['model'].predict(df_prepared)[0]
    display_price = max(0, float(predicted_price))

    usd_price = display_price / 83.0

    st.markdown(f"""
        <div class="prediction-card">
            <div class="price-label">Estimated Market Value</div>
            <div class="price-value">₹ {display_price:,.0f}</div>
            <div class="usd-value">$ {usd_price:,.0f} USD</div>
            <div class="exchange-rate">(1 USD ≈ ₹ 83 INR)</div>
        </div>
    """, unsafe_allow_html=True)

    st.write("### Submitted Details")
    summary_df = pd.DataFrame([
        {"Parameter": "Brand", "Value": brand.capitalize()},
        {"Parameter": "Year", "Value": str(year)},
        {"Parameter": "KM Driven", "Value": f"{km_driven:,} km"},
        {"Parameter": "Fuel / Transmission", "Value": f"{fuel} / {transmission}"},
        {"Parameter": "Ownership", "Value": owner},
        {"Parameter": "Engine / Power", "Value": f"{engine_cc} CC / {max_power} bph"},
        {"Parameter": "Mileage", "Value": f"{mileage} kmpl"},
    ])
    st.dataframe(summary_df, hide_index=True, use_container_width=True)
