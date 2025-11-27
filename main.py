
import os
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import io
from dateutil import parser
import requests
import datetime


# ------------------------------
# Custom CSS
# ------------------------------
st.markdown(
    """
    <style>
    .title-background {
        background-color: #1a1c23;  /* Slightly lighter than #0e1117 */
        padding: 1rem;
        text-align: center;
        font-size: 2.5em;
        font-weight: bold;
        color: #ffffff;
        width: 100%;
        display: block;
    }
    .css-1d391kg { padding-top: 0.5rem; }
    </style>
    """,
    unsafe_allow_html=True
)

# ------------------------------
# Centered Title
# ------------------------------
st.markdown('<div class="title-background">📊 AQI and Pollution Trends (India)</div>', unsafe_allow_html=True)

# ------------------------------
# Sidebar Navigation
# ------------------------------
with st.sidebar:
    st.title("Navigation")

    # Dataset
    with st.expander("1) Dataset", expanded=True):
        uploaded = st.file_uploader("Upload CSV")
        if uploaded is not None:
            show_info = st.checkbox("Dataset Quality")

    # Data Cleaning
    with st.expander("2) Data Cleaning", expanded=False):
        garbage_values = st.checkbox("Remove Garbage Values")
        fix_types = st.checkbox("Fix Data Types")
        remove_anomalies = st.checkbox("Remove Anomalies/Nulls")
        impute_missing = st.checkbox("Impute Missing Values")
        normalize_cols = st.checkbox("Normalize Columns")

    # Charts
    with st.expander("3) Dataset Analysis-Charts", expanded=False):
        # Choose dataset for charts
        st.markdown("### Dataset Input")
        chart_dataset_choice = st.radio(
            "",
            options=["Original Dataset", "Cleaned Dataset"],
            index=1
        )
        st.markdown("### Chart Options")
        ts_aqi_plot = st.checkbox("Monthly AQI Trend")
        corr_heatmap_plot = st.checkbox("Correlation Heatmap")
        rolling_trend_checkbox = st.checkbox("PM2.5 Rolling / EMA Trend")
        top_pm25_plot = st.checkbox("Top 10 Cities PM2.5")
        scatter_plot = st.checkbox("PM2.5 vs AQI Scatter")
        custom_plot = st.checkbox("Custom Plot")
        

    # Exploration / Other
    with st.expander("4) Exploratory Analysis of Dataset", expanded=False):
        summary_stats_checkbox = st.checkbox("Summary Statistics")
        column_info_checkbox = st.checkbox("Column Information")
        correlation_matrix_checkbox = st.checkbox("Correlation Matrix")
    
    with st.expander("5) Realtime AQI Analysis of Nearest City", expanded=False):
        aqi_api= st.checkbox("Fetch Nearest City AQI")

# ------------------------------
# Caching functions
# ------------------------------
@st.cache_data
def load_csv(uploaded_file):
    return pd.read_csv(uploaded_file)

@st.cache_data
def generate_pm25_rolling(df):
    if 'PM2.5' not in df.columns:
        return pd.DataFrame()
    df_plot = df[['PM2.5']].copy()
    df_plot = df_plot.reset_index() if df.index.name=='Date' else df_plot.reset_index()
    if 'Date' not in df_plot.columns:
        df_plot = df_plot.rename(columns={df_plot.columns[0]: 'Date'})
    df_plot = df_plot.sort_values('Date')
    df_plot['PM2.5_rolling'] = df_plot['PM2.5'].rolling(window=7).mean()
    df_plot['PM2.5_ema'] = df_plot['PM2.5'].ewm(span=7).mean()
    return df_plot

@st.cache_data(ttl=300)
def fetch_aqi():
    API_KEY = st.secrets.get("AQI_API_KEY")  
    url = f"http://api.airvisual.com/v2/nearest_city?key={API_KEY}"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching AQI data: {e}")
        return None

# ------------------------------
# Main App Logic
# ------------------------------
if uploaded:
    df_original = load_csv(uploaded)

    # Initialize cleaned df in session state
    if 'df_cleaned' not in st.session_state:
        st.session_state.df_cleaned = df_original.copy()

    # Initialize Dataset View in session state
    if 'dataset_view' not in st.session_state:
        st.session_state.dataset_view = "None"

    df_cleaned = st.session_state.df_cleaned

    # ------------------------------
    # Dataset View & Column Dropping
    # ------------------------------
    # ------------------------------
# Dataset View & Column Dropping
# ------------------------------
    st.subheader("View/Edit Dataset")
    c1, c2 = st.columns([0.4,0.6], gap="small")
    with c1:
        valid_views = ["None", "Original Dataset", "Cleaned Dataset", "Both"]
        current_view = st.session_state.get('dataset_view', "None")
        if current_view not in valid_views:
            current_view = "None"
        st.session_state.dataset_view = st.radio(
            "Dataset View",
            options=valid_views,
            index=valid_views.index(current_view)
        )
    with c2:
        # Dropdown for columns to drop
        cols_to_drop = st.multiselect(
            "Columns to drop",
            options=df_cleaned.columns.tolist()
        )
        if st.button("Drop Columns") and cols_to_drop:
            df_cleaned.drop(columns=cols_to_drop, inplace=True)
            st.session_state.df_cleaned = df_cleaned
            st.success(f"Dropped columns: {cols_to_drop}")

    # ------------------------------
    # Cleaning & anomaly removal
    # ------------------------------
    if garbage_values:
        if 'AQI' in df_cleaned.columns: df_cleaned.loc[df_cleaned['AQI']>1200,'AQI']=np.nan
        if 'CO' in df_cleaned.columns: df_cleaned.loc[df_cleaned['CO']>100,'CO']=np.nan

    if remove_anomalies and all(x in df_cleaned.columns for x in ['PM10','CO','AQI']):
        df_cleaned['Date'] = pd.to_datetime(df_cleaned['Date'], errors='coerce')
        for col in ['PM10','CO','AQI']:
            df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')
        anomaly_mask = (
            ((df_cleaned['PM10'] >= 600) & (df_cleaned['AQI'] < 100)) |
            ((df_cleaned['CO'] > 65) & (df_cleaned['AQI'] < 350))
        )
        df_cleaned = df_cleaned[~anomaly_mask].reset_index(drop=True)
        st.session_state.df_cleaned = df_cleaned

    if fix_types:
        for col in df_cleaned.columns:
            if df_cleaned[col].dtype == object:
                df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='ignore')

    if impute_missing and 'City' in df_cleaned.columns:
        numeric_cols = df_cleaned.select_dtypes(include=['int64','float64']).columns.tolist()
        for city, idx in df_cleaned.groupby('City').groups.items():
            df_cleaned.loc[idx, numeric_cols] = df_cleaned.loc[idx, numeric_cols].interpolate(method='linear')
            df_cleaned.loc[idx, numeric_cols] = df_cleaned.loc[idx, numeric_cols].fillna(df_cleaned.loc[idx, numeric_cols].mean())
        df_cleaned[numeric_cols] = df_cleaned[numeric_cols].fillna(df_cleaned[numeric_cols].mean())
        st.session_state.df_cleaned = df_cleaned

    if normalize_cols:
        numeric_cols = df_cleaned.select_dtypes(include=['int64','float64']).columns
        df_cleaned[numeric_cols] = (df_cleaned[numeric_cols]-df_cleaned[numeric_cols].min()) / (df_cleaned[numeric_cols].max()-df_cleaned[numeric_cols].min())
        st.session_state.df_cleaned = df_cleaned

    # ------------------------------
    # Year/Month columns
    # ------------------------------
    if 'Date' in df_cleaned.columns:
        df_cleaned['Date'] = pd.to_datetime(df_cleaned['Date'], errors='coerce')
        if df_cleaned['Date'].notnull().any():
            df_cleaned['Year'] = df_cleaned['Date'].dt.year
            df_cleaned['Month'] = df_cleaned['Date'].dt.month
        st.session_state.df_cleaned = df_cleaned

    # ------------------------------
    # Show Dataset Quality
    # ------------------------------
    if 'show_info' in locals() and show_info:
        st.subheader("Dataset Quality")
        c1, c2, _ = st.columns([0.4, 0.4, 1], gap="small")
        with c1: dat1 = st.checkbox("Original Dataset")
        with c2: dat2 = st.checkbox("Cleaned Dataset")

        def show_missing_duplicates(df):
            missing_count = df.isnull().sum()
            missing_percent = missing_count / df.shape[0] * 100
            st.dataframe(pd.DataFrame({"Missing Count": missing_count, "Missing %": missing_percent}))
            st.subheader("Duplicate Rows")
            st.write(f"Number of duplicated rows: {df.duplicated().sum()}")

        if dat1:
            st.subheader(f"Original Dataset shape: {df_original.shape[0]} × {df_original.shape[1]}")
            show_missing_duplicates(df_original)
        if dat2:
            st.subheader(f"Cleaned Dataset shape: {df_cleaned.shape[0]} × {df_cleaned.shape[1]}")
            show_missing_duplicates(df_cleaned)

    # ------------------------------
    # Display datasets
    # ------------------------------
    if st.session_state.dataset_view in ["Original Dataset","Both"]:
        st.subheader("Original Dataset")
        st.dataframe(df_original)
    if st.session_state.dataset_view in ["Cleaned Dataset","Both"]:
        st.subheader("Cleaned Dataset")
        st.dataframe(df_cleaned)

    # ------------------------------
    # Exploratory Data
    # ------------------------------
    if summary_stats_checkbox:
        st.subheader("Summary Statistics")
        st.dataframe(df_cleaned.describe().T)
    if column_info_checkbox:
        st.subheader("Column Information")
        buffer = io.StringIO()
        df_cleaned.info(buf=buffer)
        st.text(buffer.getvalue())
    if correlation_matrix_checkbox:
        st.subheader("Pearson Correlation Matrix")
        corr_cols = df_cleaned.select_dtypes(include=['int64','float64']).columns
        st.dataframe(df_cleaned[corr_cols].corr())

# ------------------------------
# Charts with dataset selection
# ------------------------------
if uploaded:
    df_for_charts = df_original if chart_dataset_choice=="Original Dataset" else df_cleaned
    df_for_charts['Date'] = pd.to_datetime(df_for_charts['Date'], errors='coerce')
    df_for_charts['Year'] = df_for_charts['Date'].dt.year
    df_for_charts['Month'] = df_for_charts['Date'].dt.month
    numeric_cols = df_for_charts.select_dtypes(include=['int64','float64']).columns.tolist()
    sample_size = min(5000, len(df_for_charts))
    sample_df = df_for_charts.sample(n=sample_size, random_state=42)

    # --- Monthly AQI Trend
    if ts_aqi_plot and 'AQI' in df_for_charts.columns:
        st.write("### Monthly AQI Trend")
        monthly_avg_aqi = df_for_charts.groupby('Date')['AQI'].mean().resample('M').mean()
        fig, ax = plt.subplots(figsize=(14,6))
        sns.lineplot(x=monthly_avg_aqi.index, y=monthly_avg_aqi.values, ax=ax)
        ax.set_xlabel('Year'); ax.set_ylabel('Average AQI')
        st.pyplot(fig); plt.close(fig)

    # --- Top 10 Cities PM2.5
    if top_pm25_plot and 'PM2.5' in df_for_charts.columns and 'City' in df_for_charts.columns:
        st.write("### Top 10 Cities by PM2.5")
        city_avg_pm25 = df_for_charts.groupby('City')['PM2.5'].mean().sort_values(ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(12,7))
        sns.barplot(x=city_avg_pm25.index, y=city_avg_pm25.values, palette='viridis', ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_xlabel('City'); ax.set_ylabel('Average PM2.5')
        st.pyplot(fig); plt.close(fig)

    # --- Correlation Heatmap
    if corr_heatmap_plot:
        st.write("### Correlation Heatmap")
        corr_cols = [col for col in ['PM2.5','PM10','NO2','CO','SO2','O3','AQI'] if col in df_for_charts.columns]
        corr_matrix = df_for_charts[corr_cols].corr()
        fig, ax = plt.subplots(figsize=(10,8))
        sns.heatmap(corr_matrix, annot=True, cmap='CMRmap_r', fmt=".2f", linewidths=.5, ax=ax)
        st.pyplot(fig); plt.close(fig)

    # --- PM2.5 vs AQI Scatter
    if scatter_plot and all(x in df_for_charts.columns for x in ['PM2.5','AQI']):
        st.write("### PM2.5 vs AQI Scatter")
        fig, ax = plt.subplots(figsize=(8,6))
        sns.scatterplot(x=sample_df['PM2.5'], y=sample_df['AQI'], alpha=0.4, color='hotpink', ax=ax)
        ax.set_xlabel('PM2.5'); ax.set_ylabel('AQI')
        st.pyplot(fig); plt.close(fig)

    # --- PM2.5 Rolling / EMA Trend
    if rolling_trend_checkbox and 'PM2.5' in df_for_charts.columns:
        st.write("### PM2.5 Rolling / EMA Trend")
        df_plot = generate_pm25_rolling(df_for_charts)
        if not df_plot.empty:
            for col, color, title in zip(
                ['PM2.5','PM2.5_rolling','PM2.5_ema'],
                ['blue','green','orange'],
                ['Original PM2.5 Trend','7-day Rolling PM2.5 Trend','Exponential Moving Average (EMA) PM2.5 Trend']
            ):
                fig, ax = plt.subplots(figsize=(14,6))
                sns.lineplot(data=df_plot, x='Date', y=col, ax=ax, color=color)
                ax.set_title(title)
                ax.set_xlabel('Date'); ax.set_ylabel('PM2.5')
                st.pyplot(fig); plt.close(fig)

    # --- Custom Plot
    if custom_plot:
        st.write("### Custom Plot")
        max_rows = len(df_for_charts)
        num_rows = st.slider("Number of rows to include in plot", 0, max_rows, min(500, max_rows), 1)
        plot_df = df_for_charts.head(num_rows) if num_rows > 0 else pd.DataFrame(columns=df_for_charts.columns)
        x_col = st.selectbox("X-axis column", df_for_charts.columns, key="x_col")
        y_col = st.selectbox("Y-axis column", df_for_charts.columns, key="y_col")
        plot_type = st.selectbox("Plot Type", ["Line","Scatter","Bar"], key="plot_type")
        if not plot_df.empty:
            fig, ax = plt.subplots(figsize=(10,6))
            if plot_type=="Line": sns.lineplot(data=plot_df, x=x_col, y=y_col, ax=ax)
            elif plot_type=="Scatter": sns.scatterplot(data=plot_df, x=x_col, y=y_col, ax=ax)
            elif plot_type=="Bar": sns.barplot(data=plot_df, x=x_col, y=y_col, ax=ax)
            st.pyplot(fig); plt.close(fig)
        else:
            st.warning("No rows selected for plotting.")
    
if aqi_api:
    st.markdown("---")
    st.subheader("🌫️ Realtime AQI")

    data = fetch_aqi()
    if not data or "data" not in data:
        st.warning("Unable to fetch AQI data. Check your API key or rate limits.")
    else:
        city = data["data"]["city"]
        state = data["data"]["state"]
        country = data["data"]["country"]
        pollution = data["data"]["current"]["pollution"]
        weather = data["data"]["current"]["weather"]

        st.markdown(f"**Location:** {city}, {state}, {country}")
        st.metric("AQI (US Standard)", pollution.get("aqius","N/A"))
        st.write(f"Primary pollutant: **{pollution.get('mainus','N/A').upper()}**")

        col1, col2, col3 = st.columns(3)
        col1.metric("Temperature", f"{weather.get('tp','N/A')}°C")
        col2.metric("Humidity", f"{weather.get('hu','N/A')}%")
        col3.metric("Wind", f"{weather.get('ws','N/A')} m/s")

        st.write("**Pollutant Concentrations:**")
        st.write(f"PM2.5: {pollution.get('p2','N/A')} µg/m³ | PM10: {pollution.get('p1','N/A')} µg/m³")
        
