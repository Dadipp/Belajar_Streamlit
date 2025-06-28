import streamlit as st 
import pandas as pd
import numpy as np 
import pickle 
import plotly.express as px 
import plotly.graph_objects as go 
from datetime import datetime, timedelta

# --- Konfigurasi Halaman ---
st.set_page_config(
    page_title="Dashboard Analysis Penjualan",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -- Fungsi untuk Load Data Penjualan --
@st.cache_data
@st.cache_data
def load_data():
    df = pd.read_csv("data/data_dummy_retail_store.csv")
    return df

df_sales = load_data()

# -- Fungsi untuk Load Model --
@st.cache_resource
def load_model():
    with open("E:\VScode\HANDSON_32B\Belajar_Streamlit\models\model_sales.pkl", "rb") as f:
        sales_prediction_model, model_features, base_month_ordinal = pickle.load(f)
    return sales_prediction_model, model_features, base_month_ordinal

sales_prediction_model, model_features, base_month_ordinal = load_model()

# --- Judul Halaman ---
st.title("Dashboard Penjualan Toko Online")
st.markdown(
    "Dashboard interaktif ini berisi gambaran **performa penjualan**, **tren**, **distribusi**, dan **fitur prediksi sederhana**"
)
st.markdown("---")  # Garis pembatas horizontal

# -- Sidebar Halaman --
st.sidebar.header("Pengaturan & Navigasi")

pilihan_halaman = st.sidebar.radio(
    "Pilih Halaman",
    ("Overview Dashboard", "Prediksi Penjualan")
)

# Filter untuk Halamman Overview Dashboard
if pilihan_halaman == "Overview Dashboard":
    st.sidebar.markdown("### Filter Data Dashboard")

    # Filter tanggal
    df_sales = load_data()
    df_sales['Tanggal_Pesanan'] = pd.to_datetime(df_sales['Tanggal_Pesanan'], errors='coerce')

    min_date = df_sales['Tanggal_Pesanan'].min().date()
    max_date = df_sales['Tanggal_Pesanan'].max().date()

    date_range = st.sidebar.date_input(
        "Pilih Rentang Tanggal:",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    if len(date_range) == 2:
        start_date_filter = pd.to_datetime(date_range[0])
        end_date_filter = pd.to_datetime(date_range[1])
        filtered_df = df_sales[(df_sales['Tanggal_Pesanan'] >= start_date_filter) & (df_sales['Tanggal_Pesanan'] <= end_date_filter)]
    else:
        filtered_df = df_sales

# -- Filter wilayah --
    selected_regions = st.sidebar.multiselect(
        "Pilih Wilayah:",
        options=df_sales['Wilayah'].unique().tolist(),
        default=df_sales['Wilayah'].unique().tolist()
    )  
    filtered_df = filtered_df[filtered_df['Wilayah'].isin(selected_regions)]

    # Filter kategori produk
    selected_categories = st.sidebar.multiselect(
        "Pilih Kategori Produk",
        options=df_sales['Kategori'].unique().tolist(),
        default=df_sales['Kategori'].unique().tolist()
    )
    filtered_df = filtered_df[filtered_df['Kategori'].isin(selected_categories)]
else:  # untuk halaman prediksi, pakai df_sales.copy() atau tidak ada filter
    filtered_df = df_sales.copy()

if pilihan_halaman == "Overview Dashboard":
    # Metrics utama
    st.subheader("Ringkasan Performa Penjualan")

    col1, col2, col3, col4 = st.columns(4)

    # agregat metrics
    total_sales = filtered_df['Total_Penjualan'].sum()
    total_orders = filtered_df['OrderID'].nunique()
    avg_order_value = total_sales / total_orders if total_orders > 0 else 0
    total_products_sold = filtered_df['Jumlah'].sum()

    with col1:
        st.metric(label="Total Penjualan", value=f"Rp {total_sales:,.2f}")
    with col2:
        st.metric(label="Jumlah Pesanan", value=f"Rp {total_orders:,.2f}")
    with col3:
        st.metric(label="Avg. Order Value", value=f"Rp {avg_order_value:,.2f}")
    with col4:
        st.metric(label="Jumlah Produk Terjual", value=f"Rp {total_products_sold:,.2f}")

    st.markdown("---")
    st.subheader("📊 Tren Penjualan")

    sales_trend = filtered_df.groupby('Tanggal_Pesanan')['Total_Penjualan'].sum().reset_index()
    fig_trend = px.line(sales_trend, x='Tanggal_Pesanan', y='Total_Penjualan',
                        title="Tren Total Penjualan per Tanggal",
                        labels={"Tanggal_Pesanan": "Tanggal", "Total_Penjualan": "Total Penjualan"})
    st.plotly_chart(fig_trend, use_container_width=True)

    # Penjualan & Produk Terlaris
    st.subheader("Top Product & Distribusi Penjualan")

    col_vis1, col_vis2 = st.columns(2)

    with col_vis1:
        st.write("#### Top 10 Products")

        # agregat
        top_products_sale = filtered_df.groupby(['Produk'])['Total_Penjualan'].sum().nlargest(10).reset_index()

        fig_top_products = px.bar(
            top_products_sale,
            x='Total_Penjualan',
            y='Produk',
            orientation='h',
            title='Top 10 Produk Berdasarkan Total Penjualan',
            color='Total_Penjualan',
            color_continuous_scale=px.colors.sequential.Plasma[::-1], # gradasi warna,
            height=400
        )

        fig_top_products.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_top_products, use_container_width=True)
    with col_vis2:
        st.write("#### Distribusi Penjualan per Kategori")

        # Agregasi total penjualan berdasarkan kategori
        sales_by_category = filtered_df.groupby('Kategori')['Total_Penjualan'].sum().reset_index()

        # Buat pie chart (donut style) berdasarkan proporsi penjualan tiap kategori
        fig_category_pie = px.pie(
            sales_by_category,
            values='Total_Penjualan',    # Nilai yang diplot (besarannya)
            names='Kategori',            # Label di pie chart
            title='Proporsi Penjualan per Kategori',
            hole=0.3,                    # Membuat pie menjadi donut chart (ada lubangnya)
            color_discrete_sequence=px.colors.qualitative.Set2  # Skema warna yang friendly
        )

        # Tampilkan chart di Streamlit
        st.plotly_chart(fig_category_pie, use_container_width=True)

    st.subheader("💳 Penjualan Berdasarkan Metode Pembayaran dan Wilayah")
    grouped_payment = filtered_df.groupby(['Metode_Pembayaran', 'Wilayah'])['Total_Penjualan'].sum().reset_index()
    fig_payment = px.bar(grouped_payment, x='Total_Penjualan', y='Wilayah',
                         color='Metode_Pembayaran', title="Penjualan per Metode Pembayaran & Wilayah",
                         barmode='group')
    st.plotly_chart(fig_payment, use_container_width=True)

    st.subheader("🧾 Raw Data")
    st.dataframe(filtered_df)
