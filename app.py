import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import pydeck as pdk
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

st.set_page_config(page_title="Clustering Stunting", layout="wide")

# Load style dari index.html
with open("style/index.html") as f:
    st.markdown(f.read(), unsafe_allow_html=True)

st.title("📊 Dashboard Clustering Desa-Desa di Sulawesi Utara")

# Upload file
uploaded_file = st.file_uploader("📂 Upload file Excel", type=["xlsx"])

if uploaded_file:
    try:
        original_data = pd.read_excel(uploaded_file)
        data = original_data.copy()

        cols_penyediaan = [
            'Desa_Melaksanakan_rembuk_stunting_desa_dengan_melibatkan_UPTD',
            'Terdapat_Pembentukan_RDS/TPPS',
            'Aktivitas_rutin_Penyelenggaraan_posyandu_',
            'Aktivitas_rutin_Penyelenggaraan_kelas_Bina_Keluarga_Balita_',
            'Aktivitas_rutin_Penyelenggaraan_PAUD',
            'Terdapat_Pengembangan_Program_Ketahanan_Pangan'
        ]

        def mapping_biner(val):
            if pd.isna(val): return 0
            val = str(val).lower().strip()
            val = re.sub(r'\s+', ' ', val)
            return 1 if val in {'ya', 'ada', 'rutin tiap bulan', '1'} else 0

        for col in cols_penyediaan:
            data[col] = data[col].apply(mapping_biner)

        cols_penerimaan = [
            'Total_Layanan_Remaja_Putri_diterima',
            'Total_Layanan_Ibu_Hamil_dan_ibu_hamil_KEK_diterima',
            'Total_Layanan_Anak_(0-59_bulan)_diterima',
            'Total_Layanan_Keluarga_memiliki_sasaran_stunting_dan_keluarga_beresiko_stunting_diterima',
            'Total_Layanan_Calon_Pengantin_dan_calon_pasangan_usia_subur_diterima',
        ]

        fitur_clustering = cols_penerimaan + cols_penyediaan + ['Jumlah_alokasi_anggaran_untuk_mendukung_kegiatan_stunting']
        data = data[~(data[fitur_clustering].sum(axis=1) == 0)]

        scaler = MinMaxScaler()
        X_normalized = scaler.fit_transform(data[fitur_clustering])
        X_df = pd.DataFrame(X_normalized, columns=fitur_clustering)

        kmeans = KMeans(n_clusters=3, random_state=42)
        data['Cluster'] = kmeans.fit_predict(X_df)

        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_df)

        st.sidebar.markdown("<h3 style='text-align:center; color:gray;'>Clustering Desa Sulawesi Utara</h3>", unsafe_allow_html=True)
        st.sidebar.markdown("---")
        selected_view = st.sidebar.radio("", [
            "📊 Dashboard", "🗂️ Data Awal", "🧮 Preprocessed Data", "📈 Statistik Rata-rata per Cluster"
        ])
        st.sidebar.markdown("---")

        if selected_view == "📊 Dashboard":
            col1, col2, col3 = st.columns(3)
            col1.metric("📌 Jumlah Desa", len(data))
            col2.metric("💉 Total Layanan", int(data[cols_penerimaan].sum().sum()))
            col3.metric("💰 Total Anggaran", f"Rp{int(data['Jumlah_alokasi_anggaran_untuk_mendukung_kegiatan_stunting'].sum()):,}")

            st.subheader("🗺️ Peta Interaktif Klasterisasi Desa")
            if 'LONGITUDE' in data.columns and 'LATITUDE' in data.columns:
                cluster_colors = {0: [255, 0, 0], 1: [0, 255, 0], 2: [0, 0, 255]}
                data['color'] = data['Cluster'].map(cluster_colors)

                layer = pdk.Layer("ScatterplotLayer", data=data,
                                  get_position='[LONGITUDE, LATITUDE]', get_fill_color='color', get_radius=500, pickable=True)
                view_state = pdk.ViewState(latitude=data['LATITUDE'].mean(), longitude=data['LONGITUDE'].mean(), zoom=8)
                st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state, map_style=None,
                                         tooltip={"text": "Desa: {NAMA_DESA}\nCluster: {Cluster}"}))

            col4, col5 = st.columns(2)
            with col4:
                st.markdown("<h3 class='centered-title'>Distribusi Desa per Cluster</h3>", unsafe_allow_html=True)
                distribusi = data.groupby('Cluster')['NAMA_DESA'].count().reset_index(name='Jumlah Desa')
                fig, ax = plt.subplots()
                sns.barplot(data=distribusi, x='Cluster', y='Jumlah Desa', palette='Set2', ax=ax)
                st.pyplot(fig)
            with col5:
                st.markdown("<h3 class='centered-title'>PCA 2D</h3>", unsafe_allow_html=True)
                fig2, ax2 = plt.subplots()
                sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=data['Cluster'], palette='tab10', s=80, ax=ax2)
                ax2.set_xlabel("Komponen 1")
                ax2.set_ylabel("Komponen 2")
                st.pyplot(fig2)

            st.subheader("📖 Daftar Nama Desa per Cluster")
            cluster_dfs = [data[data['Cluster'] == i]['NAMA_DESA'].reset_index(drop=True) for i in range(3)]
            max_len = max(len(df) for df in cluster_dfs)
            for i in range(3):
                cluster_dfs[i] = cluster_dfs[i].reindex(range(max_len))
            merged_df = pd.concat(cluster_dfs, axis=1)
            merged_df.columns = ['Cluster 0', 'Cluster 1', 'Cluster 2']
            st.dataframe(merged_df, use_container_width=True)

        elif selected_view == "🗂️ Data Awal":
            st.header("🗂️ Data Awal")
            st.dataframe(original_data.head(), use_container_width=True)
            st.subheader("🔎 Informasi Missing Value")
            missing_summary = original_data.isnull().sum()
            st.dataframe(missing_summary[missing_summary > 0].reset_index().rename(columns={0: 'Jumlah Missing', 'index': 'Kolom'}))
            st.subheader("🔠 Variabel String")
            string_cols = original_data.select_dtypes(include='object').columns.tolist()
            st.write(f"Jumlah variabel string: {len(string_cols)}")
            st.write(string_cols)

        elif selected_view == "🧮 Preprocessed Data":
            st.header("🧮 Data Setelah Preprocessing")
            st.dataframe(X_df.head(10), use_container_width=True)

        elif selected_view == "📈 Statistik Rata-rata per Cluster":
            st.header("📈 Statistik Rata-rata per Cluster")
            rata_cluster = data.groupby('Cluster')[fitur_clustering].mean().reset_index()
            st.dataframe(rata_cluster, use_container_width=True)

    except Exception as e:
        st.error(f"🚨 Terjadi error saat memproses data: {e}")
else:
    st.warning("📂 Silakan upload file Excel untuk memulai analisis.")
