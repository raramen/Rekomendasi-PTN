import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Memuat dataset dan melakukan clustering
file_path = 'skorrrr.csv'
data = pd.read_csv(file_path)

# Mengonversi PRODI menjadi label numerik
label_encoder_prodi = LabelEncoder()
data['Prodi_Label'] = label_encoder_prodi.fit_transform(data['PRODI'])

# Menyiapkan fitur untuk clustering
features = data[['Min_Skor', 'Prodi_Label']]

# Menentukan jumlah cluster (K)
num_clusters = st.slider('Pilih Jumlah Cluster:', min_value=2, max_value=10, value=3)

# Menerapkan K-Means clustering
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
data['Cluster'] = kmeans.fit_predict(features)

# 2. Membangun antarmuka pengguna dengan Streamlit
st.title('Rekomendasi Universitas Berdasarkan Nilai dan Program Studi')

# Dropdown untuk memilih program studi
prodi_options = sorted(data['PRODI'].unique())  # Mengurutkan program studi dari A sampai Z
selected_prodi = st.selectbox('Pilih Program Studi:', prodi_options)

# Input untuk memasukkan nilai pengguna
user_score = st.number_input('Masukkan Nilai Anda:', min_value=0.0, max_value=1000.0, step=0.1)

# 3. Memproses input pengguna dan memberikan rekomendasi
if st.button('Tampilkan Rekomendasi'):
    # Filter data berdasarkan program studi yang dipilih
    filtered_data = data[data['PRODI'] == selected_prodi]
    
    # Menghitung jarak antara nilai pengguna dan nilai minimum dalam dataset
    filtered_data['Distance'] = abs(filtered_data['Min_Skor'] - user_score)
    
    # Mengurutkan berdasarkan jarak dan mengambil top 3
    top_n = filtered_data.sort_values(by='Distance').head(3)
    
    # Tampilkan hasil rekomendasi dengan menyesuaikan pesan sesuai jumlah hasil
    if len(top_n) > 0:
        st.write(f'Top {len(top_n)} Universitas:')
        st.write(top_n[['PTN', 'Min_Skor']])
    else:
        st.write('Tidak ada universitas yang cocok dengan nilai yang Anda masukkan.')

# 4. Visualisasi hasil clustering
st.subheader('Visualisasi Hasil Clustering')

# Membuat visualisasi scatter plot berdasarkan clustering
fig, ax = plt.subplots()
scatter = ax.scatter(data['Min_Skor'], data['Prodi_Label'], c=data['Cluster'], cmap='viridis', alpha=0.6, edgecolors='w', s=100)
ax.set_xlabel('Min Skor')
ax.set_ylabel('Prodi Label')
ax.set_title('Visualisasi Kluster Berdasarkan Program Studi')  # Perbaikan di sini

# Menampilkan plot
st.pyplot(fig)

# 5. Menampilkan jumlah program studi/universitas per cluster
st.subheader('Jumlah Program Studi per Cluster')
cluster_counts = data['Cluster'].value_counts().sort_index()

# Menampilkan tabel jumlah per cluster
st.table(cluster_counts)

# Visualisasi bar plot untuk distribusi cluster
fig2, ax2 = plt.subplots()
sns.barplot(x=cluster_counts.index, y=cluster_counts.values, ax=ax2, palette='viridis')

ax2.set_xlabel('Cluster')
ax2.set_ylabel('Jumlah Program Studi')
ax2.set_title('Distribusi Program Studi per Cluster')  # Perbaikan di sini

st.pyplot(fig2)
