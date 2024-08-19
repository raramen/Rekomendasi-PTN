import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Memuat dataset dan melakukan clustering
file_path = 'skorrrr.csv'
data = pd.read_csv(file_path)

# Konversi PRODI menjadi label numerik (untuk clustering)
label_encoder_prodi = LabelEncoder()
data['Cluster'] = label_encoder_prodi.fit_transform(data['PRODI'])

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
    top_3 = filtered_data.sort_values(by='Distance').head(3)
    
    # Tampilkan hasil rekomendasi
    st.write('Top 3 Universitas:')
    st.write(top_3[['PTN', 'Min_Skor']])

# 4. Visualisasi hasil clustering
st.subheader('Visualisasi Hasil Clustering')

# Membuat visualisasi scatter plot berdasarkan clustering
fig, ax = plt.subplots()
scatter = ax.scatter(data['Min_Skor'], data['Cluster'], c=data['Cluster'], cmap='viridis', alpha=0.6, edgecolors='w', s=100)

ax.set_xlabel('Min Skor')
ax.set_ylabel('Cluster')
ax.set_title('Visualisasi Kluster Berdasarkan Program Studi')

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
ax2.set_title('Distribusi Program Studi per Cluster')

st.pyplot(fig2)