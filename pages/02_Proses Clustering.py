import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans


data = pd.read_csv('dataset\\budidaya_2021.csv')
data['tahun'] = data['tahun'].astype(str)
data.rename(index=str, columns={
    'kabupaten/kota' : 'kabupaten', 
    'jenis_komoditas_perikanan' : 'jenis komoditas',
    'volume_produksi' : 'volume (Ton)',
    'nilai_produksi' : 'nilai (000 Rp)'
}, inplace=True)

def Perikanan():
    st.title("Clustering Produksi Perikanan")
    st.write("Data produksi perikanan Jawa Timur (Sebelum proses clustering)")
    
    train = data.drop(['kabupaten', 'tahun', 'jenis komoditas'], axis=1)
    st.write(data)
    scaler = MinMaxScaler(feature_range=(0, 1))
    train = scaler.fit_transform(train)
    
    st.sidebar.subheader("Nilai jumlah K")
    clust = st.sidebar.slider("Pilih jumlah cluster :", 2,10,3,1)
    
    st.sidebar.write("Metode Elbow")
    inertia = []
    for i in range(1,11):
        km = KMeans(n_clusters=i).fit(train)
        inertia.append(km.inertia_)
        
    fig, ax = plt.subplots(figsize=(12,8))
    sns.lineplot(x=list(range(1,11)), y=inertia, ax=ax)
    ax.set_title('mencari elbow')
    ax.set_xlabel('clusters')
    ax.set_ylabel('inertia')
    st.sidebar.write(fig)
    
    st.write("Visualisasi Cluster")
    kmeans = KMeans(n_clusters=clust, init='k-means++', random_state=0)
    kmeans.fit(train)
    fig, ax = plt.subplots(figsize=(10,5))
    ax.scatter(train[:,0], train[:,1], c=kmeans.labels_, cmap='rainbow')
    for i, centroid in enumerate(kmeans.cluster_centers_):
        ax.scatter(centroid[0], centroid[1], cmap='rainbow', marker='*', s=150, label='Centroid ' + str(i))
    ax.legend()
    st.pyplot(fig)

    st.write("Data produksi perikanan Jawa Timur (Sesudah proses clustering)")
    dfs = {}
    for cluster in set(kmeans.labels_):
            dfs[cluster] = data[kmeans.labels_ == cluster]
    for cluster, df in dfs.items():
        st.markdown(f'Cluster {cluster}')
        st.write(df)
    
def Bandeng():
    st.title("Clustering Produksi Bandeng")
    st.write("Data produksi Bandeng Jawa Timur (Sebelum proses clustering)")
    
    data_bandeng = data[data['jenis komoditas'] == 'Bandeng']
    train_bandeng = data_bandeng.drop(['kabupaten', 'tahun', 'jenis komoditas'], axis=1)
    st.write(data_bandeng)
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_bandeng = scaler.fit_transform(train_bandeng)
    
    st.sidebar.subheader("Nilai jumlah K")
    clust = st.sidebar.slider("Pilih jumlah cluster :", 2,10,3,1)
    
    st.sidebar.write("Metode Elbow")
    
    inertia_bandeng = []
    for i in range(1,11):
        km = KMeans(n_clusters=i).fit(train_bandeng)
        inertia_bandeng.append(km.inertia_)
        
    fig2, ax = plt.subplots(figsize=(12,8))
    sns.lineplot(x=list(range(1,11)), y=inertia_bandeng, ax=ax)
    ax.set_title('mencari elbow')
    ax.set_xlabel('clusters')
    ax.set_ylabel('inertia')
    st.sidebar.write(fig2)
    
    st.write("Visualisasi Cluster")
    
    kmeans_bandeng = KMeans(n_clusters=clust, init='k-means++', random_state=0)
    kmeans_bandeng.fit(train_bandeng)

    fig, ax = plt.subplots(figsize=(10,5))
    ax.scatter(train_bandeng[:,0], train_bandeng[:,1], c=kmeans_bandeng.labels_, cmap='rainbow')

    for i, centroid in enumerate(kmeans_bandeng.cluster_centers_):
        ax.scatter(centroid[0], centroid[1], cmap='rainbow', marker='*', s=150, label='Centroid ' + str(i))
    ax.legend()
    st.pyplot(fig)

    st.write("Data produksi perikanan Jawa Timur (Sesudah proses clustering)")
    dfs = {}
    for cluster in set(kmeans_bandeng.labels_):
            dfs[cluster] = data_bandeng[kmeans_bandeng.labels_ == cluster]
    for cluster, df in dfs.items():
        st.markdown(f'Cluster {cluster}')
        st.write(df)
    
def Lele():
    st.title("Clustering Produksi Lele")
    st.write("Data produksi Lele Jawa Timur (Sebelum proses clustering)")

    data_lele= data[data['jenis komoditas'] == 'Lele']
    train_lele = data_lele.drop(['kabupaten', 'tahun', 'jenis komoditas'], axis=1)
    st.write(data_lele)
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_lele = scaler.fit_transform(train_lele)
    
    st.sidebar.subheader("Nilai jumlah K")
    clust = st.sidebar.slider("Pilih jumlah cluster :", 2,10,3,1)
    
    st.sidebar.write("Metode Elbow")
    inertia_lele = []
    for i in range(1,11):
        km = KMeans(n_clusters=i).fit(train_lele)
        inertia_lele.append(km.inertia_)
        
    fig3, ax = plt.subplots(figsize=(12,8))
    sns.lineplot(x=list(range(1,11)), y=inertia_lele, ax=ax)
    ax.set_title('mencari elbow')
    ax.set_xlabel('clusters')
    ax.set_ylabel('inertia')
    st.sidebar.write(fig3)
    
    st.write("Visualisasi Cluster")
    
    kmeans_lele = KMeans(n_clusters=clust, init='k-means++', random_state=0)
    kmeans_lele.fit(train_lele)

    fig, ax = plt.subplots(figsize=(10,5))
    ax.scatter(train_lele[:,0], train_lele[:,1], c=kmeans_lele.labels_, cmap='rainbow')
    for i, centroid in enumerate(kmeans_lele.cluster_centers_):
        ax.scatter(centroid[0], centroid[1], cmap='rainbow', marker='*', s=150, label='Centroid ' + str(i))
    ax.legend()
    st.pyplot(fig)

    st.write("Data produksi perikanan Jawa Timur (Sesudah proses clustering)")
    dfs = {}
    for cluster in set(kmeans_lele.labels_):
            dfs[cluster] = data_lele[kmeans_lele.labels_ == cluster]
    for cluster, df in dfs.items():
        st.markdown(f'Cluster {cluster}')
        st.write(df)
def Udang_Vannamei():
    st.title("Clustering Produksi Udang Vannamei")
    st.write("Data produksi Udang Vannamei Jawa Timur (Sebelum proses clustering)")

    data_udang= data[data['jenis komoditas'] == 'Udang Vannamei']
    train_udang = data_udang.drop(['kabupaten', 'tahun', 'jenis komoditas'], axis=1)
    st.write(data_udang)
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_udang = scaler.fit_transform(train_udang)
    
    st.sidebar.subheader("Nilai jumlah K")
    clust = st.sidebar.slider("Pilih jumlah cluster :", 2,10,3,1)
    
    st.sidebar.write("Metode Elbow")
    
    inertia_udang = []
    for i in range(1,11):
        km = KMeans(n_clusters=i).fit(train_udang)
        inertia_udang.append(km.inertia_)
        
    fig4, ax = plt.subplots(figsize=(12,8))
    sns.lineplot(x=list(range(1,11)), y=inertia_udang, ax=ax)
    ax.set_title('mencari elbow')
    ax.set_xlabel('clusters')
    ax.set_ylabel('inertia')
    st.sidebar.write(fig4)
    
    st.write("Visualisasi Cluster")
    
    kmeans_udang = KMeans(n_clusters=clust, init='k-means++', random_state=0)
    kmeans_udang.fit(train_udang)

    fig, ax = plt.subplots(figsize=(10,5))
    ax.scatter(train_udang[:,0], train_udang[:,1], c=kmeans_udang.labels_, cmap='rainbow')

    for i, centroid in enumerate(kmeans_udang.cluster_centers_):
        ax.scatter(centroid[0], centroid[1], cmap='rainbow', marker='*', s=150, label='Centroid ' + str(i))
    ax.legend()
    st.pyplot(fig)

    st.write("Data produksi perikanan Jawa Timur (Sesudah proses clustering)")
    dfs = {}
    for cluster in set(kmeans_udang.labels_):
            dfs[cluster] = data_udang[kmeans_udang.labels_ == cluster]
    for cluster, df in dfs.items():
        st.markdown(f'Cluster {cluster}')
        st.write(df)


nama_cluster = {
    "Perikanan": Perikanan,
    "Bandeng": Bandeng,
    "Lele": Lele,
    "Udang Vannamei": Udang_Vannamei
} 

nama_dataset = st.sidebar.selectbox("Pilih Komoditas",nama_cluster.keys())
nama_cluster[nama_dataset]()


