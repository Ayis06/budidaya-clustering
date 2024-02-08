import streamlit as st
import pandas as pd


st.sidebar.markdown("Beranda")


st.title("Web Apps K-Means Clustering Budidaya Perikanan Jawa Timur")
st.write("""
Aplikasi berbasis web untuk mengelompokkan (Clustering) daerah produksi perikanan budidaya Jawa Timur tahun 2021 dengan 3 komoditas utama (Bandeng, Lele, Udang Vannamei).
""")
st.write("""
Data yang digunakan berikut diperoleh dari [BPS Jawa Timur](https://jatim.bps.go.id/subject/56/perikanan.html)
""")


data = pd.read_csv('dataset\\budidaya_2021.csv')
data['tahun'] = data['tahun'].astype(str)
data.rename(index=str, columns={
    'kabupaten/kota' : 'Kabupaten',
    'tahun' : 'Tahun',
    'jenis_komoditas_perikanan' : 'Jenis Komoditas',
    'volume_produksi' : 'Volume (Ton)',
    'nilai_produksi' : 'Nilai (000 Rp)'
}, inplace=True)
st.write(data)



