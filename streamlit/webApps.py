import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
from pycaret.regression import load_model, predict_model

model = load_model('PyCaret_CreditCard_model')

def load_credit_card_data():
    # Ganti path dengan path dataset yang sesuai
    data_path = "D:/kuliah/smt 4/MLOps/Application_Data.csv"
    df = pd.read_csv(data_path)
    return df

def display_credit_card_dataset():
    st.subheader("Dataset Aplikasi Kartu Kredit💳")
    credit_data = load_credit_card_data()
    st.write(credit_data)

# Fungsi untuk menampilkan tabel infografis dengan metrik model
def display_model_metrics():
    data = {
        'Model': ['Random Forest Regressor'],
        'MAE': [0.0012],
        'MSE': [0.0004],
        'RMSE': [0.0189],
        'R2': [0.9121],
        'RMSLE': [0.0134],
        'MAPE': [0.0006],
        'TT (Sec)': [0.2540]
    }

    # Membuat DataFrame dari data
    df = pd.DataFrame(data)

    # Menampilkan tabel menggunakan Streamlit
    st.subheader("Model Performance Metrics")
    st.write(df)

# Fungsi utama untuk menampilkan menu "Model"
def show_model_menu():
    st.markdown('<h1 class="title">Model Prediksi Aplikasi Kartu Kredit</h1>', unsafe_allow_html=True)
    st.subheader("Informasi Model")
    st.markdown("""
    Skor kredit merupakan teknik manajemen risiko yang banyak digunakan di sektor keuangan. Teknik ini menggunakan informasi pribadi dan data yang diberikan oleh para pemohon kartu kredit untuk memperkirakan kemungkinan kebangkrutan di masa depan dan pinjaman kartu kredit. Bank memiliki kewenangan untuk menentukan apakah akan memberikan atau tidak memberikan kartu kredit kepada pemohon. Skor kredit dapat mengestimasi tingkat risiko secara objektif.

    - Model yang digunakan: PyCaret Regresi
    - Algoritma: *Random Forest Regressor*
    - Data yang digunakan: [Data aplikasi kartu kredit](https://www.kaggle.com/datasets/rikdifos/credit-card-approval-prediction)
    """)
    
    # Tampilkan dataset
    display_credit_card_dataset()

def predict_status(data):
    # Make predictions using the loaded PyCaret model
    predictions = predict_model(model, data=data)
    
    # Display the predictions
    st.write(predictions)
    
    # Extract the predicted label (assuming 'prediction_label' column exists)
    if 'prediction_label' in predictions.columns:
        prediction_prob = predictions['prediction_label'].iloc[0]
        prediction = 1 if prediction_prob >= 0.5 else 0
        return prediction
    else:
        st.error("Prediction column 'prediction_label' not found in the model output.")
        return None

# Fungsi utama untuk aplikasi Streamlit
def main():
    st.set_page_config(page_title="Prediksi Aplikasi Kartu Kredit", page_icon=":credit_card:")

    st.markdown(
        """
        <style>
        .title {
            font-size: 40px;
            text-align: center;
            padding: 20px;
            color: #F5EFE6;
            background-color: #4F6F52;
            border-radius: 10px;
            box-shadow: 0px 0px 20px rgba(0, 0, 0, 0.3);
            margin-bottom : 20px;
        }
        </style>
        """, unsafe_allow_html=True)

    # Path ke gambar logo (ganti dengan path yang sesuai)
    logo_image = 'crediwiz_logo.png'
    
    # Tampilkan gambar logo di sidebar dengan ukuran tertentu
    st.sidebar.image(logo_image)

    # Menu di sidebar untuk model dan prediksi
    menu = st.sidebar.selectbox("Menu", options=["🔰Model", "🔎Prediksi"])


    if menu == "🔰Model":
        show_model_menu()
        display_model_metrics()
        st.sidebar.markdown("---")
        st.sidebar.info("Created by: Rike Anindhita🤗")

    elif menu == "🔎Prediksi":
        st.markdown('<h1 class="title">Prediksi Aplikasi Kartu Kredit</h1>', unsafe_allow_html=True)
        st.sidebar.title('Cara Penggunaan')
        st.sidebar.markdown("""
            - Pilih opsi menggunakan dropdown atau checkbox.
            - Isi informasi pemohon sesuai dengan data yang sebenarnya.
            - Tekan tombol 'Prediksi' untuk melihat hasil prediksi.
        """)
        st.sidebar.markdown("---")
        st.sidebar.info("Created by: Rike Anindhita🤗")

        # Create two columns for input
        col1, col2 = st.columns(2)

        # Input column 1
        with col1:
            applicant_gender = st.selectbox("Jenis Kelamin Pemohon", ["Laki-Laki", "Perempuan"], key='gender_input', help="Pilih jenis kelamin pemohon.")
            owned_car = st.selectbox("Memiliki Mobil?", ["Ya", "Tidak"], help="Pilih 'Ya' jika memiliki mobil, atau 'Tidak' jika tidak.")
            owned_realty = st.selectbox("Memiliki Properti?", ["Ya", "Tidak"], help="Pilih 'Ya' jika memiliki properti, atau 'Tidak' jika tidak.")
            income_type = st.selectbox("Jenis Pendapatan", ['Pekerja', 'Pekerja Asosiasi Komersial', 'Pegawai Negri', 'Pensiunan', 'Mahasiswa'], help="Pilih jenis pendapatan pemohon.")
            education_type = st.selectbox("Tingkat Pendidikan", ['Sekolah Menengah', 'Pendidikan Tinggi', 'Pendidikan Tinggi Tidak Lengkap', 'Sekolah Menengah Rendah', 'Gelar Akademik'], help="Pilih tingkat pendidikan pemohon.")
            family_status = st.selectbox("Status Keluarga", ['Menikah', 'Belum Menikah / Lajang', 'Pernikahan Sipil', 'Terpisah', 'Janda/Duda'], help="Pilih status keluarga pemohon.")
            housing_type = st.selectbox("Tipe Tempat Tinggal", ['Rumah/Apartemen', 'Bersama Orang Tua', 'Apartemen Pemerintah', 'Apartemen Sewaan', 'Apartemen Kantor', 'Apartemen Kerja Sama'], help="Pilih tipe tempat tinggal pemohon.")
            owned_mobile_phone = st.checkbox("Memiliki Telepon Genggam?", help="Centang jika pemohon memiliki telepon genggam.")
            owned_work_phone = st.checkbox("Memiliki Telepon Kantor?", help="Centang jika pemohon memiliki telepon kantor.")
            owned_phone = st.checkbox("Memiliki Telepon Tambahan?", help="Centang jika pemohon memiliki telepon tambahan.")
            owned_email = st.checkbox("Memiliki Alamat Email?", help="Centang jika pemohon memiliki alamat email.")

        # Input column 2
        with col2:
            job_title = st.selectbox("Jabatan Pekerjaan", ['Buruh', 'Staf Inti', 'Staf Penjualan', 'Sopir', 'Staf Teknis Keterampilan Tinggi', 'Akuntan', 'Staf Medis', 'Staf Masak', 'Staf Keamanan', 'Staf Kebersihan', 'Staf Pelayanan Pribadi', 'Buruh Keterampilan Rendah', 'Staf Pelayan/Pramugari', 'Sekretaris', 'Staf HR', 'Agen Properti', 'Staf IT', 'Jabatan Lainnya'], help="Pilih jabatan pekerjaan pemohon.")
            total_children = st.number_input("Jumlah Anak", value=0, help="Masukkan jumlah anak pemohon.")
            total_income = st.number_input("Total Pendapatan", value=0, help="Masukkan total pendapatan pemohon.")
            total_family_members = st.number_input("Jumlah Anggota Keluarga", value=1, help="Masukkan jumlah anggota keluarga pemohon.")
            applicant_age = st.number_input("Usia Pemohon", value=30, help="Masukkan usia pemohon.")
            years_of_working = st.number_input("Lama Bekerja (Tahun)", value=10, help="Masukkan tahun bekerja pemohon.")
            total_bad_debt = st.number_input("Total Hutang Buruk", value=0, help="Masukkan total pinjaman atau kewajiban keuangan yang digunakan untuk tujuan produktif atau investasi (pendidikan, investasi properti, atau pendanaan bisnis).")
            total_good_debt = st.number_input("Total Hutang Baik", value=0, help="Masukkan total pinjaman atau kewajiban keuangan yang digunakan untuk membiayai pengeluaran konsumtif (hutang kartu kredit untuk barang konsumtif atau liburan).")

        # Buat dictionary dari input pengguna
        input_data = {
            'Applicant_Gender': applicant_gender,
            'Owned_Car': 1 if owned_car == 'Ya' else 0,
            'Owned_Realty': 1 if owned_realty == 'Ya' else 0,
            'Total_Children': total_children,
            'Total_Income': total_income,
            'Income_Type': income_type,
            'Education_Type': education_type,
            'Family_Status': family_status,
            'Housing_Type': housing_type,
            'Owned_Mobile_Phone': 1 if owned_mobile_phone else 0,
            'Owned_Work_Phone': 1 if owned_work_phone else 0,
            'Owned_Phone': 1 if owned_phone else 0,
            'Owned_Email': 1 if owned_email else 0,
            'Job_Title': job_title,
            'Total_Family_Members': total_family_members,
            'Applicant_Age': applicant_age,
            'Years_of_Working': years_of_working,
            'Total_Bad_Debt': total_bad_debt,
            'Total_Good_Debt': total_good_debt
        }

        # Konversi data input menjadi DataFrame
        input_df = pd.DataFrame([input_data])

        # Lakukan prediksi ketika tombol 'Prediksi' ditekan
        if st.button('Prediksi'):
            prediction = predict_status(input_df)
            if prediction is not None:
                if prediction == 1:
                    st.success("✅Selamat! Pembuatan kartu kredit kemungkinan akan disetujui😄.")
                else:
                    st.error("❌Maaf, Pembuatan kartu kredit mungkin tidak disetujui😓.")



if __name__ == '__main__':
    main()