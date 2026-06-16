import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# 1. Load Model & Data
model = tf.keras.models.load_model('skin_model.h5')
class_names = ['acne', 'dry', 'normal', 'oily']

saran_skincare = {
    "acne": {
        "deskripsi": "Kondisi kulit yang rentan mengalami penyumbatan pori-pori dan peradangan. Fokus utama adalah meredakan kemerahan, membersihkan sebum berlebih, dan membunuh bakteri penyebab jerawat.",
        "bahan": "Salicylic Acid (BHA), Niacinamide, Tea Tree, Centella Asiatica",
        "produk": [
            {"step": "Step 1: Cleanser", "nama": "COSRX Salicylic Acid Cleanser", "link": "https://shopee.co.id/search?keyword=cosrx+salicylic+acid+cleanser"},
            {"step": "Step 2: Toner", "nama": "Npure Centella Asiatica Toner", "link": "https://shopee.co.id/search?keyword=npure+centella+asiatica+toner"},
            {"step": "Step 3: Serum", "nama": "Somethinc 2% BHA Serum", "link": "https://shopee.co.id/search?keyword=somethinc+bha+serum"},
            {"step": "Step 4: Moisturizer", "nama": "Skintific Acne Gel", "link": "https://shopee.co.id/search?keyword=skintific+acne+gel"},
            {"step": "Step 5: Sunscreen", "nama": "Azarine Sunscreen Gel", "link": "https://shopee.co.id/search?keyword=azarine+sunscreen+gel"}
        ]
    },
    "dry": {
        "deskripsi": "Kondisi kulit yang kekurangan kadar hidrasi dan kelembapan alami. Fokus utama adalah memperkuat lapisan pelindung kulit (skin barrier) dan mengunci kelembapan sepanjang hari.",
        "bahan": "Hyaluronic Acid, Ceramide, Glycerin, Squalane",
        "produk": [
            {"step": "Step 1: Cleanser", "nama": "Cetaphil Gentle Cleanser", "link": "https://shopee.co.id/search?keyword=cetaphil+gentle+cleanser"},
            {"step": "Step 2: Toner", "nama": "Skintific 4D Toner", "link": "https://shopee.co.id/search?keyword=skintific+4d+toner"},
            {"step": "Step 3: Serum", "nama": "The Ordinary Hyaluronic Acid", "link": "https://shopee.co.id/search?keyword=the+ordinary+hyaluronic+acid"},
            {"step": "Step 4: Moisturizer", "nama": "Skintific 5X Ceramide", "link": "https://shopee.co.id/search?keyword=skintific+ceramide+moisturizer"},
            {"step": "Step 5: Sunscreen", "nama": "Carasun UV Protector", "link": "https://shopee.co.id/search?keyword=carasun+uv+protector"}
        ]
    },
    "normal": {
        "deskripsi": "Kondisi kulit yang memiliki keseimbangan hidrasi dan produksi sebum yang baik. Fokus utama adalah mempertahankan kesehatan kulit, mencerahkan, dan memberikan perlindungan antioksidan.",
        "bahan": "Vitamin C, Panthenol, Antioxidants, Aloe Vera",
        "produk": [
            {"step": "Step 1: Cleanser", "nama": "Sensatia Botanicals Cleanser", "link": "https://shopee.co.id/search?keyword=sensatia+botanicals+cleanser"},
            {"step": "Step 2: Toner", "nama": "Anua Heartleaf Toner", "link": "https://shopee.co.id/search?keyword=anua+heartleaf+toner"},
            {"step": "Step 3: Serum", "nama": "Skintific Vitamin C Serum", "link": "https://shopee.co.id/search?keyword=skintific+vitamin+c+serum"},
            {"step": "Step 4: Moisturizer", "nama": "The Originote Moisturizer", "link": "https://shopee.co.id/search?keyword=the+originote+moisturizer"},
            {"step": "Step 5: Sunscreen", "nama": "Skin Aqua UV Gel", "link": "https://shopee.co.id/search?keyword=skin+aqua+uv+gel"}
        ]
    },
    "oily": {
        "deskripsi": "Kondisi kulit dengan produksi kelenjar minyak (sebum) yang aktif secara berlebihan. Fokus utama adalah mengontrol kilap minyak wajah, membersihkan pori secara mendalam, tanpa membuat kulit dehidrasi.",
        "bahan": "Niacinamide, Clay / Bentonite, Green Tea, Salicylic Acid",
        "produk": [
            {"step": "Step 1: Cleanser", "nama": "Senka Perfect Whip Fresh", "link": "https://shopee.co.id/search?keyword=senka+perfect+whip+fresh"},
            {"step": "Step 2: Toner", "nama": "Benton Green Tea Toner", "link": "https://shopee.co.id/search?keyword=benton+green+tea+toner"},
            {"step": "Step 3: Serum", "nama": "Somethinc Niacinamide", "link": "https://shopee.co.id/search?keyword=somethinc+niacinamide"},
            {"step": "Step 4: Moisturizer", "nama": "Glad2Glow Soothing Gel", "link": "https://shopee.co.id/search?keyword=glad2glow+soothing+gel"},
            {"step": "Step 5: Sunscreen", "nama": "Biore UV Aqua Rich", "link": "https://shopee.co.id/search?keyword=biore+uv+aqua+rich"}
        ]
    }
}

# 2. Page Configuration & Custom CSS injection
st.set_page_config(page_title="Skin Glow Analyzer", page_icon="✨", layout="centered")

st.markdown("""
    <style>
    .main-title {
        font-family: 'Inter', sans-serif;
        font-weight: 800;
        color: #1E1E24;
        text-align: center;
        margin-bottom: 5px;
    }
    .subtitle {
        font-family: 'Inter', sans-serif;
        color: #6C757D;
        text-align: center;
        font-size: 1.1rem;
        margin-bottom: 30px;
    }
    .card-product {
        background-color: #F8F9FA;
        border: 1px solid #E9ECEF;
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.02);
        transition: transform 0.2s ease;
    }
    .card-product:hover {
        transform: translateY(-2px);
        border-color: #CED4DA;
    }
    .step-tag {
        font-size: 0.8rem;
        font-weight: 700;
        color: #495057;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 6px;
    }
    .product-title {
        font-size: 1rem;
        font-weight: 600;
        color: #212529;
        margin-bottom: 12px;
        min-height: 48px;
    }
    .ingredient-badge {
        background-color: #E3F2FD;
        color: #0D47A1;
        padding: 6px 12px;
        border-radius: 20px;
        font-size: 0.9rem;
        display: inline-block;
        margin-right: 8px;
        margin-bottom: 8px;
        font-weight: 500;
    }
    </style>
""", unsafe_allow_html=True)

# 3. Header Section
st.markdown("<h1 class='main-title'>✨ Skin Glow Analyzer</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Temukan panduan perawatan kulit yang personal dan cerdas</p>", unsafe_allow_html=True)

# 4. Sidebar Navigation
st.sidebar.markdown("### 🛠️ Menu Dashboard")
mode = st.sidebar.radio("Metode Pemeriksaan:", ["Deteksi AI (Foto)", "Input Manual Jenis Kulit"])

result = None
confidence_scores = None

if mode == "Deteksi AI (Foto)":
    st.markdown("### 📷 Analisis Citra Digital Wajah")
    
    with st.expander("💡 Petunjuk Pengambilan Foto (Mohon Dibaca)", expanded=True):
        st.write("""
        * **Kondisi Wajah:** Wajib bersih dari produk riasan wajah (*bare face*).
        * **Pencahayaan:** Gunakan cahaya alami atau ruangan yang terang merata (hindari bayangan).
        * **Fokus Kamera:** Ambil foto secara tegak lurus (frontalis) dan pastikan gambar tidak buram.
        """)
        
    metode = st.radio("Metode pengambilan gambar:", ["📁 Unggah File", "📷 Ambil dengan Kamera"], horizontal=True)
    
    if metode == "📁 Unggah File":
        uploaded_file = st.file_uploader("Pilih file foto wajah Anda", type=["jpg", "jpeg", "png"])
    else:
        uploaded_file = st.camera_input("Posisikan wajah Anda di tengah kamera")
        
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(image, caption="Foto Berhasil Dimuat", use_container_width=True)
            
        with col2:
            with st.spinner("Menganalisis tekstur dan matriks kulit..."):
                img = image.convert('RGB').resize((224, 224))
                img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
                prediction = model.predict(img_array)[0]
                
                idx_tertinggi = np.argmax(prediction)
                result = class_names[idx_tertinggi]
                confidence_scores = prediction
                
                st.markdown("#### **Hasil Analisis Struktur AI:**")
                st.markdown(f"Tipe Kulit Dominan: **{result.upper()}**")
                
                for name, score in zip(class_names, confidence_scores):
                    st.write(f"- {name.capitalize()}: {score*100:.1f}%")
                    st.progress(float(score))

else:
    st.markdown("### 📝 Input Manual Karakteristik Kulit")
    pilihan = st.selectbox("Pilih kategori kondisi kulit Anda saat ini:", [c.capitalize() for c in class_names])
    result = pilihan.lower()

# 5. Hasil Analisis & Rekomendasi Skincare
if result:
    st.markdown("---")
    st.markdown(f"## 🛍️ Panduan Komprehensif: **{result.upper()} SKIN**")
    st.markdown(f"*{saran_skincare[result]['deskripsi']}*")
    
    st.markdown("#### **Kombinasi Bahan Aktif Rekomendasi:**")
    for bahan in saran_skincare[result]['bahan'].split(', '):
        st.markdown(f"<span class='ingredient-badge'>{bahan}</span>", unsafe_allow_html=True)
    
    st.markdown("<br>#### **Saran Urutan Pemakaian Produk (5-Step Routine):**", unsafe_allow_html=True)
    
    col_p1, col_p2, col_p3 = st.columns(3)
    col_p4, col_p5, _ = st.columns(3)
    
    semua_kolom = [col_p1, col_p2, col_p3, col_p4, col_p5]
    
    for idx, prod in enumerate(saran_skincare[result]["produk"]):
        with semua_kolom[idx]:
            st.markdown(f"""
                <div class='card-product'>
                    <div class='step-tag'>{prod['step']}</div>
                    <div class='product-title'>{prod['nama']}</div>
                </div>
            """, unsafe_allow_html=True)
            st.link_button("Cek Detail Produk", prod['link'], use_container_width=True)

    # 6. Catatan Medis & Footer Keamanan Data
    st.markdown("<br><hr>", unsafe_allow_html=True)
    st.caption("""
        **⚠️ Pernyataan Batasan Tanggung Jawab (Disclaimer):**  
        Analisis visual ini diproses secara otomatis oleh model komputasi cerdas (Machine Learning) untuk tujuan edukasi awal perawatan kulit dasar. 
        Saran yang tercantum tidak menggantikan diagnosis klinis ataupun resep dokter. Jika Anda memiliki riwayat alergi parah, iritasi aktif, 
        atau kondisi kulit medis tertentu, sangat disarankan untuk melakukan konsultasi langsung dengan dokter spesialis dermatologi.
    """)
    
    st.markdown(
        "<p style='text-align: center; color: #ADB5BD; font-size: 0.8rem; margin-top: 30px;'>"
        "Informatics Project cc</p>", 
        unsafe_allow_html=True
    )
