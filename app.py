import os
import urllib.request
import streamlit as st
import tensorflow as tf

# Link resolve dari Hugging Face kamu
MODEL_URL = "https://huggingface.co/spaces/hasnahumaira19/deteksi-jenis-kulit/resolve/main/skin_model.h5"
MODEL_PATH = "skin_model.h5"

# Fungsi pengecekan file
if not os.path.exists(MODEL_PATH):
    with st.spinner('Menghubungkan ke server Hugging Face untuk mengambil model...'):
        try:
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
            st.success("Model berhasil diunduh!")
        except Exception as e:
            st.error(f"Gagal mengambil model: {e}")

# Load model seperti biasa
model = tf.keras.models.load_model(MODEL_PATH)