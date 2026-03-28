#!/bin/bash

# Master Orchestrator: Karaman Takımı Uzay Havası Sistemi

echo "[MASTER ORCHESTRATOR] Sistem bileşenleri başlatılıyor..."

# 1. MLOps Sürekli Eğitim Ajanı (Arka Planda Çalışır)
# 3000 saniye (50 dakika) bekler, sonra modeli günceller ve döngüye devam eder.
(
  while true; do
    echo "[TRAINING AGENT] 50 dakikalık veri toplama döngüsü başladı..."
    sleep 3000
    echo "[TRAINING AGENT] SQLite veritabanından ağırlıklar güncelleniyor..."
    python3 train_pipeline.py
  done
) &
TRAINER_PID=$!

# 2. Ön Yüz / Görsel Komuta Merkezi (Ön Planda Çalışır)
echo "[FRONTEND] Streamlit arayüzü ayağa kaldırılıyor..."
python3 -m streamlit run app.py

# 3. Güvenli Kapatma Protokolü (Graceful Shutdown)
# Kullanıcı terminalde Ctrl+C yaptığında arkada çalışan eğitim döngüsünü de temizler (Zombi process bırakmaz).
trap "echo '[MASTER ORCHESTRATOR] Sistem kapatılıyor. Ajanlar durduruldu.'; kill $TRAINER_PID; exit" SIGINT SIGTERM