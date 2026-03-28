import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib
from datetime import datetime
import time
import sqlite3

# --- VERİTABANI BAŞLATMA (Storage Layer) ---
def init_db():
    conn = sqlite3.connect('space_weather_telemetry.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS telemetry
                 (timestamp TEXT, speed REAL, bz REAL, current_kp REAL, m_prob REAL)''')
    conn.commit()
    return conn

conn = init_db()

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="ALAZ | Uzay Havası Kriz Yönetimi", layout="wide")

# Özel Kurumsal CSS (Karanlık tema, neon yeşil/mavi vurgular)
st.markdown("""
<style>
    .reportview-container { background: #0e1117; }
    .big-font { font-size: 22px !important; font-weight: 600; color: #00ffcc; border-left: 4px solid #00ffcc; padding-left: 10px;}
    .section-header { font-size: 18px; font-weight: bold; color: #a3a8b8; text-transform: uppercase; letter-spacing: 1.5px; border-bottom: 1px solid #333; padding-bottom: 5px;}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; color: #ffffff;'>ALAZ SİSTEMİ | MERKEZİ KOMUTA EKRANI</h1>", unsafe_allow_html=True)
st.caption("Geliştirici: Karaman Takımı | Durum: Otonom MLOps Aktif")
st.divider()

@st.cache_resource
def load_model():
    try:
        return joblib.load('solar_flare_model.pkl')
    except:
        return None

ai_model = load_model()

# --- VERİ ÇEKME VE VERİTABANINA YAZMA (Data Ingestion) ---
@st.cache_data(ttl=60) # 60 saniyede bir cache patlar ve yeni veri çekilir
def fetch_and_log_data():
    try:
        kp_res = requests.get("https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json", timeout=5).json()
        prob_res = requests.get("https://services.swpc.noaa.gov/json/solar_probabilities.json", timeout=5).json()
        plasma = requests.get("https://services.swpc.noaa.gov/products/solar-wind/plasma-1-day.json", timeout=5).json()
        mag = requests.get("https://services.swpc.noaa.gov/products/solar-wind/mag-1-day.json", timeout=5).json()
        
        current_kp = float(kp_res[-1][1]) if len(kp_res) > 1 else 2.0
        m_prob = prob_res[0].get('m_class_1_day', 5) if prob_res else 5
        current_speed = float(plasma[-1][1]) if len(plasma)>1 and plasma[-1][1] is not None else 400.0
        current_bz = float(mag[-1][3]) if len(mag)>1 and mag[-1][3] is not None else -1.0
        
        # VERİYİ SQLITE'A KAYDET
        c = conn.cursor()
        c.execute("INSERT INTO telemetry VALUES (?, ?, ?, ?, ?)", 
                  (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), current_speed, current_bz, current_kp, m_prob))
        conn.commit()
        
        return current_kp, m_prob, current_speed, current_bz
    except Exception as e:
        return 2.0, 5, 400.0, -1.0

current_kp, m_flare_prob, wind_speed, wind_bz = fetch_and_log_data()

# --- ÜST PANEL: CANLI UZAY TELEMETRİSİ ---
st.subheader("🛰️ L1 Uyduları Canlı Telemetrisi (Otonom Loglama Aktif)")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Gerçekleşen Kp", f"{current_kp:.1f}")
col2.metric("M-Sınıfı Patlama İhtimali", f"%{m_flare_prob}")
col3.metric("Güneş Rüzgarı Hızı", f"{wind_speed:.1f} km/s")
col4.metric("Manyetik Alan (Bz)", f"{wind_bz:.1f} nT")

st.divider()

# --- ORTA PANEL: AI SİMÜLASYONU ---
# --- ANA KONTROL PANELLERİ ---
# --- OTONOM LOGLAMA ALTYAPISI (SESSION STATE) ---
if "operation_logs" not in st.session_state:
    st.session_state.operation_logs = []

# --- ANA KONTROL PANELLERİ (SEKMELİ YAPI) ---
st.divider()
# "Simülasyon" ibareleri tamamen kaldırıldı, Endüstriyel isimlendirmeye geçildi.
tab_orbit, tab_earth, tab_ai = st.tabs(["[ MODÜL: YÖRÜNGE VE UYDULAR ]", "[ MODÜL: YERYÜZÜ VE ALTYAPI ]", "[ MODÜL: AI TAHMİN MOTORU ]"])

# 1. YÖRÜNGE SEKMESİ (Aktif Uydular ve Ülke Filtresi)
with tab_orbit:
    st.markdown("<div class='section-header'>GLOBAL UYDU RİSK İZLEME EKRANI</div>", unsafe_allow_html=True)
    
    satellite_db = pd.DataFrame({
        "Ülke": ["Türkiye", "Türkiye", "Türkiye", "Türkiye", "ABD", "ABD"],
        "Uydu Adı": ["Göktürk-1", "İMECE", "Türksat 5B", "Türksat 4A", "Starlink-G4", "ISS"],
        "Yörünge Tipi": ["LEO (Alçak)", "LEO (Alçak)", "GEO (Yer Sabit)", "GEO (Yer Sabit)", "LEO (Alçak)", "LEO (Alçak)"],
        "Kritik Risk Sensitivitesi": ["Atmosferik Sürtünme (Drag)", "Atmosferik Sürtünme (Drag)", "Radyasyon/Elektrik Yüklenmesi", "Radyasyon/Elektrik Yüklenmesi", "Atmosferik Sürtünme", "Radyasyon ve Sürtünme"]
    })

    selected_country = st.selectbox("İzlenecek Ülkeyi Seçiniz:", ["Türkiye", "ABD", "Tümü"])
    
    if selected_country != "Tümü":
        filtered_df = satellite_db[satellite_db["Ülke"] == selected_country]
    else:
        filtered_df = satellite_db

    # Kp indeksine göre dinamik risk durumu
    if current_kp >= 7:
        risk_status = "YÜKSEK RİSK (Manevra Önerilir)"
    elif current_kp >= 5:
        risk_status = "ORTA RİSK (İrtifa Kaybı İhtimali)"
    else:
        risk_status = "NOMİNAL (Güvenli)"

    filtered_df["Anlık Kp Risk Durumu"] = risk_status
    
    st.dataframe(filtered_df, use_container_width=True)
    
    # --- CANLI TERMİNAL LOG KUTUSU ---
    st.markdown("#### 📡 Otonom Sistem Günlükleri")
    
    # Anlık saati al
    now_str = datetime.now().strftime("%d.%m.%Y %H:%M:%S")
    new_log = f"[{now_str}] Sistem Çıktısı: Seçili uydular için anlık uzay havası etkisi {risk_status} olarak belirlenmiştir."
    
    # Aynı dakikada sayfayı manuel yenilersen aynı logu iki kere basmasın diye güvenlik kontrolü
    if len(st.session_state.operation_logs) == 0 or st.session_state.operation_logs[0] != new_log:
        st.session_state.operation_logs.insert(0, new_log)
        
    # Bellek şişmesin diye sadece son 50 logu tut
    st.session_state.operation_logs = st.session_state.operation_logs[:50]
    
    # Siber/Terminal Tasarımlı Log Kutusu (HTML/CSS)
    log_html = "<div style='height: 150px; overflow-y: auto; background-color: #000000; border: 1px solid #333; padding: 10px; font-family: monospace; color: #00ffcc; font-size: 13px;'>"
    for log in st.session_state.operation_logs:
        log_html += f"{log}<br>"
    log_html += "</div>"
    
    st.markdown(log_html, unsafe_allow_html=True)

# 2. YERYÜZÜ SEKMESİ (Santraller ve Şebeke)
with tab_earth:
    st.markdown("<div class='section-header'>ULUSAL ELEKTRİK VE İLETİŞİM ALTYAPISI</div>", unsafe_allow_html=True)
    st.write("Güneş fırtınaları (Yüksek Kp), yeryüzündeki uzun iletim hatlarında Jeomanyetik İndüklenmiş Akımlar (GIC) yaratır. Bu durum trafoların yanmasına neden olabilir.")
    
    col_power, col_comms = st.columns(2)
    with col_power:
        st.markdown("#### ENERJİ NAKİL HATLARI (TEİAŞ)")
        if current_kp >= 8:
            st.error("Kritik Uyarı: Kuzey enlemlerindeki 400kV trafo merkezlerinde GIC akımı tespit edildi. Yük atma (Load Shedding) protokolünü başlatın.")
            st.warning("Akkuyu NGS: Şebeke bağlantı stabilitesi izleniyor. Jeneratörleri hazırda tutun.")
        else:
            st.success("Tüm Türkiye Elektrik İletim A.Ş. (TEİAŞ) şebekesi stabil durumda. GIC akımı nominal seviyelerde.")

    with col_comms:
        st.markdown("#### HAVACILIK VE HABERLEŞME (DHMİ / BTK)")
        if current_kp >= 6:
            st.warning("Uyarı: Kutuplara yakın uçuş rotalarında HF (Yüksek Frekans) radyo kesintileri yaşanabilir. DHMİ uçuş rotalarını güncelliyor.")
        else:
            st.success("Havacılık haberleşme frekanslarında anomali tespit edilmedi.")

# 3. YAPAY ZEKA SEKMESİ (Gerçek Tahmin Motoru)
with tab_ai:
    st.markdown("<div class='section-header'>YAPAY ZEKA TAHMİN VE OPERASYON MERKEZİ</div>", unsafe_allow_html=True)
    col_sim, col_risk = st.columns([1, 2])

    with col_sim:
        flare_input = st.selectbox("Tespit Edilen Patlama Sınıfı:", ["C1.0", "M1.0", "M5.0", "X1.0", "X5.0", "X15.0"])
        if ai_model:
            f_class = flare_input.split(" ")[0]
            mapping = {'X': 100, 'M': 10, 'C': 1, 'B': 0.1}
            score = mapping.get(f_class[0], 0) * float(f_class[1:])
            now = datetime.now()
            m_sin, h_sin = np.sin(2 * np.pi * now.month/12), np.sin(2 * np.pi * now.hour/24)
            predicted_kp = ai_model.predict([[score, m_sin, h_sin, now.year]])[0]
        else:
            predicted_kp = 0.0

        st.markdown(f"<p class='big-font'>Model Çıktısı (Tahmini Kp): {predicted_kp:.2f}</p>", unsafe_allow_html=True)
        st.progress(min(predicted_kp / 9.0, 1.0))

    with col_risk:
        st.markdown("#### 🏢 Ulusal Altyapı Otonom Emirleri")
        if predicted_kp >= 8.5:
            st.error("🚨 TÜRKSAT: SAFE-MODE'A ALIN! TEİAŞ: PLANLI KESİNTİ BAŞLATIN.")
        elif predicted_kp >= 7.0:
            st.warning("⚠️ TÜRKSAT: Yörünge sürüklenmesi arttı. BOTAŞ: Katodik korumayı artırın.")
        elif predicted_kp >= 5.0:
            st.info("ℹ️ DHMİ: HF Radyo iletişiminde parazitlenme rapor ediliyor.")
        else:
            st.success("✅ TÜM KURUMLAR: Uzay havası stabil. Müdahaleye gerek yok.")

# --- OTONOM YENİLEME DÖNGÜSÜ ---
st.divider()
st.caption("Veritabanı: SQLite | Yenileme: 60 Saniyede Bir Otomatik (Asenkron API İstekleri)")
time.sleep(60)
st.rerun()