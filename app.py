import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys, os, io

sys.path.append(os.path.join(os.path.dirname(__file__), "Src"))

from veri_isleme  import (veri_yukle, veri_temizle, ozellik_hazirla, veri_ozeti,
                           KANALLAR, KANAL_ETIKETLER, as_parametreleri_cikart)
from mmm_model    import mmm_regresyon, kanal_katkisi, gercek_vs_tahmin, mmm_gelismis, kanal_katkisi_as
from optimizasyon import (butce_optimizasyonu, optimizasyon_tablosu,
                           genetik_optimizasyon, genetik_optimizasyon_as, _tekli_tahmin_as)
from duyarlilik   import duyarlilik_analizi, butce_duyarliligi

st.set_page_config(
    page_title="Dijital Reklam Bütçesi Optimizasyonu",
    page_icon="💄",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main { background-color: #0f0f1a; }
    .block-container { padding: 2rem 2rem 2rem 2rem; }
    h1 { color: #e91e8c; font-size: 2rem !important; }
    h2 { color: #c2185b; }
    h3 { color: #f06292; }
    .stMetric { background: #1a1a2e; border-radius: 10px; padding: 0.5rem; }
    div[data-testid="stSidebarContent"] { background-color: #1a1a2e; }

    /* ── Dosya yükleyici iç metinlerini Türkçeleştirme ───────────────────── */
    [data-testid="stFileUploaderDropzoneInstructions"] > div > span:first-child {
        font-size: 0 !important; line-height: 0;
    }
    [data-testid="stFileUploaderDropzoneInstructions"] > div > span:first-child::before {
        content: "Dosyayı buraya sürükleyin";
        font-size: 0.875rem; font-weight: 400; line-height: 1.5;
    }
    [data-testid="stFileUploaderDropzoneInstructions"] > div > small {
        font-size: 0 !important;
    }
    [data-testid="stFileUploaderDropzoneInstructions"] > div > small::before {
        content: "Desteklenen format: .xlsx"; font-size: 0.75rem;
    }
</style>
""", unsafe_allow_html=True)


# ── Session state ──────────────────────────────────────────────────────────
if "kullanici_veri_aktif" not in st.session_state:
    st.session_state.kullanici_veri_aktif = False
if "kullanici_paket" not in st.session_state:
    st.session_state.kullanici_paket = None


# ── Sabit kanal renk paleti (tüm grafiklerde tutarlı) ─────────────────────
_RENK_SIRASI = ["#4285F4", "#AB47BC", "#FF5252", "#FF7043", "#E91E8C", "#26C6DA"]
KANAL_RENK_MAP = {KANAL_ETIKETLER[k]: c for k, c in zip(KANALLAR, _RENK_SIRASI)}


# ══════════════════════════════════════════════════════════════════════════
# ÖNBELLEK: VARSAYİLAN MODEL
# ══════════════════════════════════════════════════════════════════════════
@st.cache_resource
def model_yukle():
    df   = veri_temizle(veri_yukle())
    X, y = ozellik_hazirla(df)
    model, scaler, y_pred, metrikler = mmm_regresyon(X, y)
    return df, X, y, model, scaler, y_pred, metrikler


# ══════════════════════════════════════════════════════════════════════════
# ÖNBELLEK: EXCEL ŞABLONU
# ══════════════════════════════════════════════════════════════════════════
@st.cache_data
def _sablon_olustur() -> bytes | None:
    try:
        sutunlar = (["Ay", "Ciro_TL"] + KANALLAR +
                    ["Kampanya_Donemi", "Fiyat_Endeksi", "Mevsimsellik_Indeksi"])
        df_sablon = pd.DataFrame(columns=sutunlar)
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            pd.DataFrame([["MMM Karar Destek Sistemi – Veri Şablonu"]]).to_excel(
                writer, sheet_name="Veri", index=False, header=False, startrow=0)
            pd.DataFrame([[
                "Ay: YYYY-MM | Ciro ve harcamalar TL | "
                "Kampanya_Donemi: 0/1 | Fiyat_Endeksi, Mevsimsellik_Indeksi: pozitif reel sayı"
            ]]).to_excel(writer, sheet_name="Veri", index=False, header=False, startrow=1)
            df_sablon.to_excel(writer, sheet_name="Veri", index=False, startrow=2)
        return output.getvalue()
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════
# KULLANICI VERİSİ DOĞRULAMA & MODEL EĞİTİMİ
# ══════════════════════════════════════════════════════════════════════════
def kullanici_verisi_isle(yuklenen_dosya):
    try:
        df_kull = pd.read_excel(yuklened_dosya, sheet_name="Veri", header=2)
        gerekli = (["Ay", "Ciro_TL"] + KANALLAR +
                   ["Kampanya_Donemi", "Fiyat_Endeksi", "Mevsimsellik_Indeksi"])
        eksik = [s for s in gerekli if s not in df_kull.columns]
        if eksik:
            return (None,) * 7 + (f"Eksik sütunlar: {', '.join(eksik)}",)
        if len(df_kull) < 12:
            return (None,) * 7 + ("En az 12 aylık veri gereklidir.",)
        if df_kull[gerekli].isnull().any().any():
            return (None,) * 7 + ("Veri setinde eksik (boş) değerler var.",)
        df_kull = veri_temizle(df_kull)
        X_kull, y_kull = ozellik_hazirla(df_kull)
        m_k, sc_k, yp_k, mt_k = mmm_regresyon(X_kull, y_kull)
        return df_kull, X_kull, y_kull, m_k, sc_k, yp_k, mt_k, None
    except Exception as e:
        return (None,) * 7 + (str(e),)


# ══════════════════════════════════════════════════════════════════════════
# YARDIMCI: TAM MODEL TAHMİNİ (intercept dahil)
# ══════════════════════════════════════════════════════════════════════════
def _tam_ciro_hesapla(sonuc, model, scaler, mm_scaler, as_params, df):
    optimal_vek = np.array([sonuc["optimal_butceler"][k] for k in KANALLAR])
    if as_params is not None and mm_scaler is not None:
        if "Predict Tabanlı" in sonuc.get("yontem", ""):
            return float(sonuc["tahmini_satis"])
        try:
            return float(_tekli_tahmin_as(optimal_vek, model, scaler, mm_scaler, as_params))
        except Exception:
            return float(sonuc["tahmini_satis"])
    else:
        kontrol_ort = np.array([
            df["Kampanya_Donemi"].mean(),
            df["Fiyat_Endeksi"].mean(),
            df["Mevsimsellik_Indeksi"].mean(),
        ])
        x_row = np.concatenate([optimal_vek, kontrol_ort]).reshape(1, -1)
        try:
            return float(model.predict(scaler.transform(x_row))[0])
        except Exception:
            return float(sonuc["tahmini_satis"])


# ══════════════════════════════════════════════════════════════════════════
# YARDIMCI: CV R² MESAJ FONKSİYONLARI
# ══════════════════════════════════════════════════════════════════════════
def _cv_kisa(cv_r2: float) -> str:
    if cv_r2 < 0:
        return "⚠️ CV R² düşük"
    if cv_r2 < 0.20:
        return f"CV R² = {cv_r2:.4f} (düşük)"
    if cv_r2 < 0.50:
        return f"CV R² = {cv_r2:.4f} (orta)"
    return f"CV R² = {cv_r2:.4f} (iyi)"


def _cv_detay(cv_r2: float) -> str:
    if cv_r2 < 0:
        return (
            "CV performans metrikleri düşük seyredebilir. "
            "Metrikler birlikte yorumlanmalı; sistem karar destek amacıyla kullanılmalıdır."
        )
    return f"CV R² = {cv_r2:.4f} — zaman serisi çapraz doğrulaması (dönemler arası tutarlılık)."


def _format_degisim_pct(pct, yumusatma_aktif, max_pct, tolerans=1.5) -> str:
    s = f"{pct:+.1f}%"
    if yumusatma_aktif and abs(pct) >= max_pct - tolerans:
        s += " (Limit)"
    return s


# ══════════════════════════════════════════════════════════════════════════
# AKTİF VERİ / MODEL
# ══════════════════════════════════════════════════════════════════════════
_df_v, _X_v, _y_v, _model_v, _scaler_v, _y_pred_v, _metrikler_v = model_yukle()

if st.session_state.kullanici_veri_aktif and st.session_state.kullanici_paket:
    df, X, y, model, scaler, y_pred, metrikler = st.session_state.kullanici_paket
else:
    df, X, y, model, scaler, y_pred, metrikler = (
        _df_v, _X_v, _y_v, _model_v, _scaler_v, _y_pred_v, _metrikler_v
    )

ozet     = veri_ozeti(df)
kanal_ort = {k: float(df[k].mean()) for k in KANALLAR}

VARSAYILAN_DECAY = {
    "Google_Ads_Harcama_TL" : 0.4,
    "Meta_Harcama_TL"       : 0.5,
    "YouTube_Harcama_TL"    : 0.7,
    "Pazaryeri_Harcama_TL"  : 0.3,
    "Influencer_Harcama_TL" : 0.5,
    "Twitter_X_Harcama_TL"  : 0.3,
}


# ══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/lipstick.png", width=60)
    st.title("⚙️ Kontrol Paneli")
    st.markdown("---")

    # ── Veri Yükleme (Opsiyonel) ─────────────────────────────────────────
    with st.expander("📊 Kendi Verinizi Yükleyin (Opsiyonel)", expanded=False):
        st.caption("Varsayılan demo verisi aktif. Kendi Excel verinizi yükleyebilirsiniz.")

        sablon_bytes = _sablon_olustur()
        if sablon_bytes:
            st.download_button(
                "📥 Excel Şablonu İndir",
                data=sablon_bytes,
                file_name="mmm_veri_sablonu.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )

        with st.expander("📋 Dosya formatı hakkında", expanded=False):
            st.caption(
                "**Sayfa adı:** Veri | **Başlık:** 3. satırda\n\n"
                "**Zorunlu sütunlar:** Ay (YYYY-MM), Ciro_TL, 6 kanal harcaması, "
                "Kampanya_Donemi (0/1), Fiyat_Endeksi, Mevsimsellik_Indeksi\n\n"
                "**Minimum:** 12 dönem | Eksik değer olmamalı"
            )

        st.caption("📎 Dosyayı buraya sürükleyin veya seçin:")
        yuklened_dosya = st.file_uploader(
            "Excel dosyası (.xlsx)",
            type=["xlsx"],
            label_visibility="collapsed",
            key="dosya_yukle",
        )

        if yuklened_dosya is not None:
            with st.spinner("Veri doğrulanıyor ve model hazırlanıyor..."):
                sonuc_paket = kullanici_verisi_isle(yuklened_dosya)
            hata = sonuc_paket[-1]
            if hata is None:
                st.session_state.kullanici_veri_aktif = True
                st.session_state.kullanici_paket = sonuc_paket[:-1]
                st.success("✅ Veri başarıyla doğrulandı.")
            else:
                st.error(f"⚠️ {hata}")
                st.session_state.kullanici_veri_aktif = False
                st.session_state.kullanici_paket = None

        if st.session_state.kullanici_veri_aktif:
            if st.button("🔄 Demo Veriye Dön", use_container_width=True):
                st.session_state.kullanici_veri_aktif = False
                st.session_state.kullanici_paket = None
                st.rerun()
        else:
            st.caption("Şu an demo verisi aktif.")

    st.markdown("---")
    st.subheader("💰 Bütçe Ayarları")
    toplam_butce = st.number_input(
        "Toplam Aylık Bütçe (TL)",
        min_value=100_000, max_value=2_000_000,
        value=500_000, step=50_000, format="%d",
    )

    st.markdown("---")
    st.subheader("🧬 Model Seçimi")
    model_secimi = st.radio(
        "Optimizasyon Yöntemi:",
        ["Doğrusal Programlama", "Genetik Algoritma"],
        index=0,
    )
    kullan_gelismis = st.toggle("Adstock + Saturation Kullan", value=False)

    if kullan_gelismis:
        st.markdown("---")
        st.subheader("📡 Adstock Parametreleri")
        st.caption("Her kanal için gecikmeli etki oranı (0 = anlık, 1 = uzun vadeli)")
        decay_rates = {}
        for k in KANALLAR:
            decay_rates[k] = st.slider(
                KANAL_ETIKETLER[k],
                min_value=0.0, max_value=0.9,
                value=VARSAYILAN_DECAY.get(k, 0.5), step=0.1,
                key=f"decay_{k}"
            )

        st.markdown("---")
        st.subheader("📈 Saturation Parametresi")
        st.caption("Yüksek β: keskin doyum eğrisi — Düşük β: daha kademeli")
        beta_sat = st.slider("Beta (Eğim)", min_value=0.5, max_value=5.0, value=3.0, step=0.5)
    else:
        decay_rates = {k: VARSAYILAN_DECAY.get(k, 0.5) for k in KANALLAR}
        beta_sat = 3.0

    st.markdown("---")
    st.subheader("📊 Kanal Bütçe Kısıtları")
    st.caption("Her kanal için izin verilen minimum ve maksimum aylık harcama aralığı")
    min_b, max_b = {}, {}
    for k in KANALLAR:
        ad = KANAL_ETIKETLER[k]
        default_min = max(0, int(kanal_ort[k] * 0.5))
        default_max = min(int(toplam_butce), int(kanal_ort[k] * 1.5))
        if default_max <= default_min:
            default_max = min(int(toplam_butce), default_min + 10_000)

        st.caption(f"**{ad}**")
        c1, c2 = st.columns(2)
        with c1:
            min_b[k] = st.number_input(
                "Min", min_value=0, max_value=int(toplam_butce),
                value=default_min, step=5000, key=f"min_{k}",
                label_visibility="visible"
            )
        with c2:
            max_b[k] = st.number_input(
                "Max", min_value=0, max_value=int(toplam_butce),
                value=default_max, step=5000, key=f"max_{k}",
                label_visibility="visible"
            )

    # ── KISIT DOĞRULAMA ─────────────────────────────────────────────────
    kisit_hatalar = [k for k in KANALLAR if min_b[k] >= max_b[k]]
    sum_min = sum(min_b[k] for k in KANALLAR)
    sum_max = sum(max_b[k] for k in KANALLAR)
    fizibilite_hatalar = []
    if sum_min > toplam_butce:
        fizibilite_hatalar.append(
            f"Min toplamı ({sum_min:,.0f} TL) > Toplam bütçe ({toplam_butce:,.0f} TL)."
        )
    if sum_max < toplam_butce:
        fizibilite_hatalar.append(
            f"Max toplamı ({sum_max:,.0f} TL) < Toplam bütçe ({toplam_butce:,.0f} TL)."
        )

    if kisit_hatalar:
        hatali = ", ".join([KANAL_ETIKETLER[k] for k in kisit_hatalar])
        st.error(f"⚠️ **Min ≥ Max:** {hatali} — Lütfen düzeltin.")
    for msg in fizibilite_hatalar:
        st.error(f"⚠️ **Fizibilite:** {msg}")

    has_error = bool(kisit_hatalar) or bool(fizibilite_hatalar)

    st.markdown("---")
    st.subheader("🔧 Bütçe Değişim Sınırı")
    st.caption("Her kanalın tarihsel ortalamadan maksimum sapma oranı")
    yumusatma_aktif = st.toggle("Değişim Sınırı Uygula", value=False)
    if yumusatma_aktif:
        max_degisim_pct = st.slider(
            "Maksimum Değişim (%)", min_value=10, max_value=100, value=40, step=5
        )
        max_degisim = max_degisim_pct / 100.0
        st.caption(f"Her kanal tarihsel ortalamasından en fazla ±{max_degisim_pct}% sapabilir.")
    else:
        max_degisim = None

    st.markdown("---")
    optimize_btn = st.button(
        "🚀 Optimizasyonu Çalıştır",
        use_container_width=True,
        disabled=has_error,
    )
    if has_error:
        st.caption("⛔ Kısıt hataları giderilmeden optimizasyon çalışmaz.")


# ── Ana Başlık ─────────────────────────────────────────────────────────────
st.title("💄 Dijital Reklam Bütçesi Optimizasyon Sistemi")
st.caption("Media Mix Modeling + Doğrusal Programlama & Genetik Algoritma | Karar Destek Prototipi")
st.markdown("---")

# ── Model hazırlama ────────────────────────────────────────────────────────
if kullan_gelismis:
    with st.spinner("Adstock + Saturation modeli hazırlanıyor..."):
        aktif_model, aktif_scaler, aktif_y_pred, aktif_metrikler, df_as, X_as, aktif_mm_scaler = mmm_gelismis(
            df, decay_rates=decay_rates, beta=beta_sat
        )
        aktif_X         = X_as
        aktif_katki     = kanal_katkisi_as(aktif_model, aktif_scaler, X_as)
        aktif_as_params = as_parametreleri_cikart(df, decay_rates=decay_rates, beta=beta_sat)
else:
    aktif_model      = model
    aktif_scaler     = scaler
    aktif_y_pred     = y_pred
    aktif_metrikler  = metrikler
    aktif_X          = X
    aktif_katki      = kanal_katkisi(model, scaler, X)
    aktif_mm_scaler  = None
    aktif_as_params  = None

# ── Tablar ─────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Veri & Model",
    "🤖 Model Karşılaştırma",
    "🎯 Optimizasyon",
    "🔍 Duyarlılık Analizi",
    "📈 Gerçek vs Tahmin"
])


# ══════════════════════════════════════════════════════════════════════════
# TAB 1 — VERİ & MODEL
# ══════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("📦 Veri Özeti")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Dönem Sayısı",    f"{ozet['gozlem_sayisi']} ay")
    c2.metric("Toplam Ciro",     f"{ozet['toplam_ciro']/1e6:.2f} M TL")
    c3.metric("Toplam Harcama",  f"{ozet['toplam_harcama']/1e6:.2f} M TL")
    c4.metric("Ort. ROAS",       f"{ozet['ortalama_roas']:.2f}")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("En Yüksek Ciro",  ozet["max_ciro_ay"])
    c6.metric("En Düşük Ciro",   ozet["min_ciro_ay"])
    c7.metric("Kampanyalı Dönem", f"{ozet['kampanya_sayisi']} ay")
    c8.metric("Stoksuz Dönem",    f"{ozet['stok_yok_sayisi']} ay")

    st.markdown("---")
    st.subheader("🤖 Aktif Model Performansı")

    if kullan_gelismis:
        st.info("🧠 Adstock + Saturation modeli aktif (Ridge Regresyon, α=50)")
    else:
        st.info("📐 Temel MMM modeli aktif (OLS Doğrusal Regresyon)")

    cv_r2_ort = aktif_metrikler["CV_R2_Ort"]

    mc1, mc2, mc3, mc4, mc5 = st.columns(5)
    mc1.metric(
        "R²", aktif_metrikler["R2"],
        help="Modelin eğitim verisine uyum oranı. Yüksek eğitim R²'si tek başına güçlü genelleme anlamına gelmez; CV R² ile birlikte değerlendirin."
    )
    mc2.metric("RMSE (TL)", f"{aktif_metrikler['RMSE']:,.0f}")
    mc3.metric("MAPE (%)",  aktif_metrikler["MAPE"])
    mc4.metric(
        "CV R² Ort.", aktif_metrikler["CV_R2_Ort"],
        help=_cv_detay(cv_r2_ort)
    )
    mc5.metric(
        "CV R² Std.", aktif_metrikler["CV_R2_Std"],
        help="CV foldları arasındaki R² standart sapması. Düşük değer dönemler arası tutarlılığa işaret eder."
    )

    st.caption(
        f"CV Yöntemi: {aktif_metrikler.get('CV_Yontem', 'ZamanSerisi-3')} "
        "— zaman serisi yapısına uygun çapraz doğrulama; dönemler arası model tutarlılığını ölçer"
    )

    # Performans notu — yalnızca CV R² düşük olduğunda gösterilir
    if cv_r2_ort < 0:
        with st.expander("ℹ️ Performans Notu", expanded=False):
            st.caption(
                "CV performans metrikleri düşük seyredebilir; bu durum veri yapısı ve "
                "model karmaşıklığına bağlıdır. Metrikler birlikte yorumlanmalıdır. "
                "Sistem senaryo analizi ve karar destek amacıyla kullanılmalıdır."
            )

    # Model yorumlama — expander'da
    with st.expander("📘 Model Yorumlama Notları", expanded=False):
        if kullan_gelismis:
            st.caption(
                "**Adstock + Saturation + Ridge:** Gecikmeli reklam etkileri (adstock) ve azalan marjinal "
                "getiri (saturation) modele dahil edilir. Ridge düzenlileştirme fazla uyumu azaltmak için "
                "katsayıları sıkıştırır; eğitim R²'si OLS'den düşük çıkabilir, bu beklenen bir davranıştır."
            )
        else:
            st.caption(
                "**Temel MMM (OLS):** Standart çoklu regresyon ile kanal etki katsayılarını tahmin eder. "
                "Yüksek eğitim R²'si tek başına güçlü genelleme anlamına gelmez; CV R² ile birlikte "
                "yorumlanması önerilir."
            )

    st.markdown("---")
    st.subheader("📈 Kanal Bazlı Satış Etki Payları (%)")
    fig_katki = px.bar(
        aktif_katki, x="Kanal_Adi", y="Etki_Payi",
        color="Kanal_Adi",
        color_discrete_map=KANAL_RENK_MAP,
        title="Kanal Bazlı Satış Etki Payları (%)",
        text="Etki_Payi",
        labels={"Kanal_Adi": "Kanal", "Etki_Payi": "Etki Payı (%)"},
    )
    fig_katki.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig_katki.update_layout(
        plot_bgcolor="#1a1a2e", paper_bgcolor="#1a1a2e",
        font_color="white", showlegend=False, title_font_color="#e91e8c"
    )
    st.plotly_chart(fig_katki, use_container_width=True)

    st.subheader("📅 Ciro Zaman Serisi")
    fig_ts = px.line(
        df, x="Ay", y="Ciro_TL", markers=True,
        title="Aylık Satış Cirosu (TL)",
        color_discrete_sequence=["#e91e8c"],
        labels={"Ciro_TL": "Ciro (TL)", "Ay": "Ay"},
    )
    fig_ts.update_layout(
        plot_bgcolor="#1a1a2e", paper_bgcolor="#1a1a2e",
        font_color="white", title_font_color="#e91e8c"
    )
    st.plotly_chart(fig_ts, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════
# TAB 2 — MODEL KARŞILAŞTIRMA
# ══════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("🤖 Temel MMM vs Adstock+Saturation Karşılaştırması")

    with st.expander("📘 Performans Metrikleri Kılavuzu", expanded=False):
        st.caption(
            "Bu bölüm iki modelleme yaklaşımını temel performans göstergeleri üzerinden karşılaştırır. "
            "Uyum, hata ve dönemler arası tutarlılık metrikleri birlikte yorumlanmalıdır. "
            "Ridge düzenlileştirme ve dönüşüm yapıları eğitim R²'sini etkileyebilir; bu beklenen bir davranıştır. "
            "Karşılaştırma sonuçları karar destek ve senaryo analizi amacıyla kullanılmalıdır."
        )

    with st.spinner("Modeller karşılaştırılıyor..."):
        m_temel, s_temel, yp_temel, mt_temel = mmm_regresyon(X, y)
        m_gelismis, s_gelismis, yp_gelismis, mt_gelismis, _, X_gelismis, _ = mmm_gelismis(
            df, decay_rates=decay_rates, beta=beta_sat
        )

    kiyaslama = pd.DataFrame({
        "Metrik"           : ["R²", "RMSE (TL)", "MAPE (%)", "CV R² Ort.", "CV R² Std.", "Düzenlileştirme", "CV Yöntemi"],
        "Temel MMM"        : [
            mt_temel["R2"], f"{mt_temel['RMSE']:,.0f}", mt_temel["MAPE"],
            mt_temel["CV_R2_Ort"], mt_temel["CV_R2_Std"],
            mt_temel.get("Regularizasyon", "OLS"),
            mt_temel.get("CV_Yontem", "ZamanSerisi-3"),
        ],
        "Adstock+Saturation": [
            mt_gelismis["R2"], f"{mt_gelismis['RMSE']:,.0f}", mt_gelismis["MAPE"],
            mt_gelismis["CV_R2_Ort"], mt_gelismis["CV_R2_Std"],
            mt_gelismis.get("Regularizasyon", "Ridge(α=50.0)"),
            mt_gelismis.get("CV_Yontem", "ZamanSerisi-3"),
        ],
    })
    st.dataframe(kiyaslama, use_container_width=True)

    st.markdown("---")
    k_temel    = kanal_katkisi(m_temel, s_temel, X)
    k_gelismis = kanal_katkisi_as(m_gelismis, s_gelismis, X_gelismis)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Temel MMM — Kanal Etki Payları")
        fig1 = px.pie(
            k_temel, values="Etki_Payi", names="Kanal_Adi",
            color="Kanal_Adi", color_discrete_map=KANAL_RENK_MAP,
            labels={"Etki_Payi": "Etki Payı (%)", "Kanal_Adi": "Kanal"},
        )
        fig1.update_layout(paper_bgcolor="#1a1a2e", font_color="white")
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.markdown("#### Adstock+Saturation — Kanal Etki Payları")
        fig2 = px.pie(
            k_gelismis, values="Etki_Payi", names="Kanal_Adi",
            color="Kanal_Adi", color_discrete_map=KANAL_RENK_MAP,
            labels={"Etki_Payi": "Etki Payı (%)", "Kanal_Adi": "Kanal"},
        )
        fig2.update_layout(paper_bgcolor="#1a1a2e", font_color="white")
        st.plotly_chart(fig2, use_container_width=True)

    st.caption(
        "📊 Modeller arasındaki kanal payı farkları, gecikmeli etki dönüşümü (adstock), "
        "varsa saturation etkisi ve düzenlileştirme yaklaşımındaki farklardan kaynaklanabilir."
    )


# ══════════════════════════════════════════════════════════════════════════
# TAB 3 — OPTİMİZASYON
# ══════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("🎯 Bütçe Optimizasyon Sonuçları")

    if has_error:
        st.error("⛔ Sidebar'daki kısıt hataları giderilmeden optimizasyon çalışmaz.")
    else:
        if model_secimi == "Doğrusal Programlama":
            st.caption("ℹ️ LP sonucu sınır çözümüne eğilim gösterebilir.")
            with st.expander("LP Yöntemi Detayı", expanded=False):
                st.caption(
                    "Doğrusal Programlama, doğrusal amaç fonksiyonu kullandığından bütçeyi "
                    "kanal sınırlarına yaslandırma eğilimindedir. Bu matematiksel olarak normaldir. "
                    "Daha dengeli dağılım için Genetik Algoritma + AS modunu deneyin."
                )

        try:
            if model_secimi == "Genetik Algoritma" and kullan_gelismis:
                with st.spinner("🧬 GA + Adstock + Saturation çalışıyor..."):
                    sonuc = genetik_optimizasyon_as(
                        aktif_model, aktif_scaler, aktif_X,
                        toplam_butce=toplam_butce,
                        as_params=aktif_as_params,
                        mm_scaler=aktif_mm_scaler,
                        min_butceler=min_b, max_butceler=max_b,
                        df_orijinal=df,
                        max_degisim_orani=max_degisim,
                    )
            elif model_secimi == "Genetik Algoritma":
                with st.spinner("🧬 Genetik Algoritma çalışıyor..."):
                    sonuc = genetik_optimizasyon(
                        aktif_model, aktif_scaler, aktif_X,
                        toplam_butce=toplam_butce,
                        min_butceler=min_b, max_butceler=max_b,
                        df_orijinal=df,
                        max_degisim_orani=max_degisim,
                    )
            else:
                sonuc = butce_optimizasyonu(
                    aktif_model, aktif_scaler, aktif_X,
                    toplam_butce=toplam_butce,
                    min_butceler=min_b, max_butceler=max_b,
                    df_orijinal=df,
                    max_degisim_orani=max_degisim,
                )

            tahmini_ciro = _tam_ciro_hesapla(
                sonuc, aktif_model, aktif_scaler, aktif_mm_scaler, aktif_as_params, df
            )
            tablo = optimizasyon_tablosu(sonuc)

            # Kısa yöntem etiketi (kelime kesimi önlenir)
            if kullan_gelismis:
                yontem_goster = "GA + AS" if model_secimi == "Genetik Algoritma" else "LP + AS"
            else:
                yontem_goster = "GA" if model_secimi == "Genetik Algoritma" else "LP"

            oc1, oc2, oc3, oc4 = st.columns(4)
            oc1.metric("Toplam Bütçe", f"{toplam_butce:,.0f} TL")
            oc2.metric(
                "Tahmini Ciro", f"{tahmini_ciro:,.0f} TL",
                help="Seçili bütçe dağılımı için modelin tahmini toplam ciro çıktısı. Reklam dışı etkiler ve temel satış seviyesi de bu tahmine dahildir."
            )
            gelir_butce_orani = tahmini_ciro / toplam_butce if toplam_butce > 0 else 0
            oc3.metric(
                "Gelir/Bütçe Oranı", f"{gelir_butce_orani:.2f}",
                help="Tahmini Ciro / Toplam Bütçe. Klasik ROAS değildir; model tahmini baz alındığından karar destek amacıyla yorumlanmalıdır."
            )
            oc4.metric("Yöntem", yontem_goster)

            st.caption("📌 Bütçe değişimi toplam ciroda marjinal etki yaratabilir.")
            with st.expander("Bütçe–Ciro İlişkisi Detayı", expanded=False):
                st.caption(
                    "Model, toplam satışın yalnızca bir kısmını reklam harcamalarıyla "
                    "ilişkilendirmektedir. Temel satış seviyesi ve reklam dışı etkiler "
                    "de modelde yer aldığından, bütçe değişimleri toplam ciroda daha "
                    "sınırlı / marjinal değişimler oluşturabilir."
                )

            if max_degisim is not None:
                st.caption(
                    f"🔧 Bütçe değişim sınırı aktif (±{int(max_degisim * 100)}%). "
                    f"Sınıra ulaşan kanallar tabloda '(Limit)' etiketi ile gösterilir."
                )

            st.markdown("---")

            # Grafik için görüntü kopyası (sütun isimleri insan dostu)
            fig_opt = go.Figure()
            fig_opt.add_trace(go.Bar(
                name="Mevcut Bütçe",
                x=tablo["Kanal"], y=tablo["Mevcut_Butce_TL"],
                marker_color="#555577"
            ))
            fig_opt.add_trace(go.Bar(
                name="Önerilen Bütçe",
                x=tablo["Kanal"], y=tablo["Optimal_Butce_TL"],
                marker_color="#e91e8c"
            ))
            fig_opt.update_layout(
                barmode="group",
                title="Mevcut vs Önerilen Bütçe Dağılımı (TL)",
                xaxis_title="Kanal",
                yaxis_title="Bütçe (TL)",
                plot_bgcolor="#1a1a2e", paper_bgcolor="#1a1a2e",
                font_color="white", title_font_color="#e91e8c",
                legend=dict(bgcolor="#1a1a2e")
            )
            st.plotly_chart(fig_opt, use_container_width=True)

            fig_pie = px.pie(
                tablo, values="Optimal_Butce_TL", names="Kanal",
                title="Önerilen Bütçe Dağılımı",
                color="Kanal",
                color_discrete_map=KANAL_RENK_MAP,
                labels={"Optimal_Butce_TL": "Önerilen Bütçe (TL)", "Kanal": "Kanal"},
            )
            fig_pie.update_layout(paper_bgcolor="#1a1a2e", font_color="white",
                                  title_font_color="#e91e8c")
            st.plotly_chart(fig_pie, use_container_width=True)

            st.subheader("📋 Detaylı Tablo")

            with st.expander("ℹ️ Sütun Açıklamaları", expanded=False):
                st.caption(
                    "**Mevcut Bütçe:** Tarihsel aylık ortalama kanal harcaması.\n\n"
                    "**Önerilen Bütçe:** Model çıktısına göre optimize edilmiş aylık bütçe.\n\n"
                    "**Değişim (%):** Önerilen bütçenin tarihe göre % sapması. "
                    "Değişim sınırı aktifse (Limit) etiketi görünür.\n\n"
                    "**Tahmini Katkı:** Kanalın modeldeki göreli katkı payı. "
                    "Toplam tahmini ciro ile birebir aynı kavram değildir."
                )

            # Kullanıcı dostu sütun isimleriyle tablo
            tablo_d = tablo.rename(columns={
                "Mevcut_Butce_TL"  : "Mevcut Bütçe (TL)",
                "Optimal_Butce_TL" : "Önerilen Bütçe (TL)",
                "Degisim_Pct"      : "Değişim (%)",
                "Tahmini_Katki_TL" : "Tahmini Katkı (TL)",
            })

            if yumusatma_aktif:
                tablo_d["Değişim (%)"] = tablo["Degisim_Pct"].apply(
                    lambda p: _format_degisim_pct(p, yumusatma_aktif, max_degisim_pct)
                )
                st.dataframe(
                    tablo_d.style.format({
                        "Mevcut Bütçe (TL)"   : "{:,.0f}",
                        "Önerilen Bütçe (TL)" : "{:,.0f}",
                        "Tahmini Katkı (TL)"  : "{:,.0f}",
                    }),
                    use_container_width=True
                )
            else:
                st.dataframe(
                    tablo_d.style.format({
                        "Mevcut Bütçe (TL)"   : "{:,.0f}",
                        "Önerilen Bütçe (TL)" : "{:,.0f}",
                        "Değişim (%)"         : "{:+.1f}%",
                        "Tahmini Katkı (TL)"  : "{:,.0f}",
                    }),
                    use_container_width=True
                )

        except Exception as e:
            st.error(f"Optimizasyon hatası: {e}")


# ══════════════════════════════════════════════════════════════════════════
# TAB 4 — DUYARLILIK ANALİZİ
# ══════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("🔍 Duyarlılık Analizi")
    col_d1, col_d2 = st.columns(2)

    # ── Sol: Toplam Bütçe → Tahmini Ciro ─────────────────────────────────
    with col_d1:
        st.markdown("#### Toplam Bütçe → Tahmini Ciro")

        with st.expander("ℹ️ Nasıl hesaplanır?", expanded=False):
            st.caption(
                "Toplam bütçe, tarihsel kanal oranlarına göre dağıtılır. "
                "Her bütçe seviyesi için modelin tahmini ciro çıktısı hesaplanır. "
                "AS modunda saturation etkisi S-eğrisi oluşturur."
            )

        df_butce_duy = butce_duyarliligi(
            aktif_model, aktif_scaler, aktif_X,
            df_orijinal=df,
            as_params=aktif_as_params,
            mm_scaler=aktif_mm_scaler,
        )
        df_bd_d = df_butce_duy.rename(columns={
            "Toplam_Butce_TL"  : "Toplam Bütçe (TL)",
            "Tahmini_Satis_TL" : "Tahmini Ciro (TL)",
        })
        fig_bd = px.line(
            df_bd_d, x="Toplam Bütçe (TL)", y="Tahmini Ciro (TL)",
            markers=True, title="Toplam Bütçe Artışının Tahmini Ciroya Etkisi",
            color_discrete_sequence=["#e91e8c"]
        )
        fig_bd.update_layout(
            plot_bgcolor="#1a1a2e", paper_bgcolor="#1a1a2e",
            font_color="white", title_font_color="#e91e8c"
        )
        st.plotly_chart(fig_bd, use_container_width=True)

    # ── Sağ: Kanal Bütçesi → Tahmini Ciro ────────────────────────────────
    with col_d2:
        st.markdown("#### Kanal Bütçesi → Tahmini Ciro")

        with st.expander("ℹ️ Nasıl hesaplanır?", expanded=False):
            st.caption(
                "Seçilen kanalın bütçesi ±% oranında değiştirilir; "
                "diğer kanallar sabit tutulur. "
                "Her iki modda da doğrudan bütçe değişimi üzerinden model çıktısı hesaplanır."
            )

        secili_kanal = st.selectbox(
            "Kanal seçin:",
            options=KANALLAR, format_func=lambda x: KANAL_ETIKETLER[x]
        )

        if kullan_gelismis:
            degisim_araligi_kd = [-50, -30, -20, -10, 0, 10, 20, 30, 50, 100, 200]
        else:
            degisim_araligi_kd = [-30, -20, -10, 0, 10, 20, 30]

        df_katki_duy = duyarlilik_analizi(
            aktif_model, aktif_scaler, aktif_X, toplam_butce,
            df_orijinal=df,
            degisim_araligi=degisim_araligi_kd,
            as_params=aktif_as_params,
            mm_scaler=aktif_mm_scaler,
        )

        df_filtre = df_katki_duy[
            df_katki_duy["Degisen_Kanal"] == KANAL_ETIKETLER[secili_kanal]
        ][["Degisim_Pct", "Tahmini_Satis_TL"]].drop_duplicates()

        df_filtre_d = df_filtre.rename(columns={
            "Degisim_Pct"      : "Değişim (%)",
            "Tahmini_Satis_TL" : "Tahmini Ciro (TL)",
        })

        baslik_kd = f"{KANAL_ETIKETLER[secili_kanal]} — Bütçe Değişiminin Tahmini Ciroya Etkisi"

        if kullan_gelismis:
            fig_kd = px.line(
                df_filtre_d, x="Değişim (%)", y="Tahmini Ciro (TL)",
                markers=True, title=baslik_kd,
                color_discrete_sequence=["#7c4dff"]
            )
        else:
            fig_kd = px.bar(
                df_filtre_d, x="Değişim (%)", y="Tahmini Ciro (TL)",
                color="Tahmini Ciro (TL)", color_continuous_scale="RdPu",
                title=baslik_kd
            )

        fig_kd.update_layout(
            plot_bgcolor="#1a1a2e", paper_bgcolor="#1a1a2e",
            font_color="white", title_font_color="#e91e8c", showlegend=False
        )
        st.plotly_chart(fig_kd, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════
# TAB 5 — GERÇEK VS TAHMİN
# ══════════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("📈 Gerçek vs Tahmin Karşılaştırması")
    df_karsi = gercek_vs_tahmin(df, aktif_y_pred)

    fig_gvt = go.Figure()
    fig_gvt.add_trace(go.Scatter(
        x=df_karsi["Ay"], y=df_karsi["Gercek_Ciro"],
        name="Gerçek Ciro", mode="lines+markers",
        line=dict(color="#e91e8c", width=2), marker=dict(size=7)
    ))
    fig_gvt.add_trace(go.Scatter(
        x=df_karsi["Ay"], y=df_karsi["Tahmin_Ciro"],
        name="Tahmin Edilen Ciro", mode="lines+markers",
        line=dict(color="#7c4dff", width=2, dash="dash"), marker=dict(size=7)
    ))
    fig_gvt.update_layout(
        title="Gerçek vs Tahmin Edilen Ciro (TL)",
        xaxis_title="Dönem",
        yaxis_title="Ciro (TL)",
        plot_bgcolor="#1a1a2e", paper_bgcolor="#1a1a2e",
        font_color="white", title_font_color="#e91e8c",
        legend=dict(bgcolor="#1a1a2e"), xaxis=dict(tickangle=45)
    )
    st.plotly_chart(fig_gvt, use_container_width=True)

    st.subheader("📋 Dönemsel Hata Tablosu")

    with st.expander("ℹ️ Tablo Açıklaması", expanded=False):
        st.caption(
            "**Hata (TL):** Gerçek − Tahmin Edilen Ciro\n\n"
            "**Hata (%):** (Gerçek − Tahmin) / Gerçek × 100\n\n"
            "**Renk:** Sarı = küçük hata (iyi uyum) | Kırmızı = büyük hata\n\n"
            "**İşaret:** Pozitif = model düşük tahmin | Negatif = model yüksek tahmin"
        )

    if len(df_karsi) > 0:
        max_abs_hata = float(df_karsi["Hata_Pct"].abs().max())
        if max_abs_hata == 0:
            max_abs_hata = 1.0

        df_karsi_d = df_karsi.rename(columns={
            "Gercek_Ciro" : "Gerçek Ciro (TL)",
            "Tahmin_Ciro" : "Tahmin Edilen Ciro (TL)",
            "Hata_TL"     : "Hata (TL)",
            "Hata_Pct"    : "Hata (%)",
        })

        st.dataframe(
            df_karsi_d.style.format({
                "Gerçek Ciro (TL)"        : "{:,.0f}",
                "Tahmin Edilen Ciro (TL)" : "{:,.0f}",
                "Hata (TL)"               : "{:+,.0f}",
                "Hata (%)"                : "{:+.2f}%",
            }).background_gradient(
                subset=["Hata (%)"],
                cmap="RdYlGn_r",
                vmin=-max_abs_hata,
                vmax=max_abs_hata,
            ),
            use_container_width=True
        )