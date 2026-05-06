import pandas as pd
import numpy as np
from pathlib import Path

# Kanal isimleri (proje genelinde standart)
KANALLAR = [
    "Google_Ads_Harcama_TL",
    "Meta_Harcama_TL",
    "YouTube_Harcama_TL",
    "Pazaryeri_Harcama_TL",
    "Influencer_Harcama_TL",
    "Twitter_X_Harcama_TL"
]

KANAL_ETIKETLER = {
    "Google_Ads_Harcama_TL"    : "Google Ads",
    "Meta_Harcama_TL"          : "Meta / Instagram",
    "YouTube_Harcama_TL"       : "YouTube",
    "Pazaryeri_Harcama_TL"     : "Pazaryeri",
    "Influencer_Harcama_TL"    : "Influencer",
    "Twitter_X_Harcama_TL"     : "Twitter / X"
}

def veri_yukle(dosya_yolu: str = None) -> pd.DataFrame:
    """Excel dosyasını yükler ve döndürür."""
    if dosya_yolu is None:
        base = Path(__file__).resolve().parent.parent
        dosya_yolu = base / "Data" / "Dijital_Reklam_Veri_Seti_2024_2025.xlsx"

    df = pd.read_excel(dosya_yolu, sheet_name="Ana_Veri_Seti", header=2)
    df.columns = df.columns.str.strip()
    return df

def veri_temizle(df: pd.DataFrame) -> pd.DataFrame:
    """Sütun isimlerini standardize eder ve veriyi temizler."""

    sutun_map = {
        "Ay"                        : "Ay",
        "Ciro (TL)"                 : "Ciro_TL",
        "Google Ads (TL)"           : "Google_Ads_Harcama_TL",
        "Meta / Instagram (TL)"     : "Meta_Harcama_TL",
        "YouTube (TL)"              : "YouTube_Harcama_TL",
        "Pazaryeri (TL)"            : "Pazaryeri_Harcama_TL",
        "Influencer (TL)"           : "Influencer_Harcama_TL",
        "Twitter/X (TL)"            : "Twitter_X_Harcama_TL",
        "Toplam Harcama (TL)"       : "Toplam_Reklam_Harcamasi_TL",
        "ROAS"                      : "ROAS",
        "Kampanya\nDönemi"          : "Kampanya_Donemi",
        "Fiyat\nEndeksi"            : "Fiyat_Endeksi",
        "Stok\nYok"                 : "Stok_Yok",
        "Mevsimsellik"              : "Mevsimsellik_Indeksi",
    }

    df = df.rename(columns=sutun_map)

    # Toplam satırını düşür
    df = df[df["Ay"].notna()].copy()
    df = df[df["Ay"] != "TOPLAM / ORT."].copy()

    # Sayısal dönüşüm
    sayisal = ["Ciro_TL"] + KANALLAR + [
        "Toplam_Reklam_Harcamasi_TL", "ROAS",
        "Kampanya_Donemi", "Fiyat_Endeksi",
        "Stok_Yok", "Mevsimsellik_Indeksi"
    ]
    for col in sayisal:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.reset_index(drop=True)
    return df

def ozellik_hazirla(df: pd.DataFrame) -> tuple:
    """
    MMM modeli için X (bağımsız) ve y (bağımlı) değişkenleri hazırlar.
    X: 6 kanal harcaması + 3 kontrol değişkeni
    y: Ciro_TL
    """
    ozellikler = KANALLAR + ["Kampanya_Donemi", "Fiyat_Endeksi", "Mevsimsellik_Indeksi"]
    X = df[ozellikler].copy()
    y = df["Ciro_TL"].copy()
    return X, y

def veri_ozeti(df: pd.DataFrame) -> dict:
    """Arayüzde gösterilecek temel istatistikleri döndürür."""
    return {
        "gozlem_sayisi"         : len(df),
        "toplam_ciro"           : df["Ciro_TL"].sum(),
        "toplam_harcama"        : df["Toplam_Reklam_Harcamasi_TL"].sum(),
        "ortalama_roas"         : df["ROAS"].mean(),
        "max_ciro_ay"           : df.loc[df["Ciro_TL"].idxmax(), "Ay"],
        "min_ciro_ay"           : df.loc[df["Ciro_TL"].idxmin(), "Ay"],
        "kampanya_sayisi"       : int(df["Kampanya_Donemi"].sum()),
        "stok_yok_sayisi"       : int(df["Stok_Yok"].sum()),
    }

# ── Test ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df_ham  = veri_yukle()
    df      = veri_temizle(df_ham)
    X, y    = ozellik_hazirla(df)
    ozet    = veri_ozeti(df)

    print("✅ Veri başarıyla yüklendi!")
    print(f"   Satır sayısı : {len(df)}")
    print(f"   Sütunlar     : {list(df.columns)}")
    print(f"\n📊 Özet İstatistikler:")
    for k, v in ozet.items():
        print(f"   {k:25s}: {v}")
    print(f"\n🔢 X shape: {X.shape}")
    print(f"🎯 y shape: {y.shape}")

# ══════════════════════════════════════════════════════════════════════════
# ADSTOCK & SATURATION FONKSİYONLARI
# ══════════════════════════════════════════════════════════════════════════

def adstock_donusumu(seri: pd.Series, decay_rate: float) -> pd.Series:
    """
    Geometrik Adstock dönüşümü.
    A_t = X_t + decay_rate * A_(t-1)

    decay_rate: 0-1 arası
        0.0 → reklam etkisi anlık, gecikmeli etki yok
        0.9 → reklam etkisi uzun süre devam eder
    """
    adstock = np.zeros(len(seri))
    for t in range(len(seri)):
        if t == 0:
            adstock[t] = seri.iloc[t]
        else:
            adstock[t] = seri.iloc[t] + decay_rate * adstock[t - 1]
    return pd.Series(adstock, index=seri.index)


def saturation_donusumu(seri: pd.Series, beta: float = 2.0, gamma: float = None) -> pd.Series:
    """
    Hill Saturation fonksiyonu — normalize edilmiş çıktı [0, 1] arası.
    S(X) = X^beta / (X^beta + gamma^beta)

    beta  → eğim parametresi (varsayılan 2.0)
    gamma → doygunluk eşiği (varsayılan serinin medyanı)
    """
    if gamma is None:
        gamma = seri.median()
    if gamma == 0:
        gamma = 1.0
    # Önce normalize et [0,1] aralığına
    seri_norm = seri / (seri.max() + 1e-8)
    gamma_norm = gamma / (seri.max() + 1e-8)
    result = seri_norm ** beta / (seri_norm ** beta + gamma_norm ** beta)
    return result


def adstock_saturation_uygula(
    df: pd.DataFrame,
    decay_rates: dict = None,
    beta: float = 2.0,
) -> pd.DataFrame:
    """
    Tüm kanallara adstock ve saturation uygular.
    Yeni sütunlar _AS (AdStock+Saturation) son ekiyle eklenir.

    decay_rates: Her kanal için decay oranı dict'i
                 Varsayılan: tüm kanallar için 0.5
    """
    if decay_rates is None:
        decay_rates = {k: 0.5 for k in KANALLAR}

    df_yeni = df.copy()
    for kanal in KANALLAR:
        # 1. Adstock
        adstock = adstock_donusumu(df[kanal], decay_rates[kanal])
        # 2. Saturation
        saturation = saturation_donusumu(adstock, beta=beta)
        df_yeni[kanal + "_AS"] = saturation

    return df_yeni


def ozellik_hazirla_as(df: pd.DataFrame) -> tuple:
    """
    Adstock+Saturation uygulanmış özelliklerle X ve y hazırlar.
    """
    as_kolonlar = [k + "_AS" for k in KANALLAR]
    kontrol     = ["Kampanya_Donemi", "Fiyat_Endeksi", "Mevsimsellik_Indeksi"]
    ozellikler  = as_kolonlar + kontrol
    X = df[ozellikler].copy()
    y = df["Ciro_TL"].copy()
    return X, y


# ══════════════════════════════════════════════════════════════════════════
# YENİ: GA + AS OPTİMİZASYONU İÇİN YARDIMCI FONKSİYON
# ══════════════════════════════════════════════════════════════════════════

def as_parametreleri_cikart(
    df: pd.DataFrame,
    decay_rates: dict = None,
    beta: float = 2.0,
) -> dict:
    """
    Genetik Algoritma + Adstock + Saturation optimizasyonu için eğitim
    verisinden gerekli parametreleri çıkarır.

    Bu fonksiyon tek dönemlik (tek ay) bir bütçe kararı için gerekli
    tüm referans noktalarını döndürür:

    - son_carryover   : Her kanalın son dönem adstock değeri (TL).
                        Yeni ay hesabında "geçen aydan kalan etki" olarak kullanılır.
    - adstock_max     : Her kanalın adstock serisinin maksimumu (TL).
                        Saturation fonksiyonundaki normalizasyon için gerekli.
    - adstock_gamma   : Her kanalın adstock serisinin medyanı (TL).
                        Hill fonksiyonundaki doygunluk eşiği (gamma parametresi).
    - kontrol_ortalama: Kontrol değişkenlerinin (Kampanya, Fiyat, Mevsimsellik)
                        eğitim verisi ortalamaları. Tek dönem tahmininde sabit
                        olarak kullanılır.
    - decay_rates     : Kullanılan decay oranları (dönüşüm sırasında tutarlılık için saklanır).
    - beta            : Kullanılan saturation beta parametresi.

    Kullanım:
        as_params = as_parametreleri_cikart(df, decay_rates, beta)
        # → sonra genetik_optimizasyon_as() fonksiyonuna geçilir
    """
    if decay_rates is None:
        decay_rates = {k: 0.5 for k in KANALLAR}

    kontrol_kolonlar = ["Kampanya_Donemi", "Fiyat_Endeksi", "Mevsimsellik_Indeksi"]

    son_carryover  = {}
    adstock_max    = {}
    adstock_gamma  = {}

    for kanal in KANALLAR:
        adstock_serisi = adstock_donusumu(df[kanal], decay_rates[kanal])
        # Son dönemden gelen carryover (bir sonraki döneme taşınan etki)
        son_carryover[kanal]  = float(adstock_serisi.iloc[-1])
        # Maksimum değer: saturation normalizasyonu bu değere göre yapılır
        adstock_max[kanal]    = float(adstock_serisi.max())
        # Medyan değer: Hill fonksiyonunda gamma (doygunluk eşiği) olarak kullanılır
        adstock_gamma[kanal]  = float(adstock_serisi.median())

    # Kontrol değişkenlerinin ortalaması (eğitim verisi referansı)
    kontrol_ortalama = {
        k: float(df[k].mean())
        for k in kontrol_kolonlar
        if k in df.columns
    }

    return {
        "son_carryover"    : son_carryover,
        "adstock_max"      : adstock_max,
        "adstock_gamma"    : adstock_gamma,
        "kontrol_ortalama" : kontrol_ortalama,
        "decay_rates"      : decay_rates,
        "beta"             : beta,
    }