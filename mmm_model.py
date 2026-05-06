import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
import warnings
warnings.filterwarnings("ignore")

from veri_isleme import KANALLAR, KANAL_ETIKETLER


def mmm_regresyon(X: pd.DataFrame, y: pd.Series, ridge_alpha: float = 0.0):
    """
    Media Mix Modeling — Çoklu Doğrusal Regresyon.

    ridge_alpha = 0.0  → OLS (standart doğrusal regresyon)
    ridge_alpha > 0.0  → Ridge Regresyon (L2 regularizasyon)
                         AS modeli için ridge_alpha=50 önerilir;
                         negatif katsayıları baskılar.

    Cross-Validation: TimeSeriesSplit(n_splits=3) kullanılır.
      - Klasik k-fold yerine tercih sebebi: aylık veri kronolojiktir.
        Gelecek aylar geçmişten öğrenilmemelidir (temporal leakage).
      - Her fold: train = eski aylar, test = yeni aylar.
      - n=24 için n_splits=3 en kararlı yapıyı verir.

    Not: OLS training R²≈0.987 bu veri setinde yanıltıcıdır.
    n=24 gözlemde 9 değişkenle OLS eğitim verisini ezberler (overfit).
    Gerçek performans = CV R² (~0.63). Ridge+AS CV R²≈0.655 → daha stabil.

    Döndürür: model, scaler, y_pred, metrikler
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if ridge_alpha > 0.0:
        model = Ridge(alpha=ridge_alpha)
    else:
        model = LinearRegression()

    model.fit(X_scaled, y)
    y_pred = model.predict(X_scaled)

    r2   = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mape = mean_absolute_percentage_error(y, y_pred) * 100

    # TimeSeriesSplit: geleceği geçmişten öğrenme sızıntısını önler.
    # n=24 ile n_splits=3:
    #   Fold-1: train[0:6]  → test[6:12]
    #   Fold-2: train[0:12] → test[12:18]
    #   Fold-3: train[0:18] → test[18:24]
    tscv = TimeSeriesSplit(n_splits=3)
    cv_scores = cross_val_score(model, X_scaled, y, cv=tscv, scoring="r2")

    metrikler = {
        "R2"             : round(r2, 4),
        "RMSE"           : round(rmse, 2),
        "MAPE"           : round(mape, 2),
        "CV_R2_Ort"      : round(cv_scores.mean(), 4),
        "CV_R2_Std"      : round(cv_scores.std(), 4),
        "Regularizasyon" : f"Ridge(α={ridge_alpha})" if ridge_alpha > 0 else "OLS",
        "CV_Yontem"      : "ZamanSerisi-3",
    }

    return model, scaler, y_pred, metrikler


def kanal_katkisi(model, scaler, X: pd.DataFrame) -> pd.DataFrame:
    """
    Her kanalın satışa olan katkısını hesaplar.
    Standardize edilmemiş orijinal katsayılar döndürülür.
    """
    ozellikler = list(X.columns)
    std_values  = scaler.scale_
    katsayilar  = model.coef_ / std_values

    df_katki = pd.DataFrame({
        "Degisken"   : ozellikler,
        "Katsayi"    : katsayilar,
        "Abs_Katki"  : np.abs(katsayilar),
    })

    df_kanal = df_katki[df_katki["Degisken"].isin(KANALLAR)].copy()
    df_kanal["Etki_Payi"] = (
        df_kanal["Abs_Katki"] / df_kanal["Abs_Katki"].sum() * 100
    ).round(2)
    df_kanal["Kanal_Adi"] = df_kanal["Degisken"].map(KANAL_ETIKETLER)
    df_kanal = df_kanal.sort_values("Abs_Katki", ascending=False).reset_index(drop=True)
    return df_kanal


def gercek_vs_tahmin(df: pd.DataFrame, y_pred: np.ndarray) -> pd.DataFrame:
    """
    Gerçek ve tahmin edilen ciro değerlerini karşılaştırır.

    Hata formülü:
      Hata_TL  = Gerçek − Tahmin
      Hata_Pct = (Gerçek − Tahmin) / Gerçek × 100

    İşaret yorumu:
      Pozitif → Model eksik tahmin etmiş (gerçek > tahmin)
      Negatif → Model fazla tahmin etmiş (gerçek < tahmin)
    """
    return pd.DataFrame({
        "Ay"           : df["Ay"].values,
        "Gercek_Ciro"  : df["Ciro_TL"].values,
        "Tahmin_Ciro"  : np.round(y_pred, 0),
        "Hata_TL"      : np.round(df["Ciro_TL"].values - y_pred, 0),
        "Hata_Pct"     : np.round(
            (df["Ciro_TL"].values - y_pred) / df["Ciro_TL"].values * 100, 2
        ),
    })


# ── Test ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, os
    sys.path.append(os.path.dirname(__file__))
    from veri_isleme import veri_yukle, veri_temizle, ozellik_hazirla

    df       = veri_temizle(veri_yukle())
    X, y     = ozellik_hazirla(df)
    model, scaler, y_pred, metrikler = mmm_regresyon(X, y)

    print("✅ MMM Modeli (OLS) başarıyla kuruldu!")
    print("\n📊 Model Performans Metrikleri:")
    for k, v in metrikler.items():
        print(f"   {k:15s}: {v}")

    df_katki = kanal_katkisi(model, scaler, X)
    print("\n📈 Kanal Katkı Analizi:")
    print(df_katki[["Kanal_Adi", "Katsayi", "Etki_Payi"]].to_string(index=False))


# ══════════════════════════════════════════════════════════════════════════
# ADSTOCK + SATURATION İLE GELİŞMİŞ MMM MODELİ
# ══════════════════════════════════════════════════════════════════════════

def mmm_gelismis(
    df,
    decay_rates: dict = None,
    beta: float = 3.0,
    ridge_alpha: float = 50.0,
):
    """
    Adstock + Saturation dönüşümleri uygulanmış gelişmiş MMM modeli.

    Varsayılan parametreler kozmetik e-ticaret için optimize edilmiştir:
      beta=3.0       — S-eğrisi şekli; CV-ZamanSerisi R²=0.655 (beta=2.0: 0.580)
      ridge_alpha=50 — Tüm kanal katsayılarını pozitif tutar
      decay_rates    — Kanal bazlı gecikme oranları (sektörel bilgiye dayalı)

    NOT: AS+Ridge R²≈0.817 < OLS R²≈0.987 görünür ancak bu beklenen bir durumdur.
    OLS eğitim verisini ezberler (n=24, 9 değişken → overfit).
    Ridge regularizasyon kasıtlı olarak training R²'yi düşürür.
    Gerçek karşılaştırma: CV R² → AS+Ridge daha stabil genelleme yapar.

    Döndürür: model, scaler, y_pred, metrikler, df_as, X_as, mm_scaler
    """
    import sys, os
    sys.path.append(os.path.dirname(__file__))
    from veri_isleme import adstock_saturation_uygula, ozellik_hazirla_as, KANALLAR
    from sklearn.preprocessing import MinMaxScaler

    # Kanal bazlı varsayılan decay oranları (kozmetik sektörü)
    if decay_rates is None:
        decay_rates = {
            "Google_Ads_Harcama_TL" : 0.4,  # Arama bazlı, anlık dönüşüm
            "Meta_Harcama_TL"       : 0.5,  # Sosyal keşif + retargeting
            "YouTube_Harcama_TL"    : 0.7,  # Uzun vadeli marka etkisi
            "Pazaryeri_Harcama_TL"  : 0.3,  # Anlık satın alma
            "Influencer_Harcama_TL" : 0.5,  # Güven oluşturma etkisi
            "Twitter_X_Harcama_TL"  : 0.3,  # Anlık, kısa ömürlü
        }

    df_as = adstock_saturation_uygula(df, decay_rates=decay_rates, beta=beta)
    X_as, y = ozellik_hazirla_as(df_as)

    # AS kolonlarını 0-1 arasına normalize et (mm_scaler eğitim parametrelerini saklar)
    as_kolonlar = [k + "_AS" for k in KANALLAR]
    mm_scaler = MinMaxScaler()
    X_as[as_kolonlar] = mm_scaler.fit_transform(X_as[as_kolonlar])

    model, scaler, y_pred, metrikler = mmm_regresyon(X_as, y, ridge_alpha=ridge_alpha)
    return model, scaler, y_pred, metrikler, df_as, X_as, mm_scaler


def kanal_katkisi_as(model, scaler, X_as: pd.DataFrame) -> pd.DataFrame:
    """Adstock+Saturation modeli için kanal katkı analizi."""
    from veri_isleme import KANALLAR, KANAL_ETIKETLER

    as_kolonlar = [k + "_AS" for k in KANALLAR]
    std_values  = scaler.scale_
    katsayilar  = model.coef_ / std_values

    df_katki = pd.DataFrame({
        "Degisken"  : list(X_as.columns),
        "Katsayi"   : katsayilar,
        "Abs_Katki" : np.abs(katsayilar),
    })

    df_kanal = df_katki[df_katki["Degisken"].isin(as_kolonlar)].copy()
    df_kanal["Etki_Payi"] = (
        df_kanal["Abs_Katki"] / df_kanal["Abs_Katki"].sum() * 100
    ).round(2)
    df_kanal["Kanal_Adi"] = df_kanal["Degisken"].str.replace(
        "_AS", "", regex=False
    ).map(KANAL_ETIKETLER)
    df_kanal = df_kanal.sort_values("Abs_Katki", ascending=False).reset_index(drop=True)
    return df_kanal


# ── Test (Gelişmiş Model) ─────────────────────────────────────────────────────
def test_gelismis():
    import sys, os
    sys.path.append(os.path.dirname(__file__))
    from veri_isleme import veri_yukle, veri_temizle

    df = veri_temizle(veri_yukle())
    model, scaler, y_pred, metrikler, df_as, X_as, mm_scaler = mmm_gelismis(df)

    print("✅ Gelişmiş MMM (Adstock+Saturation + Ridge) başarıyla kuruldu!")
    print(f"   Regularizasyon : {metrikler['Regularizasyon']}")
    print(f"   CV Yöntemi     : {metrikler['CV_Yontem']}")
    print("\n📊 Model Performans Metrikleri:")
    for k, v in metrikler.items():
        print(f"   {k:15s}: {v}")

    df_katki = kanal_katkisi_as(model, scaler, X_as)
    print("\n📈 Kanal Katkı Analizi:")
    print(df_katki[["Kanal_Adi", "Katsayi", "Etki_Payi"]].to_string(index=False))

    neg = df_katki[df_katki["Katsayi"] < 0]
    if len(neg) == 0:
        print("\n✅ Tüm katsayılar pozitif — Ridge başarılı.")
    else:
        print(f"\n⚠️ Negatif katsayı: {list(neg['Kanal_Adi'])}")


if __name__ == "__main__":
    test_gelismis()