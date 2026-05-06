import numpy as np
import pandas as pd
from scipy.optimize import linprog, differential_evolution
import warnings
warnings.filterwarnings("ignore")

from veri_isleme import KANALLAR, KANAL_ETIKETLER


def _kanal_kolonlari(X):
    """AS kolonları varsa onları, yoksa normal kolonları döndür."""
    as_kolonlar = [k + "_AS" for k in KANALLAR]
    if as_kolonlar[0] in X.columns:
        return as_kolonlar
    return KANALLAR


def _beta_hesapla(model, scaler, X):
    """Kanal katsayılarını hesaplar ve gerekirse normalize eder."""
    kolonlar = _kanal_kolonlari(X)
    kanal_indeksleri = [list(X.columns).index(k) for k in kolonlar]

    std_values = scaler.scale_
    beta_tum = model.coef_ / std_values
    beta_kanallar = beta_tum[kanal_indeksleri]

    maks = np.max(np.abs(beta_kanallar))
    if maks > 1000:
        beta_kanallar = beta_kanallar / maks

    return beta_kanallar


def _mevcut_butceler(df_orijinal):
    """Orijinal veri setinden ortalama aylık harcamaları döndürür."""
    return {k: round(df_orijinal[k].mean(), 2) for k in KANALLAR}


def _mevcut_butceler_fallback(X):
    """
    df_orijinal verilmemişse X üzerinden mevcut bütçeleri tahmin eder.

    Temel modelde X TL ölçeğindedir — doğrudan kullanılabilir.
    AS modunda X 0–1 ölçeğindedir — orijinal sütun adları X'te yoktur,
    bu durumda 0 döner (görsel uyarı amacıyla).
    """
    return {k: round(X[k].mean() if k in X.columns else 0, 2) for k in KANALLAR}


def _yumusatma_bounds(
    min_butceler: dict,
    max_butceler: dict,
    max_degisim_orani: float,
    mevcut: dict,
) -> tuple:
    """
    Yumuşatma kısıtı: her kanalın bütçesi mevcut ortalamadan
    en fazla ±max_degisim_orani kadar değişebilir.

    Sonuçta min/max sınırları daraltılır; mevcut bound'ların dışına çıkılmaz.
    lb > ub durumunda lb = ub alınır (sıfır serbestlik: kanal sabit kalır).

    Döndürür: (min_butceler_yeni, max_butceler_yeni)
    """
    min_yeni = dict(min_butceler)
    max_yeni = dict(max_butceler)

    for k in KANALLAR:
        m = mevcut.get(k, 0)
        if m > 0:
            lb = max(min_butceler[k], m * (1 - max_degisim_orani))
            ub = min(max_butceler[k], m * (1 + max_degisim_orani))
            if lb > ub:
                lb = ub  # infeasible: kanalın bütçesi sabit tutulur
            min_yeni[k] = lb
            max_yeni[k] = ub

    return min_yeni, max_yeni


def butce_optimizasyonu(
    model,
    scaler,
    X: pd.DataFrame,
    toplam_butce: float,
    min_butceler: dict = None,
    max_butceler: dict = None,
    df_orijinal: pd.DataFrame = None,
    max_degisim_orani: float = None,
) -> dict:
    """
    Doğrusal Programlama ile bütçe optimizasyonu.

    toplam_butce, min_butceler, max_butceler : Her zaman gerçek TL değerleri.
    df_orijinal      : Orijinal ham veri seti. Sonuç tablosundaki "mevcut_butceler"
                       sütununun gerçek TL ile gösterilmesi için kullanılır.
    max_degisim_orani: 0.0–1.0 arası; örn. 0.40 = ±%40 değişim sınırı.
                       Her kanalın optimal bütçesi tarihsel ortalamasından
                       bu kadar sapabilir. None ise sınır uygulanmaz.
    """
    beta_kanallar = _beta_hesapla(model, scaler, X)
    n = len(KANALLAR)

    # Varsayılan bütçe sınırları
    if min_butceler is None:
        min_butceler = {k: 0.05 * toplam_butce / n for k in KANALLAR}
    if max_butceler is None:
        max_butceler = {k: 0.50 * toplam_butce for k in KANALLAR}

    # Yumuşatma kısıtı uygula (aktifse)
    if max_degisim_orani is not None:
        mevcut = (
            _mevcut_butceler(df_orijinal) if df_orijinal is not None
            else _mevcut_butceler_fallback(X)
        )
        min_butceler, max_butceler = _yumusatma_bounds(
            min_butceler, max_butceler, max_degisim_orani, mevcut
        )

    c = -beta_kanallar
    A_eq = [[1.0] * n]
    b_eq = [toplam_butce]
    bounds = [(min_butceler.get(k, 0), max_butceler.get(k, toplam_butce)) for k in KANALLAR]

    sonuc = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")

    if not sonuc.success:
        raise ValueError(f"Optimizasyon başarısız: {sonuc.message}")

    optimal_butceler = dict(zip(KANALLAR, np.round(sonuc.x, 2)))
    tahmini_katki = {
        k: round(beta_kanallar[i] * optimal_butceler[k], 2)
        for i, k in enumerate(KANALLAR)
    }

    # Mevcut bütçe karşılaştırması: orijinal TL değerleri kullanılır.
    if df_orijinal is not None:
        mevcut = _mevcut_butceler(df_orijinal)
    else:
        mevcut = _mevcut_butceler_fallback(X)

    return {
        "optimal_butceler"  : optimal_butceler,
        "mevcut_butceler"   : mevcut,
        "tahmini_katki"     : tahmini_katki,
        "toplam_butce"      : toplam_butce,
        "tahmini_satis"     : round(-sonuc.fun, 2),
        "yontem"            : "Doğrusal Programlama",
    }


def genetik_optimizasyon(
    model,
    scaler,
    X: pd.DataFrame,
    toplam_butce: float,
    min_butceler: dict = None,
    max_butceler: dict = None,
    df_orijinal: pd.DataFrame = None,
    max_degisim_orani: float = None,
) -> dict:
    """
    Genetik Algoritma (Diferansiyel Evrim) ile bütçe optimizasyonu.

    toplam_butce, min_butceler, max_butceler : Her zaman gerçek TL değerleri.
    df_orijinal      : butce_optimizasyonu ile aynı amaçla kullanılır.
    max_degisim_orani: Yumuşatma kısıtı; None ise uygulanmaz.

    NOT: Bu fonksiyon katsayı × bütçe (doğrusal) yaklaşımını kullanır.
         AS aktifken gerçek predict tabanlı optimizasyon için
         genetik_optimizasyon_as() fonksiyonunu kullanınız.
    """
    beta_kanallar = _beta_hesapla(model, scaler, X)
    n = len(KANALLAR)

    # Varsayılan bütçe sınırları
    if min_butceler is None:
        min_butceler = {k: 0.05 * toplam_butce / n for k in KANALLAR}
    if max_butceler is None:
        max_butceler = {k: 0.50 * toplam_butce for k in KANALLAR}

    # Yumuşatma kısıtı uygula (aktifse)
    if max_degisim_orani is not None:
        mevcut = (
            _mevcut_butceler(df_orijinal) if df_orijinal is not None
            else _mevcut_butceler_fallback(X)
        )
        min_butceler, max_butceler = _yumusatma_bounds(
            min_butceler, max_butceler, max_degisim_orani, mevcut
        )

    bounds = [
        (min_butceler.get(k, 0), max_butceler.get(k, toplam_butce))
        for k in KANALLAR
    ]

    def amac_fonksiyonu(x):
        ceza  = abs(sum(x) - toplam_butce) * 1e6
        katki = sum(beta_kanallar[i] * x[i] for i in range(n))
        return -katki + ceza

    sonuc = differential_evolution(
        amac_fonksiyonu,
        bounds=bounds,
        seed=42,
        maxiter=1000,
        tol=1e-8,
        polish=True,
    )

    optimal_butceler = dict(zip(KANALLAR, np.round(sonuc.x, 2)))
    tahmini_katki = {
        k: round(beta_kanallar[i] * optimal_butceler[k], 2)
        for i, k in enumerate(KANALLAR)
    }

    if df_orijinal is not None:
        mevcut = _mevcut_butceler(df_orijinal)
    else:
        mevcut = _mevcut_butceler_fallback(X)

    return {
        "optimal_butceler"  : optimal_butceler,
        "mevcut_butceler"   : mevcut,
        "tahmini_katki"     : tahmini_katki,
        "toplam_butce"      : toplam_butce,
        "tahmini_satis"     : round(-sonuc.fun, 2),
        "yontem"            : "Genetik Algoritma (Diferansiyel Evrim)",
    }


# ══════════════════════════════════════════════════════════════════════════
# GA + ADSTOCK + SATURATION — GERÇEK PREDICT TABANLI OPTİMİZASYON
# ══════════════════════════════════════════════════════════════════════════

def _tekli_tahmin_as(
    butce_vektoru: np.ndarray,
    model,
    scaler,
    mm_scaler,
    as_params: dict,
) -> float:
    """
    Tek bir bütçe vektörü (TL) için tam AS dönüşüm zincirini çalıştırır
    ve model.predict() ile tahmini satışı döndürür.

    Zincir:
      TL bütçe  →  Adstock (single step)
               →  Saturation (Hill fonksiyonu, eğitim parametreleriyle)
               →  MinMaxScaler (mm_scaler ile, eğitim ölçeğine normalize)
               →  Kontrol değişkenleri eklenir (eğitim ortalamaları)
               →  StandardScaler (scaler ile)
               →  model.predict()

    Parametreler:
        butce_vektoru : [b1, ..., b6] — her kanal için TL harcama
        model         : Eğitilmiş Ridge/LinearRegression nesnesi
        scaler        : mmm_regresyon'dan dönen StandardScaler
        mm_scaler     : mmm_gelismis'ten dönen MinMaxScaler (AS kolonları için)
        as_params     : as_parametreleri_cikart() çıktısı (carryover, max, gamma vb.)
    """
    decay_rates = as_params["decay_rates"]
    beta        = as_params["beta"]

    # ── ADIM 1: Tek dönemlik adstock ─────────────────────────────────────────
    # A_yeni = harcama + decay × son_carryover (eğitim verisinin son değeri)
    adstock_vals = []
    for i, kanal in enumerate(KANALLAR):
        carry  = as_params["son_carryover"][kanal]
        decay  = decay_rates[kanal]
        a_yeni = butce_vektoru[i] + decay * carry
        adstock_vals.append(a_yeni)

    # ── ADIM 2: Saturation (Hill fonksiyonu) ─────────────────────────────────
    # S = x_norm^beta / (x_norm^beta + gamma_norm^beta)
    sat_vals = []
    for i, kanal in enumerate(KANALLAR):
        a       = adstock_vals[i]
        a_max   = as_params["adstock_max"][kanal]
        gamma   = as_params["adstock_gamma"][kanal]

        a_norm     = a     / (a_max + 1e-8)
        gamma_norm = gamma / (a_max + 1e-8)
        sat = a_norm ** beta / (a_norm ** beta + gamma_norm ** beta + 1e-10)
        sat_vals.append(sat)

    # ── ADIM 3: MinMax ölçekleme (mm_scaler) ─────────────────────────────────
    sat_arr    = np.array(sat_vals, dtype=float).reshape(1, -1)
    sat_scaled = mm_scaler.transform(sat_arr)[0]

    # ── ADIM 4: Kontrol değişkenleri ─────────────────────────────────────────
    kontrol = np.array([
        as_params["kontrol_ortalama"].get("Kampanya_Donemi",      0.0),
        as_params["kontrol_ortalama"].get("Fiyat_Endeksi",        1.0),
        as_params["kontrol_ortalama"].get("Mevsimsellik_Indeksi", 1.0),
    ], dtype=float)

    # ── ADIM 5: Tam özellik vektörü ──────────────────────────────────────────
    # Sıra: [kanal_AS × 6, Kampanya_Donemi, Fiyat_Endeksi, Mevsimsellik_Indeksi]
    x_row = np.concatenate([sat_scaled, kontrol]).reshape(1, -1)

    # ── ADIM 6: StandardScaler + predict ─────────────────────────────────────
    x_std = scaler.transform(x_row)
    pred  = model.predict(x_std)[0]
    return float(pred)


def genetik_optimizasyon_as(
    model,
    scaler,
    X_as: pd.DataFrame,
    toplam_butce: float,
    as_params: dict,
    mm_scaler,
    min_butceler: dict = None,
    max_butceler: dict = None,
    df_orijinal: pd.DataFrame = None,
    max_degisim_orani: float = None,
) -> dict:
    """
    GA + Adstock + Saturation — Gerçek model.predict tabanlı optimizasyon.

    Her aday bütçe çözümü için tam dönüşüm zinciri çalışır:
        TL bütçe → adstock → saturation → mm_scale → std_scale → model.predict

    Parametreler:
        model             : mmm_gelismis'ten dönen eğitilmiş model
        scaler            : mmm_gelismis'ten dönen StandardScaler
        X_as              : mmm_gelismis'ten dönen dönüştürülmüş özellik matrisi
        toplam_butce      : Sidebar'dan gelen gerçek TL değeri
        as_params         : as_parametreleri_cikart() çıktısı
        mm_scaler         : mmm_gelismis'ten dönen MinMaxScaler
        min_butceler      : Her kanal için minimum TL bütçe (opsiyonel)
        max_butceler      : Her kanal için maksimum TL bütçe (opsiyonel)
        df_orijinal       : Mevcut bütçe karşılaştırması için orijinal veri seti
        max_degisim_orani : Yumuşatma kısıtı; None ise uygulanmaz
    """
    n = len(KANALLAR)

    # Varsayılan bütçe sınırları
    if min_butceler is None:
        min_butceler = {k: 0.05 * toplam_butce / n for k in KANALLAR}
    if max_butceler is None:
        max_butceler = {k: 0.50 * toplam_butce for k in KANALLAR}

    # Yumuşatma kısıtı uygula (aktifse)
    if max_degisim_orani is not None:
        mevcut_ref = (
            _mevcut_butceler(df_orijinal) if df_orijinal is not None
            else {k: 0.0 for k in KANALLAR}
        )
        min_butceler, max_butceler = _yumusatma_bounds(
            min_butceler, max_butceler, max_degisim_orani, mevcut_ref
        )

    bounds = [
        (min_butceler.get(k, 0), max_butceler.get(k, toplam_butce))
        for k in KANALLAR
    ]

    def amac_fonksiyonu_as(x):
        ceza = abs(sum(x) - toplam_butce) * 1e6
        try:
            pred = _tekli_tahmin_as(
                np.array(x, dtype=float),
                model, scaler, mm_scaler, as_params,
            )
        except Exception:
            pred = -1e12
        return -pred + ceza

    sonuc = differential_evolution(
        amac_fonksiyonu_as,
        bounds=bounds,
        seed=42,
        maxiter=1000,
        tol=1e-8,
        polish=True,
    )

    optimal_butceler = dict(zip(KANALLAR, np.round(sonuc.x, 2)))

    # Optimal noktada temiz tahmin (ceza etkisi olmadan)
    tahmini_satis = round(
        _tekli_tahmin_as(sonuc.x, model, scaler, mm_scaler, as_params), 2
    )

    # Kanal bazlı marjinal katkı: ±%5 nümerik türev
    tahmini_katki = {}
    for i, k in enumerate(KANALLAR):
        delta = max(optimal_butceler[k] * 0.05, 1000)
        b_yukar = sonuc.x.copy(); b_yukar[i] += delta
        b_asagi = sonuc.x.copy(); b_asagi[i] -= delta
        try:
            p_yukar = _tekli_tahmin_as(b_yukar, model, scaler, mm_scaler, as_params)
            p_asagi = _tekli_tahmin_as(b_asagi, model, scaler, mm_scaler, as_params)
            marginal = (p_yukar - p_asagi) / (2 * delta)
            tahmini_katki[k] = round(marginal * optimal_butceler[k], 2)
        except Exception:
            tahmini_katki[k] = 0.0

    if df_orijinal is not None:
        mevcut = _mevcut_butceler(df_orijinal)
    else:
        mevcut = {k: 0.0 for k in KANALLAR}

    return {
        "optimal_butceler"  : optimal_butceler,
        "mevcut_butceler"   : mevcut,
        "tahmini_katki"     : tahmini_katki,
        "toplam_butce"      : toplam_butce,
        "tahmini_satis"     : tahmini_satis,
        "yontem"            : "Genetik Algoritma + Adstock + Saturation (Predict Tabanlı)",
    }


def optimizasyon_tablosu(sonuc: dict) -> pd.DataFrame:
    """Optimizasyon sonuçlarını karşılaştırmalı tablo olarak döndürür."""
    satirlar = []
    for k in KANALLAR:
        mevcut  = sonuc["mevcut_butceler"][k]
        optimal = sonuc["optimal_butceler"][k]
        katki   = sonuc["tahmini_katki"][k]
        degisim = round(((optimal - mevcut) / mevcut) * 100, 1) if mevcut > 0 else 0
        satirlar.append({
            "Kanal"             : KANAL_ETIKETLER[k],
            "Mevcut_Butce_TL"   : mevcut,
            "Optimal_Butce_TL"  : optimal,
            "Degisim_Pct"       : degisim,
            "Tahmini_Katki_TL"  : katki,
        })
    df = pd.DataFrame(satirlar)
    df = df.sort_values("Optimal_Butce_TL", ascending=False).reset_index(drop=True)
    return df