import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from veri_isleme import KANALLAR, KANAL_ETIKETLER
from optimizasyon import _tekli_tahmin_as


def _kanal_indeksleri(X):
    """AS kolonları varsa onları, yoksa normal kolonları kullan."""
    as_kolonlar = [k + "_AS" for k in KANALLAR]
    if as_kolonlar[0] in X.columns:
        return as_kolonlar, True
    return KANALLAR, False


def _tekli_tahmin_temel(
    butce_vektoru: np.ndarray,
    model,
    scaler,
    X: pd.DataFrame,
    df_orijinal: pd.DataFrame,
) -> float:
    """
    Temel MMM modeli (AS olmayan) için tek dönem tahmini.

    Bütçe vektörü (TL) + kontrol değişkeni ortalamaları → tam model.predict().

    Bu fonksiyon katsayı çarpımı değil, gerçek model.predict() kullanır.
    Intercept ve kontrol değişkeni etkileri de dahildir.

    Parametreler:
        butce_vektoru : [b1..b6] — her kanal için TL harcama
        model         : Eğitilmiş LinearRegression / Ridge nesnesi
        scaler        : mmm_regresyon'dan dönen StandardScaler
        X             : Eğitim özellik matrisi (sütun sırası için gerekli)
        df_orijinal   : Kontrol değişkeni ortalamalarını almak için
    """
    # Kontrol değişkeni sütunlarını X'ten çıkar (KANALLAR dışındakiler)
    kontrol_kolonlar = [c for c in X.columns if c not in KANALLAR]

    # Kontrol değişkenlerinin tarihsel ortalamalarını al
    if df_orijinal is not None:
        kontrol_ort = np.array([
            float(df_orijinal[k].mean()) if k in df_orijinal.columns else float(X[k].mean())
            for k in kontrol_kolonlar
        ])
    else:
        kontrol_ort = np.array([float(X[k].mean()) for k in kontrol_kolonlar])

    # Tam özellik vektörü: [kanal_TL × 6, kontrol × 3]
    x_row = np.concatenate([butce_vektoru, kontrol_ort]).reshape(1, -1)

    # Standardize + tahmin
    x_scaled = scaler.transform(x_row)
    return float(model.predict(x_scaled)[0])


def duyarlilik_analizi(
    model,
    scaler,
    X: pd.DataFrame,
    toplam_butce: float,
    df_orijinal: pd.DataFrame = None,
    degisim_araligi: list = None,
    as_params: dict = None,
    mm_scaler=None,
) -> pd.DataFrame:
    """
    Kanal bütçesi değişiminin tahmini satışa etkisini analiz eder.

    Her iki modda (AS ve temel) aynı mantık geçerlidir:
      - Seçilen kanalın bütçesi değiştirilir (±%X).
      - Diğer kanallar tarihsel aylık ortalamalarında SABİT tutulur.
      - Model.predict() ile tahmini satış hesaplanır.

    ÖNEMLI: Katsayı perturbasyon yaklaşımı KULLANILMAZ.
    Bu analiz "kanal bütçesinin satışa elastikiyetini" gösterir,
    "modelin katsayı hatasına duyarlılığını" değil.

    AS modu (as_params + mm_scaler verilmişse):
      → _tekli_tahmin_as() ile tam AS dönüşüm zinciri çalışır.

    Temel mod (as_params = None):
      → _tekli_tahmin_temel() ile direkt model.predict() çalışır.

    Sabit tutulan kanallar: tarihsel aylık ortalama bütçe (df_orijinal'dan).
    """
    if degisim_araligi is None:
        degisim_araligi = [-30, -20, -10, 0, 10, 20, 30]

    as_modu = (as_params is not None) and (mm_scaler is not None)
    sonuclar = []

    # Tarihsel kanal ortalamaları (diğer kanallar için sabit referans)
    if df_orijinal is not None:
        kanal_ort = {k: float(df_orijinal[k].mean()) for k in KANALLAR}
    else:
        # Fallback: X üzerinden (AS modda 0-1 ölçeğinde olabilir, dikkat)
        kanal_ort = {k: float(X[k].mean()) if k in X.columns else 0.0 for k in KANALLAR}

    # ── HER İKİ MOD: BÜTÇE DUYARLILIĞI ──────────────────────────────────────
    # Seçilen kanalın bütçesi değiştirilir, diğerleri sabit kalır.
    # "Sabit" = tarihsel aylık ortalama bütçe (df_orijinal bazlı).
    for i, kanal in enumerate(KANALLAR):
        for degisim_pct in degisim_araligi:
            # Temel bütçe vektörü: tüm kanallar ortalamada
            butce_vek = np.array([kanal_ort[k] for k in KANALLAR], dtype=float)
            # Sadece seçilen kanal değiştiriliyor
            butce_vek[i] = max(butce_vek[i] * (1 + degisim_pct / 100), 0.0)

            try:
                if as_modu:
                    # AS modu: tam dönüşüm zinciri (adstock → saturation → mm_scale → predict)
                    tahmin = _tekli_tahmin_as(
                        butce_vek, model, scaler, mm_scaler, as_params
                    )
                else:
                    # Temel mod: TL bütçe + kontrol ortalamaları → predict
                    tahmin = _tekli_tahmin_temel(
                        butce_vek, model, scaler, X, df_orijinal
                    )

                sonuclar.append({
                    "Degisen_Kanal"    : KANAL_ETIKETLER[kanal],
                    "Degisim_Pct"      : degisim_pct,
                    "Tahmini_Satis_TL" : tahmin,
                })
            except Exception:
                pass

    return pd.DataFrame(sonuclar)


def butce_duyarliligi(
    model,
    scaler,
    X: pd.DataFrame,
    df_orijinal: pd.DataFrame = None,
    butce_araligi: list = None,
    as_params: dict = None,
    mm_scaler=None,
) -> pd.DataFrame:
    """
    Toplam bütçeyi kademeli artırarak tahmini satış ve ROAS değişimini gösterir.

    AS modu:
      Bütçe tarihsel kanal oranlarına göre dağıtılır.
      _tekli_tahmin_as() ile tam AS dönüşüm zinciri çalışır (S-eğrisi görünür).

    Temel mod:
      Her bütçe seviyesi için kanal oranları ile doğrudan model.predict() kullanılır.
      Bu, intercept dahil tam model tahminini verir.

    ROAS = Tahmini Ciro / Toplam Bütçe
    Not: Bu oran modelin baseline (intercept) katkısını içerir.
    """
    as_modu = (as_params is not None) and (mm_scaler is not None)
    kolonlar, as_aktif = _kanal_indeksleri(X)

    if butce_araligi is None:
        if df_orijinal is not None:
            # Referans: orijinal TL harcamalarının aylık ortalaması
            mevcut = df_orijinal[KANALLAR].sum(axis=1).mean()
        elif not as_aktif:
            mevcut = X[kolonlar].sum(axis=1).mean()
        else:
            raise ValueError(
                "AS aktifken df_orijinal zorunludur. X, 0-1 ölçeğindedir."
            )

        butce_araligi = [
            round(mevcut * c)
            for c in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5, 2.0]
        ]

    # Kanal ağırlıkları: tarihsel ortalamaya göre orantılı dağılım
    if df_orijinal is not None:
        kanal_ort_arr = np.array([float(df_orijinal[k].mean()) for k in KANALLAR], dtype=float)
    else:
        kanal_ort_arr = np.ones(len(KANALLAR), dtype=float)

    toplam_ort = kanal_ort_arr.sum()
    kanal_oranlari = kanal_ort_arr / toplam_ort if toplam_ort > 0 else np.ones(len(KANALLAR)) / len(KANALLAR)

    sonuclar = []

    for butce in butce_araligi:
        try:
            butce_vektoru = kanal_oranlari * butce  # tarihsel orana göre dağıt

            if as_modu:
                # AS modu: tam AS dönüşüm zinciri
                tahmin = _tekli_tahmin_as(
                    butce_vektoru, model, scaler, mm_scaler, as_params
                )
            else:
                # Temel mod: tam model.predict() (intercept dahil)
                tahmin = _tekli_tahmin_temel(
                    butce_vektoru, model, scaler, X, df_orijinal
                )

            sonuclar.append({
                "Toplam_Butce_TL"  : butce,
                "Tahmini_Satis_TL" : tahmin,
                "ROAS"             : round(tahmin / butce, 2) if butce > 0 else 0,
            })
        except Exception:
            pass

    return pd.DataFrame(sonuclar)