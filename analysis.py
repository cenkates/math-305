import pandas as pd
import numpy as np
from ldf import local_dependence
from rolling_ldf import rolling_ldf_matrix



# Log-return dosyasını oku
log_returns = pd.read_csv("data/processed/nasdaq_log_returns.csv", index_col=0, parse_dates=True)

print("Veri seti boyutu:", log_returns.shape)   # (gün sayısı, hisse sayısı)
print("\nİlk 5 satır:")
print(log_returns.head())

print("\nÖzet istatistikler:")
print(log_returns.describe())

# Korelasyon matrisi
corr = log_returns.corr()
corr.to_csv("data/processed/nasdaq_corr_matrix.csv")
print("\nKorelasyon matrisi kaydedildi: data/processed/nasdaq_corr_matrix.csv")
from ldf import local_dependence

# İki hisse seç
aapl = log_returns["AAPL"]
msft = log_returns["MSFT"]

# LDF hesapla
ldf_vals = local_dependence(aapl, msft)

print("\nLDF örnek çıktısı:")
print(ldf_vals[:10])
from rolling_ldf import rolling_ldf_matrix

print("\nRolling LDF hesaplanıyor...")
ldf_rolling = rolling_ldf_matrix(log_returns)

np.save("data/processed/rolling_ldf.npy", ldf_rolling)

print("Rolling LDF boyutu:", ldf_rolling.shape)
print("Örnek pencere LDF matrisi:\n", ldf_rolling[-1])
