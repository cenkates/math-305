import yfinance as yf
import pandas as pd
import numpy as np
import os

# Klasörleri oluştur (yoksa)
os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

tickers = ["AAPL", "MSFT", "AMZN", "GOOGL", "META"]

# 2) Veri indirme – auto_adjust=False ekliyoruz
raw_data = yf.download(
    tickers,
    start="2018-01-01",
    end="2024-12-31",
    auto_adjust=False  # ÖNEMLİ
)

# Çoklu kolonlardan sadece 'Adj Close' seviyesini al
data = raw_data["Adj Close"]

# 3) Günlük log-return hesaplama
returns = data.pct_change().dropna()
log_returns = (1 + returns).apply(np.log)

# 4) Kaydet
raw_data.to_csv("data/raw/nasdaq_full_download.csv")
data.to_csv("data/raw/nasdaq_adj_close.csv")
log_returns.to_csv("data/processed/nasdaq_log_returns.csv")

print("İşlem tamam! CSV dosyaları kaydedildi 🎉")

