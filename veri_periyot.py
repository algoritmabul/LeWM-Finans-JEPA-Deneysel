#veri_periyot.py
import pandas as pd
import numpy as np
import os
from datetime import datetime
from pathlib import Path

class VeriPeriyot:
    """60 dk ve Günlük CSV'lerden istenen periyotta veri sağlar"""
    
    def __init__(self, 
                 csv_60dk_dizini=r'D:\new\yeniden\metastock-60dk-csv',
                 csv_gunluk_dizini=r'D:\new\yeniden\metastock-gun-csv'):
        self.csv_60dk_dizini = Path(csv_60dk_dizini)
        self.csv_gunluk_dizini = Path(csv_gunluk_dizini)
        self._cache_60dk = {}
        self._cache_gunluk = {}
        self._cache_periyot = {}
        
        # Hangi periyotlar hangi kaynaktan?
        self.kaynak_map = {
            '60dk': '60dk',
            '240dk': '60dk',
            'gunluk': 'gunluk',
            'haftalik': 'gunluk',
            'aylik': 'gunluk'
        }
    
    def _oku_60dk(self, hisse):
        """60 dk CSV'yi oku"""
        hisse = hisse.upper()
        
        if hisse in self._cache_60dk:
            return self._cache_60dk[hisse]
        
        dosya = self.csv_60dk_dizini / f"{hisse}.csv"
        if not dosya.exists():
            raise FileNotFoundError(f"{hisse}.csv (60dk) bulunamadı")
        
        df = pd.read_csv(dosya)
        df['DATETIME'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
        df = df.sort_values('DATETIME').set_index('DATETIME')
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        df = df.dropna(subset=['Open', 'High', 'Low', 'Close'])
        
        mask = df['High'] < df['Low']
        df.loc[mask, ['High', 'Low']] = df.loc[mask, ['Low', 'High']].values
        
        self._cache_60dk[hisse] = df
        return df
    
    def _oku_gunluk(self, hisse):
        """Günlük CSV'yi oku (2002'den beri)"""
        hisse = hisse.upper()
        
        if hisse in self._cache_gunluk:
            return self._cache_gunluk[hisse]
        
        dosya = self.csv_gunluk_dizini / f"{hisse}.csv"
        if not dosya.exists():
            raise FileNotFoundError(f"{hisse}.csv (günlük) bulunamadı")
        
        df = pd.read_csv(dosya)
        
        # Günlük CSV formatı: Date, Open, High, Low, Close, Volume
        if 'Date' in df.columns:
            df['DATETIME'] = pd.to_datetime(df['Date'])
        else:
            raise ValueError("Günlük CSV'de 'Date' kolonu bulunamadı")
        
        df = df.sort_values('DATETIME').set_index('DATETIME')
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        df = df.dropna(subset=['Open', 'High', 'Low', 'Close'])
        
        mask = df['High'] < df['Low']
        df.loc[mask, ['High', 'Low']] = df.loc[mask, ['Low', 'High']].values
        
        self._cache_gunluk[hisse] = df
        return df
    
    def _periyoda_cevir(self, df_kaynak, periyot, kaynak_tipi):
        """Veriyi istenen periyoda çevir"""
        
        if periyot == '60dk':
            return df_kaynak
        
        # 240 dk için özel (sadece 60dk kaynaktan)
        if periyot == '240dk' and kaynak_tipi == '60dk':
            df = df_kaynak.copy()
            df['Saat'] = df.index.hour
            df['Gun'] = df.index.date
            
            sonuclar = []
            for gun, grup in df.groupby('Gun'):
                # 1. BLOK: 10:00-14:00 (10,11,12,13)
                dilim1 = grup[(grup['Saat'] >= 10) & (grup['Saat'] < 14)]
                if len(dilim1) == 4:
                    bar = {
                        'Open': dilim1['Open'].iloc[0],
                        'High': dilim1['High'].max(),
                        'Low': dilim1['Low'].min(),
                        'Close': dilim1['Close'].iloc[-1],
                        'Volume': dilim1['Volume'].sum()
                    }
                    sonuclar.append((pd.Timestamp(f"{gun} 10:00:00"), bar))
                
                # 2. BLOK: 14:00-18:00 (14,15,16,17,18)
                dilim2 = grup[(grup['Saat'] >= 14) & (grup['Saat'] <= 18)]
                if len(dilim2) == 5:
                    bar = {
                        'Open': dilim2['Open'].iloc[0],
                        'High': dilim2['High'].max(),
                        'Low': dilim2['Low'].min(),
                        'Close': dilim2['Close'].iloc[-1],
                        'Volume': dilim2['Volume'].sum()
                    }
                    sonuclar.append((pd.Timestamp(f"{gun} 14:00:00"), bar))
            
            if sonuclar:
                df_240 = pd.DataFrame([s[1] for s in sonuclar], index=[s[0] for s in sonuclar])
                return df_240.sort_index()
            return pd.DataFrame()
        
        # Günlük, Haftalık, Aylık için resample
        periyot_map = {
            'gunluk': 'D',
            'haftalik': 'W',
            'aylik': 'ME'
        }
        
        if periyot not in periyot_map:
            raise ValueError(f"Geçersiz periyot: {periyot}")
        
        freq = periyot_map[periyot]
        df = df_kaynak.resample(freq).agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()
        
        return df
    
    def get(self, hisse, periyot='60dk'):
        """İstenen periyotta OHLCV verisi döndürür. CSV yoksa None döner."""
        hisse = hisse.upper()
        periyot = periyot.lower()

        cache_key = f"{hisse}_{periyot}"
        if cache_key in self._cache_periyot:
            return self._cache_periyot[cache_key].copy()

        # Hangi kaynağı kullanacağını belirle
        kaynak_tipi = self.kaynak_map.get(periyot, '60dk')

        try:
            if kaynak_tipi == '60dk':
                df_kaynak = self._oku_60dk(hisse)
            else:
                try:
                    df_kaynak = self._oku_gunluk(hisse)
                except FileNotFoundError:
                    # Gunluk CSV yoksa 60dk'dan dene
                    print(f"   [WARN] {hisse} icin gunluk veri yok, 60dk'dan uretiliyor...")
                    df_kaynak = self._oku_60dk(hisse)
                    kaynak_tipi = '60dk'
        except FileNotFoundError:
            print(f"   [WARN] {hisse} icin veri dosyasi yok; atlaniyor.")
            return None

        df = self._periyoda_cevir(df_kaynak, periyot, kaynak_tipi)
        self._cache_periyot[cache_key] = df
        return df.copy()
    
    def periyotlari_goster(self, hisse):
        """Tüm periyotlardaki bar sayılarını göster"""
        print(f"\n📊 {hisse.upper()} Periyot Analizi")
        print("="*60)
        
        try:
            df_60 = self.get(hisse, '60dk')
            print(f"60 dk   : {len(df_60):6} bar (2022'den itibaren)")
        except:
            print(f"60 dk   : {'':6} yok")
        
        try:
            df_240 = self.get(hisse, '240dk')
            print(f"240 dk  : {len(df_240):6} bar (60dk'dan üretildi)")
        except:
            print(f"240 dk  : {'':6} yok")
        
        try:
            df_gun = self.get(hisse, 'gunluk')
            print(f"Günlük  : {len(df_gun):6} bar (2002'den itibaren)")
        except:
            print(f"Günlük  : {'':6} yok")
        
        try:
            df_haf = self.get(hisse, 'haftalik')
            print(f"Haftalık: {len(df_haf):6} bar")
        except:
            print(f"Haftalık: {'':6} yok")
        
        try:
            df_ay = self.get(hisse, 'aylik')
            print(f"Aylık   : {len(df_ay):6} bar")
        except:
            print(f"Aylık   : {'':6} yok")


# ========== TEST ==========
if __name__ == "__main__":
    
    vp = VeriPeriyot(
        csv_60dk_dizini=r'D:\new\yeniden\metastock-60dk-csv',
        csv_gunluk_dizini=r'D:\new\yeniden\metastock-gun-csv'
    )
    
    # EDATA için periyotları göster
    vp.periyotlari_goster('EDATA')
    
    # Günlük veriyi kontrol et
    print("\n" + "="*60)
    print("📊 GÜNLÜK VERİ (2002'den itibaren)")
    print("="*60)
    
    try:
        df_gun = vp.get('thyao', 'gunluk')
        print(f"Toplam bar: {len(df_gun)}")
        print(f"İlk tarih: {df_gun.index[0]}")
        print(f"Son tarih: {df_gun.index[-1]}")
        print("\nSon 5 gün:")
        print(df_gun[['Open', 'High', 'Low', 'Close']].tail(5))
    except Exception as e:
        print(f"Hata: {e}")
