import pandas as pd
import re
import numpy as np

def clean_market_data(raw_text):
    """
    Karışık finansal veri metnini temizleyip Pandas DataFrame'e çevirir.
    """
    if not raw_text:
        return pd.DataFrame()

    # Gereksiz başlıkları temizle
    text = raw_text.replace("ParityT+F", "")
    
    # Rakamla biten endeks isimlerini korumaya al (örn: US30 -> US30|)
    known_indices = ["US30", "NAS100", "E DJI", "Em NQ", "NIKKEI JPN IND", "S&P500", "DAX", "^GDAXI"]
    for index in known_indices:
        text = text.replace(index, f"{index}|")

    # Regex ile Ayıklama
    # Mantık: İsim kısmı | Değer kısmı
    pattern = r"([A-Za-z0-9/\s\(\)\.-]+?)(?:\|)?(-?\d+\.\d+|-?\d+|(?<=[A-Za-z])-|$)((?=[A-Z])|$)"
    matches = re.findall(pattern, text)

    # Regex çıktısını temizle
    clean_matches = [[m[0].strip(), m[1]] for m in matches]

    # DataFrame Oluşturma
    df = pd.DataFrame(clean_matches, columns=["Enstrüman", "Değer"])

    # Sayısal Temizlik
    # '-' veya boşluk olanları NaN yap, gerisini sayıya çevir
    df["Değer"] = df["Değer"].replace(["-", ""], np.nan)
    df["Değer"] = pd.to_numeric(df["Değer"], errors='coerce')

    return df