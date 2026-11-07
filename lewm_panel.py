# -*- coding: utf-8 -*-
"""
lewm_panel.py — LEWM-Finans TEK-DOSYA sonuç paneli
==================================================
Bir komutla: çıktı klasörlerini tarar → bilinen CSV'leri toplar → özet
kartları, düz-Türkçe anlatı, komut rehberi ve sözlüğü içeren TEK dosyalık
HTML paneli üretir (lewm_panel.html).

  * İnternet gerekmez — dosya offline açılır.
  * Paylaşılabilir — HTML'i olduğu gibi gönderebilirsiniz; alıcı kendi
    CSV'sini sürükleyip bırakarak paneli kendi verisiyle kullanır.
  * Aynı CSV birden çok klasördeyse EN YENİSİ kullanılır (sürüm karışmaz).
  * Koyu/açık tema: başlıktaki 🌙/☀️ düğmesi; seçim tarayıcıda hatırlanır.

Kullanım:
  python lewm_panel.py                                    # varsayılan klasörler
  python lewm_panel.py --dir lewm_cikti --dir lewm_senaryo_cikti
  python lewm_panel.py --out rapor.html --ac              # üretince tarayıcıda aç

Varsayılan taranan klasörler (varsa): lewm_cikti, lewm_senaryo_cikti,
lewm_al_cikti, lewm_radar_cikti + içinde bulunulursa "." klasörü.
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
import webbrowser
from pathlib import Path

import numpy as np
import pandas as pd


def _utf8():
    """Windows konsolunda Türkçe çıktı güvenliği."""
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


# --- bilinen çıktı kayıt defteri ---------------------------------------------
# (key, dosya adı, sekme adı, üreten komut)
KAYIT = [
    ("al", "al_listesi.csv", "AL Listesi",
     "python lewm_al_listesi.py --piyasa-onay"),
    ("dogrulama", "al_dogrulama.csv", "Doğrulama",
     "python lewm_al_listesi.py  (doğrulama bölümü otomatik gelir)"),
    ("hindsight", "al_degerlendirme.csv", "Hindsight",
     "python lewm_al_listesi.py --son-tarih 2026-10-16 --kontrol-tarih 2026-10-23"),
    ("senaryo", "senaryo_tarama.csv", "Senaryo",
     "python lewm_cli.py uygula --tum --senaryo-tarama --model lewm_cikti\\lewm_model_eniyi.pt --cikti lewm_senaryo_cikti"),
    ("surpriz", "surprise_tarama.csv", "Sürpriz",
     "python lewm_cli.py uygula --tum --model lewm_cikti\\lewm_model_eniyi.pt --cikti lewm_cikti"),
    ("radar", "piyasa_radari.csv", "Radar",
     "python lewm_piyasa_radari.py"),
]

INDEKSLI = ("dogrulama", "radar")   # bu CSV'lerde 1. kolon tarih indeksidir


def _tablo_okuma(key: str, yol: Path) -> pd.DataFrame:
    """CSV'yi standart şemaya getirir (tarih kolonu, Türkçe kolon adları korunur)."""
    indeksli = key in INDEKSLI
    df = pd.read_csv(yol, encoding="utf-8-sig",
                     index_col=0 if indeksli else None)
    if indeksli:
        df = df.reset_index()
        ilk = df.columns[0]
        if ilk not in ("tarih", "Date", "date"):
            df = df.rename(columns={ilk: "tarih"})
        dt = pd.to_datetime(df["tarih"], errors="coerce")
        df = df[dt.notna()].copy()
        df["tarih"] = dt[dt.notna()].dt.strftime("%Y-%m-%d")
    if "baglam_sonu" in df.columns:            # senaryo taraması
        dt = pd.to_datetime(df["baglam_sonu"], errors="coerce")
        df["baglam_sonu"] = dt.dt.strftime("%Y-%m-%d")
    return df


def _kayitlar(df: pd.DataFrame, yuvarla: int = 2) -> list:
    """DataFrame → JSON güvenli kayıt listesi (NaN→None, bool→bool)."""
    kayitlar = []
    for _, r in df.iterrows():
        satir = {}
        for c in df.columns:
            v = r[c]
            try:
                bos = v is None or pd.isna(v)
            except Exception:
                bos = False
            if bos:
                satir[c] = None
            elif isinstance(v, (bool, np.bool_)):
                satir[c] = bool(v)
            elif isinstance(v, (int, np.integer)):
                satir[c] = int(v)
            elif isinstance(v, (float, np.floating)):
                satir[c] = round(float(v), yuvarla)
            else:
                satir[c] = str(v)
        kayitlar.append(satir)
    return kayitlar


def veriyi_topla(dizinler: list) -> tuple:
    """Klasörleri tarar; her bilinen CSV için EN YENİ kopyayı seçer.

    Döner: (veri{key: kayıtlar}, dosya_bilgi[...], cift[...])
    """
    aday = {}
    for d in dizinler:
        for key, ad, _, _ in KAYIT:
            p = Path(d) / ad
            if p.exists():
                aday.setdefault(key, []).append(p)

    secilen, cift = {}, []
    for key, lst in aday.items():
        lst.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        secilen[key] = lst[0]
        if len(lst) > 1:
            cift.append({"key": key, "yollar": [str(p) for p in lst]})

    veri, dosya_bilgi = {}, []
    for key, ad, sekme, ureten in KAYIT:
        if key not in secilen:
            dosya_bilgi.append({"key": key, "sekme": sekme, "ureten": ureten,
                                "bulundu": False})
            continue
        bilgi = {"key": key, "sekme": sekme, "ureten": ureten, "bulundu": True,
                 "dosya": str(secilen[key]),
                 "guncelleme": datetime.datetime.fromtimestamp(
                     secilen[key].stat().st_mtime).strftime("%d.%m.%Y %H:%M")}
        try:
            veri[key] = _kayitlar(_tablo_okuma(key, secilen[key]))
            bilgi["satir"] = len(veri[key])
        except Exception as e:
            bilgi["hata"] = f"{type(e).__name__}: {e}"
        dosya_bilgi.append(bilgi)
    return veri, dosya_bilgi, cift


# --- özet kartları + düz-Türkçe anlatı ---------------------------------------
def _medyan(x: list):
    return float(np.median(x)) if x else None


def _rejim_bilgisi(veri: dict) -> dict:
    """Rejimi mümkünse radar'dan, yoksa doğrulamanın son satırından okur."""
    if veri.get("radar"):
        son = veri["radar"][-1]
        return {"ad": (son.get("gz_rejim") or son.get("rejim") or "?"),
                "neden": son.get("gz_neden") or "", "tarih": son.get("tarih", ""),
                "kaynak": "piyasa_radari.csv"}
    if veri.get("dogrulama"):
        son = veri["dogrulama"][-1]
        return {"ad": (son.get("gz_rejim") or "?"), "neden": "",
                "tarih": son.get("tarih", ""), "kaynak": "al_dogrulama.csv"}
    return None


def ozet_uret(veri: dict) -> dict:
    oz = {"kesit": "", "rejim": _rejim_bilgisi(veri), "al": None,
          "dogrulama": None, "senaryo": None, "surpriz": None,
          "anlati": [], "tavsiye": "", "tavsiye_seviye": "bilgi"}

    # kesit tarihi = eldeki en güncel okuma
    tarihler = []
    if veri.get("radar"):
        tarihler.append(veri["radar"][-1].get("tarih"))
    if veri.get("dogrulama"):
        tarihler.append(veri["dogrulama"][-1].get("tarih"))
    if veri.get("senaryo"):
        tarihler += [r.get("baglam_sonu") for r in veri["senaryo"]]
    tarihler = [t for t in tarihler if t]
    oz["kesit"] = max(tarihler) if tarihler else ""

    # --- AL listesi -----------------------------------------------------------
    if veri.get("al") is not None:
        L = veri["al"]
        guclu = [r for r in L if r.get("kategori") == "GÜÇLÜ AL"]
        oz["al"] = {"n": len(L), "guclu": len(guclu),
                    "top": (L[0]["hisse"] if L else ""), "top_skor": (L[0].get("skor") if L else None)}

    # --- doğrulama ------------------------------------------------------------
    if veri.get("dogrulama"):
        olculebilir = [r for r in veri["dogrulama"] if r.get("fark%") is not None]
        bekleyen = len(veri["dogrulama"]) - len(olculebilir)
        if olculebilir:
            farklar = [r["fark%"] for r in olculebilir]
            alarm = [r["fark%"] for r in olculebilir if r.get("gz_rejim") == "ALARM"]
            poz = sum(1 for f in farklar if f > 0)
            oz["dogrulama"] = {
                "n_gun": len(olculebilir), "bekleyen": bekleyen,
                "medyan_fark": round(_medyan(farklar), 2),
                "pozitif_gun_%": round(poz / len(farklar) * 100),
                "alarm_gun": len(alarm),
                "alarm_fark": round(_medyan(alarm), 2) if alarm else None}

    # --- senaryo --------------------------------------------------------------
    if veri.get("senaryo"):
        S = veri["senaryo"]
        pozitif = [r for r in S if (r.get("medyan_senaryo_getiri_%") or 0) > 0]
        p10lar = [r for r in S if r.get("p10_%") is not None]
        en_kotu = min(p10lar, key=lambda r: r["p10_%"]) if p10lar else None
        oz["senaryo"] = {"n": len(S), "pozitif": len(pozitif),
                         "top": (S[0]["hisse"] if S else ""),
                         "top_medyan": (S[0].get("medyan_senaryo_getiri_%") if S else None),
                         "kotu_p10_hisse": (en_kotu["hisse"] if en_kotu else ""),
                         "kotu_p10": (en_kotu["p10_%"] if en_kotu else None)}

    # --- sürpriz --------------------------------------------------------------
    if veri.get("surpriz"):
        P = [r for r in veri["surpriz"] if r.get("sigma_skoru") is not None]
        alarm = [r for r in P if abs(r["sigma_skoru"]) > 3]
        oz["surpriz"] = {"n": len(P), "alarm": len(alarm),
                         "top": [(r["hisse"], r["sigma_skoru"]) for r in alarm[:3]]}

    # --- düz-Türkçe anlatı ----------------------------------------------------
    an = oz["anlati"]
    if oz["kesit"]:
        an.append(f"Veri kesiti: {oz['kesit']} kapanışı.")
    if oz["rejim"]:
        n = f"Piyasa rejimi: {oz['rejim']['ad']}"
        if oz["rejim"]["neden"]:
            n += f" ({oz['rejim']['neden']})"
        n += f" — kaynak {oz['rejim']['kaynak']}."
        an.append(n)
    if oz["al"]:
        a = oz["al"]
        an.append(f"AL listesi: {a['n']} hisse ({a['guclu']} GÜÇLÜ AL)"
                  + (f", başta {a['top']}" if a["top"] else "") + ".")
    d = oz["dogrulama"]
    if d and d["n_gun"]:
        n = (f"Doğrulama ({d['n_gun']} ölçülebilir gün): liste, piyasa medyanından "
             f"{d['medyan_fark']:+.2f} pt; günlerin %{d['pozitif_gun_%']}'inde liste önde.")
        if d["alarm_gun"]:
            n += f" Piyasa ALARM'dayken fark {d['alarm_fark']:+.2f} pt ({d['alarm_gun']} gün)."
        an.append(n)
    s = oz["senaryo"]
    if s and s["n"]:
        n = (f"Senaryo taraması: {s['n']} hissede {s['pozitif']} tanesinin medyan "
             f"senaryo getirisi pozitif; en yüksek {s['top']} "
             f"({(s['top_medyan'] or 0):+.2f}%).")
        if s["kotu_p10"] is not None:
            n += f" En derin kuyruk {s['kotu_p10_hisse']} (p10 {s['kotu_p10']:+.2f}%)."
        an.append(n)
    p = oz["surpriz"]
    if p and p["n"]:
        n = f"Sürpriz taraması: {p['n']} hissede {p['alarm']} tanesi ±3σ dışında."
        if p["top"]:
            n += " En olağandışı: " + ", ".join(
                f"{h} ({v:+.1f}σ)" for h, v in p["top"]) + "."
        an.append(n)

    # --- tavsiye kutusu -------------------------------------------------------
    rejim_ad = oz["rejim"]["ad"] if oz["rejim"] else "BİLİNMEYEN"
    if rejim_ad == "ALARM":
        oz["tavsiye"] = ("Piyasa ALARM'da. --piyasa-onay ile AL listesi üretilmez; "
                         "yeni alım için bekle. Açık pozisyonlarda gzσ ve Radar "
                         "sekmelerini izle. Senaryo taraması keşif amaçlıdır, işlem "
                         "kararı üretmez.")
        oz["tavsiye_seviye"] = "alarm"
    elif rejim_ad == "TEDBİR":
        oz["tavsiye"] = ("Piyasa TEDBİR'de. Liste üretilebilir ama riskli bölge: "
                         "pozisyon boyutunu küçült, yalnız skor 6 (GÜÇLÜ AL) "
                         "hisselere ve p10 sığ (kuyruk derin olmayan) senaryolara bak.")
        oz["tavsiye_seviye"] = "tedbir"
    else:
        oz["tavsiye"] = ("Piyasa NORMAL. AL listesi güvenli bölgede üretildi: skor≥5 "
                         "hisselere bak; senaryo sekmesinde p10 derin olmayan ve "
                         "yükselen % yüksek adlar öne çıkar. Listeyi ertesi gün "
                         "kapanışta hindsight ile ölçmeyi unutma.")
        oz["tavsiye_seviye"] = "normal"
    return oz


def _json_gom(s: str) -> str:
    """JSON'u <script> içine güvenle gömer (</script> kırılması olmasın)."""
    return s.replace("</", "<\\/")


def ana():
    _utf8()
    p = argparse.ArgumentParser(
        description="LEWM-Finans tek-dosya sonuç paneli üretici",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--dir", action="append", default=None, metavar="KLASOR",
                   help="taracak çıktı klasörü (birden çok yazılabilir)")
    p.add_argument("--out", default="lewm_panel.html", help="panel HTML yolu")
    p.add_argument("--ac", action="store_true", help="üretince tarayıcıda aç")
    a = p.parse_args()

    varsayilan = ["lewm_cikti", "lewm_senaryo_cikti", "lewm_al_cikti",
                  "lewm_radar_cikti", "."]
    dizinler = a.dir or [d for d in varsayilan if Path(d).exists()]
    if not dizinler:
        sys.exit("[HATA] çıktı klasörü bulunamadı. --dir ile belirtin, örn:\n"
                 "  python lewm_panel.py --dir lewm_cikti --dir lewm_senaryo_cikti")

    veri, dosya_bilgi, cift = veriyi_topla(dizinler)
    if not veri:
        eksik = "\n".join(f"  - {b['sekme']:<11} ← {b['ureten']}"
                          for b in dosya_bilgi)
        sys.exit("[HATA] bilinen CSV bulunamadı. Panel şunları arar:\n"
                 + eksik)
    ozet = ozet_uret(veri)
    meta = {"uretim": datetime.datetime.now().strftime("%d.%m.%Y %H:%M"),
            "dizinler": [str(Path(d).resolve()) for d in dizinler],
            "dosyalar": dosya_bilgi, "cift": cift}

    html = (SABLON
            .replace("__VERI_JSON__", _json_gom(json.dumps(veri, ensure_ascii=False)))
            .replace("__OZET_JSON__", _json_gom(json.dumps(ozet, ensure_ascii=False)))
            .replace("__META_JSON__", _json_gom(json.dumps(meta, ensure_ascii=False))))
    out = Path(a.out)
    out.write_text(html, encoding="utf-8")
    kb = out.stat().st_size / 1024

    print("=" * 64)
    print("LeWM-Finans PANEL")
    print("=" * 64)
    print(f"[1/2] taranan klasörler : {', '.join(dizinler)}")
    for b in dosya_bilgi:
        if b.get("hata"):
            print(f"   [HATA] {b['sekme']}: {b['hata']}")
        elif b.get("bulundu"):
            print(f"   [OK]   {b['sekme']:<11} {b['satir']:>5} satır"
                  f"  ({b['guncelleme']})")
        else:
            print(f"   [YOK]  {b['sekme']:<11} ← {b['ureten']}")
    for c in cift:
        print(f"   [BİLDİR] {c['key']}: birden çok kopya bulundu, en yenisi seçildi:")
        for y in c["yollar"]:
            print(f"            {y}")
    print(f"[2/2] panel yazıldı     : {out.resolve()} ({kb:.0f} KB)")
    if ozet["kesit"]:
        print(f"       kesit            : {ozet['kesit']}")
    if ozet["rejim"]:
        print(f"       rejim            : {ozet['rejim']['ad']}"
              + (f" ({ozet['rejim']['neden']})" if ozet["rejim"]["neden"] else ""))
    print("       paneli tarayıcıda açın; CSV'yi pencereye sürükleyip "
          "bölümleri güncelleyebilirsiniz.")
    if a.ac:
        webbrowser.open(out.resolve().as_uri())


SABLON = r"""<!doctype html>
<html lang="tr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LeWM-Finans Panel</title>
<style>
:root{
  --bg:#f4f6f8; --kart:#ffffff; --metin:#1c2733; --soluk:#65758a;
  --cizgi:#e3e8ee; --mavi:#2563eb; --yesil:#15803d; --kirmizi:#dc2626;
  --amber:#b45309; --koyu:#0f172a;
  color-scheme:light;
}
:root.koyu{
  --bg:#0b1220; --kart:#151e2e; --metin:#e2e8f0; --soluk:#94a3b8;
  --cizgi:#2a3650; --mavi:#60a5fa; --yesil:#4ade80; --kirmizi:#f87171;
  --amber:#fbbf24; --koyu:#0f172a;
  color-scheme:dark;
}
:root{--kpitabani:#fbfcfe; --thtabani:#f1f4f8; --hover:#f8fafc; --dd:#3b4a5e;
  --nokta:#c3ccd8; --izge:#2563eb; --ipucu:#9aa8ba; --kutucizgi:#b8c4d4;
  --daga:#eef1f5; --sifir:#94a3b8; --ort:#0f172a; --golge:rgba(16,24,40,.04);
  --rzn-b:#dcfce7; --rzb-b:#fef3c7; --rlm-b:#fee2e2;
  --tav-b:#eff6ff; --tav-a:#fef2f2; --tav-t:#fffbeb;
  --komutb:#0f172a; --komutm:#e2e8f0; --komutbtn:#334155; --komutbtnh:#475569;
  --basalt:#9fb0c3}
:root.koyu{--kpitabani:#182236; --thtabani:#1c2740; --hover:#1b2537; --dd:#b6c2d6;
  --nokta:#44536f; --izge:#60a5fa; --ipucu:#64748b; --kutucizgi:#3b4a63;
  --daga:#1c2740; --sifir:#64748b; --ort:#e2e8f0; --golge:rgba(0,0,0,.35);
  --rzn-b:#12331f; --rzb-b:#332608; --rlm-b:#3a1518;
  --tav-b:#14263f; --tav-a:#3a1518; --tav-t:#332608;
  --komutb:#0a101f; --komutm:#dbe4f0; --komutbtn:#28344b; --komutbtnh:#33415e;
  --basalt:#9fb0c3}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--metin);
  font:14px/1.45 -apple-system,"Segoe UI",Roboto,Arial,sans-serif}
header{background:var(--koyu);color:#fff;padding:14px 22px}
header h1{margin:0;font-size:19px;letter-spacing:.3px}
header .alt{color:var(--basalt);font-size:12.5px;margin-top:3px}
header .yenile{float:right;margin-top:2px}
main{max-width:1240px;margin:0 auto;padding:18px 22px 60px}
nav.sekmeler{display:flex;flex-wrap:wrap;gap:6px;margin:14px 0 18px}
nav.sekmeler button{border:1px solid var(--cizgi);background:var(--kart);color:var(--metin);
  padding:7px 14px;border-radius:8px;cursor:pointer;font-size:13.5px}
nav.sekmeler button.aktif{background:var(--mavi);border-color:var(--mavi);color:#fff}
.kart{background:var(--kart);border:1px solid var(--cizgi);border-radius:12px;
  padding:16px 18px;margin-bottom:16px;box-shadow:0 1px 2px var(--golge)}
.kart h2{margin:0 0 10px;font-size:15.5px}
.kart h2 .soluk{font-weight:400;font-size:12.5px}
.kpi-satir{display:flex;flex-wrap:wrap;gap:12px}
.kpi{flex:1 1 170px;border:1px solid var(--cizgi);border-radius:10px;padding:11px 13px;background:var(--kpitabani)}
.kpi .deger{font-size:23px;font-weight:700}
.kpi .etiket{color:var(--soluk);font-size:12px;margin-top:2px}
.kpi .ek{font-size:12px;color:var(--soluk);margin-top:4px}
.rozet{display:inline-block;padding:2px 10px;border-radius:99px;font-size:12px;font-weight:600}
.rozet.NORMAL{background:var(--rzn-b);color:var(--yesil)}
.rozet.TEDBİR,.rozet.TEDBIR{background:var(--rzb-b);color:var(--amber)}
.rozet.ALARM{background:var(--rlm-b);color:var(--kirmizi)}
.tavsiye{border-left:5px solid var(--mavi);background:var(--tav-b);border-radius:8px;padding:11px 14px}
.tavsiye.alarm{border-color:var(--kirmizi);background:var(--tav-a)}
.tavsiye.tedbir{border-color:var(--amber);background:var(--tav-t)}
.tavsiye b{display:block;margin-bottom:3px}
ul.anlati{margin:6px 0 0;padding-left:20px}
ul.anlati li{margin:4px 0}
table{border-collapse:collapse;width:100%;font-size:13px}
th{position:sticky;top:0;background:var(--thtabani);text-align:right;padding:7px 8px;
  border-bottom:2px solid var(--cizgi);white-space:nowrap;cursor:pointer;user-select:none}
th.soltaraf{text-align:left}
td{padding:6px 8px;border-bottom:1px solid var(--cizgi);text-align:right;
  white-space:nowrap;font-variant-numeric:tabular-nums}
td.soltaraf{text-align:left}
tr:hover td{background:var(--hover)}
.poz{color:var(--yesil);font-weight:600}
.neg{color:var(--kirmizi);font-weight:600}
.nokta{color:var(--nokta);font-weight:700}
.daga{position:relative;height:14px;background:var(--daga);border-radius:4px;min-width:120px}
.daga .sifir{position:absolute;top:-3px;bottom:-3px;width:2px;background:var(--sifir)}
.daga .bant{position:absolute;top:3px;bottom:3px;border-radius:3px;
  background:linear-gradient(90deg,#fecaca,#fde68a 45%,#bbf7d0)}
.daga .ort{position:absolute;top:-2px;width:3px;bottom:-2px;background:var(--ort);border-radius:2px}
.kucuk{font-size:12px;color:var(--soluk)}
.ipucu{cursor:help;border-bottom:1px dotted var(--ipucu)}
.komut{display:flex;align-items:flex-start;gap:8px;background:var(--komutb);color:var(--komutm);
  border-radius:8px;padding:9px 12px;font:12.5px/1.5 Consolas,monospace;margin:6px 0;
  overflow-x:auto;white-space:pre}
.komut button{flex:0 0 auto;background:var(--komutbtn);border:none;color:var(--komutm);border-radius:6px;
  padding:4px 9px;cursor:pointer;font-size:11.5px}
.komut button:hover{background:var(--komutbtnh)}
.adim{display:flex;gap:12px;margin:14px 0}
.adim .no{flex:0 0 26px;height:26px;border-radius:99px;background:var(--mavi);color:#fff;
  display:flex;align-items:center;justify-content:center;font-weight:700;font-size:13px}
.adim .govde{flex:1}
.adim .govde .acik{color:var(--soluk);font-size:12.5px;margin:2px 0 4px}
dl.sozluk{margin:0}
dl.sozluk dt{font-weight:700;margin-top:11px}
dl.sozluk dd{margin:2px 0 0 0;color:var(--dd)}
.kutu{border:1px dashed var(--kutucizgi);border-radius:10px;padding:18px;text-align:center;
  color:var(--soluk);margin:10px 0}
.toast{position:fixed;bottom:18px;right:18px;background:var(--koyu);color:#fff;
  padding:10px 16px;border-radius:10px;opacity:0;transition:.25s;pointer-events:none;
  max-width:420px;font-size:13px}
.toast.gorunur{opacity:.96}
.surukle-ortu{position:fixed;inset:0;background:rgba(37,99,235,.12);border:4px dashed var(--mavi);
  display:none;align-items:center;justify-content:center;font-size:22px;font-weight:700;
  color:var(--mavi);z-index:99;pointer-events:none}
.surukle-ortu.gorunur{display:flex}
.gizli{display:none!important}
.buton{border:1px solid var(--cizgi);background:var(--kart);border-radius:8px;padding:6px 12px;
  cursor:pointer;font-size:12.5px}
.buton.ana{background:var(--mavi);border-color:var(--mavi);color:#fff}
.arama{border:1px solid var(--cizgi);border-radius:8px;padding:6px 10px;font-size:13px;width:220px}
.bolum-baslik{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:10px}
@media print{nav.sekmeler{display:none}.bolum{display:block!important;page-break-after:always}
  :root.koyu{--bg:#fff;--kart:#fff;--metin:#1c2733;--soluk:#65758a;--cizgi:#e3e8ee;
    --kpitabani:#fbfcfe;--thtabani:#f1f4f8;--hover:#f8fafc;--dd:#3b4a5e;--ort:#0f172a}}
</style>
</head>
<body>
<div class="surukle-ortu" id="surukleOrtu">CSV'yi bırak — bölüm otomatik güncellenir</div>
<header>
  <span class="yenile"><button class="buton" id="temaBtn" onclick="temaDegistir()" title="Koyu/açık tema">🌙 Koyu</button>
  <button class="buton" onclick="document.getElementById('csvDosya').click()">📁 CSV Yükle</button>
  <button class="buton" onclick="window.print()">🖨 Yazdır</button></span>
  <h1>LeWM-Finans Panel</h1>
  <div class="alt" id="ustAlt">…</div>
</header>
<input type="file" id="csvDosya" accept=".csv" multiple class="gizli">
<main>
<nav class="sekmeler" id="sekmeCubuk"></nav>
<div id="icerik"></div>
</main>
<div class="toast" id="toast"></div>
<script>
const VERI = __VERI_JSON__;
const OZET = __OZET_JSON__;
const META = __META_JSON__;
</script>
<script>
/* ================= yardımcılar ================= */
const $ = q => document.querySelector(q);
const ESC = s => String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
function toast(m){const t=$('#toast');t.textContent=m;t.classList.add('gorunur');
  clearTimeout(t._z);t._z=setTimeout(()=>t.classList.remove('gorunur'),3400);}
function medyan(d){if(!d.length)return null;const a=[...d].sort((x,y)=>x-y);
  const m=Math.floor(a.length/2);return a.length%2?a[m]:(a[m-1]+a[m])/2;}

function hucre(v,tip,hass){
  if(v===null||v===undefined||v==='')return '<span class="nokta">—</span>';
  if(tip==='metin')return ESC(v);
  if(tip==='bool')return v?'<span style="color:var(--yesil);font-weight:700">✓</span>':'<span class="nokta">·</span>';
  if(tip==='sayi'||tip==='sayi0'||tip==='sayi1'||tip==='sayi2'||tip==='sayi4'){
    const h={sayi:0,sayi0:0,sayi1:1,sayi2:2,sayi4:4}[tip];
    return typeof v==='number'?v.toLocaleString('tr-TR',{minimumFractionDigits:h,maximumFractionDigits:h}):ESC(v);}
  if(tip==='getiri1'||tip==='getiri2'){
    const n=typeof v==='number'?v:parseFloat(v);if(isNaN(n))return ESC(v);
    const h=tip==='getiri1'?1:2;const s=n.toFixed(h);
    return '<span class="'+(n>0?'poz':n<0?'neg':'')+'">'+(n>0?'+':'')+s+'</span>';}
  if(tip==='sigma'){
    const n=typeof v==='number'?v:parseFloat(v);if(isNaN(n))return ESC(v);
    const st=Math.abs(n)>3?'color:var(--kirmizi);font-weight:700':(n>1?'color:var(--amber);font-weight:600':'');
    const s=(n>0?'+':'')+n.toFixed(2);
    return st?'<span style="'+st+'">'+s+'</span>':s;}
  if(tip==='rozet_rejim'){
    const ad=String(v).replace('İ','İ');
    const cls=['NORMAL','TEDBİR','TEDBIR','ALARM'].includes(ad)?ad:'';
    return cls?'<span class="rozet '+cls+'">'+ESC(v)+'</span>':ESC(v);}
  if(tip==='kategori'){
    const renk=v==='GÜÇLÜ AL'?'background:#dcfce7;color:#15803d':v==='AL'?'background:#e0f2fe;color:#0369a1':
      v==='İZLE'?'background:#fef3c7;color:#b45309':'background:#f1f5f9;color:#64748b';
    return '<span class="rozet" style="'+renk+'">'+ESC(v)+'</span>';}
  return ESC(v);
}

/* ================= kolon şemaları ================= */
const KOSUL_ACIKLAMA={
 K1_model:'K1 — model risk filtresi: gzσ ≤ +1 ve |gzσ| ≤ 3 (anomali yok). Modelin yön iddiası YOK, yalnız anomali tespiti',
 K2_ma20:'K2 — kapanış > 20-bar ortalama (kısa trend yukarı)',
 K3_trend:'K3 — MA20 > MA50 (orta vade trend yukarı)',
 K4_goreli:'K4 — 20-bar getiri piyasa medyanının üstünde (göreli güç)',
 K5_teyit:'K5 — 0 < 5-bar getiri ≤ +%15 (teyit var ama aşırı uzama yok)',
 K6_hacim:'K6 — hacim teyidi: 5-bar ortalama hacim / 60-bar ortalama ≥ 1'};
const KOSUL=['K1_model','K2_ma20','K3_trend','K4_goreli','K5_teyit','K6_hacim'];

function kolonlarAl(){const c=[
 {c:'hisse',h:'Hisse',t:'metin',s:1},
 {c:'kategori',h:'Kategori',t:'kategori',s:1,i:'Skor ≥6 GÜÇLÜ AL, 5 AL, 4 İZLE'},
 {c:'skor',h:'Skor',t:'sayi',i:'Geçilen koşul sayısı (0-6)'},
 {c:'kapanis',h:'Kapanış',t:'sayi2',i:'Kesit günü kapanışı (TL)'},
 {c:'gz_sigma',h:'gzσ',t:'sigma',i:'Model risk filtresi — o gün GERÇEKTEN bilinen sigma (sigma +8 bar gecikmeli yayınlanır). |gzσ|>3 = olağandışı, > +1 = pozitif kuyruk riski'},
 {c:'r5_%',h:'r5%',t:'getiri1',i:'Son 5 bar getiri — 0 < r5 ≤ %15 aranan bant'},
 {c:'r20_%',h:'r20%',t:'getiri1',i:'Son 20 bar getiri'},
 {c:'rel20_%',h:'rel%',t:'getiri1',i:'r20 − piyasa medyanı: göreli güç'},
 {c:'hacim_x',h:'Hacim×',t:'sayi2',i:'5-bar ort hacim / 60-bar ort hacim (≥1 = teyit)'}];
 KOSUL.forEach(k=>c.push({c:k,h:k.split('_')[0],t:'bool',i:KOSUL_ACIKLAMA[k]}));
 c.push({c:'senaryo_med_%',h:'Sen.med%',t:'getiri1',i:'k-NN senaryo medyan getirisi (--senaryo ile gelir)'});
 c.push({c:'senaryo_yukselen_%',h:'Sen.yüks%',t:'sayi0',i:'Senaryo komşularının yükselen oranı (%)'});
 return c;}
function kolonlarSenaryo(){return [
 {c:'hisse',h:'Hisse',t:'metin',s:1},
 {c:'baglam_sonu',h:'Kesit',t:'metin',i:'Son pencerenin bittiği gün — taramanın hangi güne baktığı'},
 {c:'medyan_senaryo_getiri_%',h:'Medyan',t:'getiri2',i:'Bu hissenin son 64 barı, geçmişteki en benzer 25 pencereyle eşlendi; bu 25 komşunun SONRAKİ 8 bardaki gerçek getirisinin ORTANCASI'},
 {c:'daga',h:'p10 → p90 bandı',t:'daga',s:1,i:'Kötü senaryo (p10) ile iyi senaryo (p90) arası bant; siyah çizgi = medyan; dikey çizgi = 0. Bant dar = model daha emin'},
 {c:'p10_%',h:'p10',t:'getiri2',i:'Kötü senaryo: komşuların %10\'u bundan DAHA KÖTÜ kapandı. Derin negatif = kuyruk riski'},
 {c:'p25_%',h:'p25',t:'getiri2'},
 {c:'p75_%',h:'p75',t:'getiri2'},
 {c:'p90_%',h:'p90',t:'getiri2',i:'İyi senaryo: komşuların %90\'ı bundan aşağıda kaldı'},
 {c:'yukselen_orani_%',h:'Yükselen%',t:'sayi0',i:'25 komşudan kaçı 8 bar sonunda KÂRDA idi (%)'},
 {c:'ortalama_senaryo_getiri_%',h:'Ortalama',t:'getiri2',i:'Komşu getirilerinin ortalaması (medyanı uçlara daha duyarlı)'},
 {c:'k',h:'k',t:'sayi',i:'Komşu sayısı (25)'}];}
function kolonlarSurpriz(){return [
 {c:'hisse',h:'Hisse',t:'metin',s:1},
 {c:'sigma_skoru',h:'σ',t:'sigma',i:'Modelin "bu bar beklediğimden farklı" düzeyi (kayan medyan/MAD\'a göre σ biriminde). |σ|>3 = OLAĞANDIŞI (~%0.3 beklenen)'},
 {c:'surprise',h:'skor',t:'sayi4',i:'0-1 arası normalize olağandışılık'},
 {c:'durum',h:'Durum',t:'metin',i:'|σ|>3 ise OLAĞANDIŞI'}];}
function kolonlarDogrulama(){return [
 {c:'tarih',h:'Tarih',t:'metin',s:1},
 {c:'liste_boyutu',h:'Boyut',t:'sayi',i:'O gün AL kuralına uyan hisse sayısı'},
 {c:'liste_ileri%',h:'Liste%',t:'getiri2',i:'Listedekilerin 5-bar ileri getirisi (medyan, kapanış→kapanış)'},
 {c:'piyasa_ileri%',h:'Piyasa%',t:'getiri2',i:'TÜM hisselerin 5-bar ileri getiri medyanı'},
 {c:'fark%',h:'Fark',t:'getiri2',i:'Liste − piyasa (puan). Pozitif = kural piyasa kazanmış. Son birkaç gün "—" = 5-bar ufuk henüz dolmadı'},
 {c:'isabet%',h:'İsabet%',t:'sayi0',i:'Listedekilerin kaç %\'i pozitif kapattı'},
 {c:'gz_rejim',h:'Rejim',t:'rozet_rejim',i:'O günkü piyasa rejimi (gerçek-zamanlı görünüm)'}];}
function kolonlarRadar(){return [
 {c:'tarih',h:'Tarih',t:'metin',s:1},
 {c:'gz_alarm_oran_%',h:'GZ alarm%',t:'sayi1',i:'O gün GERÇEKTEN bilinen ±3σ dışı hisse oranı (%) — karar için bu kolon'},
 {c:'alarm_oran_%',h:'alarm%',t:'sayi1',i:'±3σ dışı hisse oranı (tahmin-tarihli; değeri +8 barda belli olur)'},
 {c:'medyan_|sigma|',h:'med|σ|',t:'sayi2',i:'Hisseler arası medyan |sigma|'},
 {c:'piyasa_1bar_%',h:'p-1bar%',t:'getiri2',i:'Piyasa medyan 1-bar getiri'},
 {c:'piyasa_5bar_%',h:'p-5bar%',t:'getiri2',i:'Piyasa medyan 5-bar getiri'},
 {c:'yukselen_5bar_%',h:'yüks5%',t:'sayi0',i:'5 barda yükselen hisse oranı (%)'},
 {c:'ileri_5bar_%',h:'ileri5%',t:'getiri2',i:'O günden 5 bar SONRA gerçekleşen piyasa getirisi'},
 {c:'rejim',h:'Rejim',t:'rozet_rejim'},
 {c:'gz_rejim',h:'GZ-Rejim',t:'rozet_rejim',i:'Gerçek-zamanlı rejim: kararlar buna göre'}];}
function kolonlarHindsight(){return [
 {c:'hisse',h:'Hisse',t:'metin',s:1},
 {c:'kategori',h:'Kategori',t:'kategori',s:1},
 {c:'skor',h:'Skor',t:'sayi'},
 {c:'gz_sigma',h:'gzσ',t:'sigma'},
 {c:'ref_tarih',h:'Ref gün',t:'metin',i:'Listenin üretildiği gün'},
 {c:'ref_kapanis',h:'Ref kapanış',t:'sayi2'},
 {c:'kontrol_tarih',h:'Kontrol gün',t:'metin',i:'Ölçüm yapılan gün'},
 {c:'kontrol_kapanis',h:'Kontrol kapanış',t:'sayi2'},
 {c:'getiri_%',h:'Getiri',t:'getiri2',i:'Ref → kontrol kapanışı arası gerçekleşen getiri (%)'}];}

const KOLONLAR={al:kolonlarAl,senaryo:kolonlarSenaryo,surpriz:kolonlarSurpriz,
  dogrulama:kolonlarDogrulama,radar:kolonlarRadar,hindsight:kolonlarHindsight};
const TABLO_BASLIK={al:'AL Listesi — bugünün sinyalleri',
  senaryo:'Senaryo Taraması — k-NN analoji (keşif aracı, doğrulanmamış)',
  surpriz:'Sürpriz Taraması — modelin şaşırdığı barlar',
  dogrulama:'Doğrulama — kuralın tarihsel defter tutanlığı',
  radar:'Piyasa Radarı — rejim serisi',
  hindsight:'Hindsight Değerlendirme — sonradan ölçüm'};

/* ================= tablo motoru ================= */
const TABLO_DURUM={};   // key -> {sirala:{k,artan}, ara:'', hepsi:false}
function durumGetir(k){if(!TABLO_DURUM[k])TABLO_DURUM[k]={sirala:null,ara:'',hepsi:false};return TABLO_DURUM[k];}

function dagaCtx(satirlar){let lo=0,hi=0;
  satirlar.forEach(r=>{if(r['p10_%']!==null&&r['p10_%']!==undefined)lo=Math.min(lo,r['p10_%']);
    if(r['p90_%']!==null&&r['p90_%']!==undefined)hi=Math.max(hi,r['p90_%']);});
  const pad=Math.max(1,(hi-lo)*0.03);return{lo:lo-pad,hi:hi+pad};}
function dagaHucre(r,ctx){const p10=r['p10_%'],p90=r['p90_%'],med=r['medyan_senaryo_getiri_%'];
  if(p10===null||p90===null||p10===undefined||p90===undefined)return '<span class="nokta">—</span>';
  const X=v=>Math.max(0,Math.min(100,(v-ctx.lo)/(ctx.hi-ctx.lo)*100));
  const a=X(p10),b=Math.max(X(p10)+1,X(p90)),m=X(med),s=X(0);
  return '<div class="daga"><div class="bant" style="left:'+a+'%;width:'+(b-a)+'%"></div>'+
    '<div class="sifir" style="left:'+s+'%"></div><div class="ort" style="left:'+m+'%"></div></div>';}

function satirHazirla(key,satirlar){let s=satirlar.map(r=>Object.assign({},r));
  if(key==='surpriz')s.forEach(r=>{r.durum=(r.sigma_skoru!==null&&r.sigma_skoru!==undefined&&Math.abs(r.sigma_skoru)>3)?'OLAĞANDIŞI':'normal';});
  const d=durumGetir(key);
  if(d.ara){const q=d.ara.toLocaleLowerCase('tr');
    s=s.filter(r=>Object.values(r).some(v=>v!==null&&v!==undefined&&String(v).toLocaleLowerCase('tr').includes(q)));}
  if(d.sirala){const{k,artan}=d.sirala;
    s.sort((a,b)=>{const x=a[k],y=b[k];
      if(x===null||x===undefined)return 1;if(y===null||y===undefined)return -1;
      const c=(typeof x==='number'&&typeof y==='number')?x-y:String(x).localeCompare(String(y),'tr');
      return artan?c:-c;});}
  return s;}

function tabloKur(key){
  const kolonlar=KOLONLAR[key](),satirlar=VERI[key]||[];
  if(!satirlar.length)return '<div class="kutu">Bu bölümün CSV\'si bulunamadı.<br><span class="kucuk">Komutlar sekmesinde üretme komutunu bulabilirsin; ya da dosyayı buraya sürükle.</span></div>';
  const d=durumGetir(key);
  const hazir=satirHazirla(key,satirlar);
  const ctx=key==='senaryo'?dagaCtx(hazir):null;
  const SINIR=d.hepsi?hazir.length:50;
  const gorunen=hazir.slice(0,SINIR);
  let h='<div class="bolum-baslik"><h2 style="margin:0">'+TABLO_BASLIK[key]+'</h2>'+
    '<input class="arama" placeholder="Ara (hisse/tarih/rejim…)" value="'+ESC(d.ara)+'" oninput="araGir(\''+key+'\',this.value)">'+
    '<span class="kucuk">'+hazir.length+' satır</span>'+
    (hazir.length>SINIR?'<button class="buton" onclick="hepsiGec(\''+key+'\')">Tümünü göster ('+hazir.length+')</button>':'')+
    (d.hepsi&&hazir.length>50?'<button class="buton" onclick="hepsiGec(\''+key+'\')">İlk 50</button>':'')+
    '</div><div style="overflow:auto;max-height:72vh"><table><thead><tr>'+
    kolonlar.map(k=>'<th class="'+(k.s?'soltaraf':'')+'" title="'+ESC(k.i||'')+'" onclick="sirala(\''+key+'\',\''+k.c+'\')">'+k.h+
      (d.sirala&&d.sirala.k===k.c?(d.sirala.artan?' ▲':' ▼'):'')+'</th>').join('')+
    '</tr></thead><tbody>'+
    gorunen.map(r=>'<tr>'+kolonlar.map(k=>{
      if(k.t==='daga')return '<td class="soltaraf">'+dagaHucre(r,ctx)+'</td>';
      return '<td class="'+(k.s?'soltaraf':'')+'">'+hucre(r[k.c],k.t,k.hass)+'</td>';}).join('')+'</tr>').join('')+
    '</tbody></table></div>';
  if(key==='senaryo')h+='<p class="kucuk">Okuma sırası: önce p10 (en kötü ne kadar?) → yükselen% (ne sıklıkla kâr?) → medyan. '+
    'Bant dar + p10 sığ + yükselen yüksek = modelin en net olumlu görüşü. Benzerlik ≠ gelecek garantisi — bu bir keşif listesidir.</p>';
  if(key==='dogrulama')h+='<p class="kucuk">"—" görünen son satırlar: 5-bar ufuk henüz dolmadı (gelecek henüz gerçekleşmedi); birkaç gün sonra otomatik ölçülür.</p>';
  return h;
}
function sirala(key,k){const d=durumGetir(key);
  d.sirala=(d.sirala&&d.sirala.k===k)?{k,artan:!d.sirala.artan}:{k,artan:false};ciz();}
function araGir(key,v){durumGetir(key).ara=v;ciz();}
function hepsiGec(key){const d=durumGetir(key);d.hepsi=!d.hepsi;ciz();}
/* ================= özet bölümü ================= */
function sparkline(){const g=(VERI.dogrulama||[]).filter(r=>r['fark%']!==null&&r['fark%']!==undefined).slice(-120);
 if(g.length<3)return '';
 const vals=g.map(r=>r['fark%']),w=620,hh=90;
 const lo=Math.min(...vals,0),hi=Math.max(...vals,0);
 const X=i=>i/(vals.length-1)*(w-10)+5;
 const Y=v=>hh-5-(v-lo)/((hi-lo)||1)*(hh-10);
 const pts=vals.map((v,i)=>X(i).toFixed(1)+','+Y(v).toFixed(1)).join(' ');
 const y0=Y(0).toFixed(1);
 return '<svg width="'+w+'" height="'+hh+'" style="max-width:100%;display:block">'+
  '<line x1="0" y1="'+y0+'" x2="'+w+'" y2="'+y0+'" style="stroke:var(--sifir)" stroke-dasharray="4 3"/>'+
  '<polyline points="'+pts+'" style="fill:none;stroke:var(--izge);stroke-width:1.8"/></svg>';}

function bolumOzet(){
 const d=OZET.dogrulama||{},s=OZET.senaryo||{},p=OZET.surpriz||{},a=OZET.al;
 let h='';
 h+='<div class="kart"><div class="kpi-satir">';
 if(OZET.rejim)h+='<div class="kpi"><div class="deger"><span class="rozet '+(OZET.rejim.ad||'')+'">'+ESC(OZET.rejim.ad||'?')+'</span></div>'+
   '<div class="etiket">Piyasa rejimi'+(OZET.rejim.tarih?' — '+ESC(OZET.rejim.tarih):'')+'</div>'+
   (OZET.rejim.neden?'<div class="ek">'+ESC(OZET.rejim.neden)+'</div>':'<div class="ek">kaynak: '+ESC(OZET.rejim.kaynak||'')+'</div>')+'</div>';
 if(a)h+='<div class="kpi"><div class="deger">'+a.n+'</div><div class="etiket">AL listesi hisse</div>'+
   '<div class="ek">'+a.guclu+' GÜÇLÜ AL'+(a.top?' • başta '+ESC(a.top):'')+'</div></div>';
 if(d.n_gun)h+='<div class="kpi"><div class="deger '+(d.medyan_fark>0?'poz':'neg')+'">'+(d.medyan_fark>0?'+':'')+d.medyan_fark+' pt</div>'+
   '<div class="etiket">Doğrulama medyan fark ('+d.n_gun+' gün)</div>'+
   '<div class="ek">%'+d['pozitif_gun_%']+'\'ünde liste önde'+(d.alarm_gun?' • ALARM günleri '+(d.alarm_fark>0?'+':'')+d.alarm_fark+' pt':'')+'</div></div>';
 if(s.n)h+='<div class="kpi"><div class="deger">'+s.pozitif+'/'+s.n+'</div><div class="etiket">Senaryoda pozitif</div>'+
   '<div class="ek">en yüksek '+ESC(s.top||'—')+' '+(s.top_medyan>0?'+':'')+(s.top_medyan||0)+'%</div></div>';
 if(p.n)h+='<div class="kpi"><div class="deger '+(p.alarm?'neg':'poz')+'">'+p.alarm+'</div><div class="etiket">±3σ dışı hisse (surprise)</div>'+
   '<div class="ek">'+p.n+' hisse tarandı</div></div>';
 h+='</div></div>';
 h+='<div class="tavsiye '+OZET.tavsiye_seviye+'"><b>🧭 Bugün ne yapmalı</b>'+ESC(OZET.tavsiye)+'</div>';
 if(OZET.anlati.length)h+='<div class="kart"><h2>📋 Panorama <span class="soluk">— düz Türkçe özet</span></h2><ul class="anlati">'+
   OZET.anlati.map(x=>'<li>'+ESC(x)+'</li>').join('')+'</ul></div>';
 if(VERI.dogrulama&&VERI.dogrulama.length)h+='<div class="kart"><h2>Doğrulama eğrisi <span class="soluk">— günlük liste − piyasa farkı (puan; son '+Math.min(120,(VERI.dogrulama||[]).filter(r=>r['fark%']!=null).length)+' ölçülebilir gün)</span></h2>'+sparkline()+'</div>';
 h+='<div class="kart"><h2>🗂 Dosya durumu</h2><table><thead><tr><th class="soltaraf">Bölüm</th><th class="soltaraf">Durum</th>'+
  '<th class="soltaraf">Dosya</th><th>Satır</th><th class="soltaraf">Güncellenme</th><th class="soltaraf">Üreten komut</th></tr></thead><tbody>'+
  META.dosyalar.map(b=>'<tr><td class="soltaraf">'+ESC(b.sekme)+'</td>'+
   '<td class="soltaraf">'+(b.hata?'<span class="neg">HATA</span>':(b.bulundu?'<span class="poz">bulundu</span>':'<span class="neg">yok</span>'))+'</td>'+
   '<td class="soltaraf kucuk">'+ESC(b.dosya||'—')+'</td><td>'+(b.satir!=null?b.satir:'—')+'</td>'+
   '<td class="soltaraf kucuk">'+ESC(b.guncelleme||'—')+'</td><td class="soltaraf kucuk">'+ESC(b.ureten||'')+'</td></tr>').join('')+
  '</tbody></table>'+
  (META.cift.length?'<p class="kucuk" style="color:var(--amber)">⚠ Aynı CSV birden çok klasörde bulundu; EN YENİSİ kullanıldı: '+
   META.cift.map(c=>ESC(c.key)).join(', ')+'. Eski sürümleri ayrı klasörde tutmak sürüm karşılaştırmasını korur.</p>':'')+
  '<p class="kucuk">Paylaşım: bu HTML dosyasını olduğu gibi gönderin — offline açılır; alıcı CSV\'yi pencereye sürükleyip kendi verisini yükleyebilir.</p></div>';
 return h;}

/* ================= komutlar ================= */
const KOMUTLAR=[
 {baslik:'HER GÜN — rutin (yaklaşık 15-40 dk toplam)',adimlar:[
  {no:'1',komut:'python lewm_cli.py uygula --tum --model lewm_cikti\\lewm_model_eniyi.pt --cikti lewm_cikti',
   acik:'SÜRPRİZ taraması → surprise_tarama.csv. "Model hangi hisseye şaşırdı?" (hızlı: ~5-10 dk)'},
  {no:'2',komut:'python lewm_cli.py uygula --tum --senaryo-tarama --model lewm_cikti\\lewm_model_eniyi.pt --cikti lewm_senaryo_cikti',
   acik:'SENARYO taraması → senaryo_tarama.csv. k-NN analoji; referans havuzu tek sefer kurulur (yavaş: 10-30 dk)'},
  {no:'3',komut:'python lewm_al_listesi.py --piyasa-onay',
   acik:'AL LİSTESİ + 250-bar doğrulama → al_listesi.csv + al_dogrulama.csv. Piyasa ALARM/TEDBİR ise liste ÜRETİLMEZ (koruma). Gözlem amaçlı liste için --piyasa-onay yazmayın (kuru koşu).'},
  {no:'4',komut:'python lewm_piyasa_radari.py',
   acik:'PİYASA RADARI → piyasa_radari.csv + piyasa_radari.png. Rejim serisi ve çöküş yakalama analizi.'},
  {no:'5',komut:'python lewm_panel.py --ac',
   acik:'BU PANELİ yenile ve tarayıcıda aç. Tüm klasörleri tarar, tek dosya üretir.'}]},
 {baslik:'GEREKTİĞİNDE — eğitim, karşılaştırma, hindsight',adimlar:[
  {no:'6',komut:'python lewm_cli.py egit --epoch 12 --cikti lewm_cikti_e12',
   acik:'YENİ EĞİTİM. KURAL: her eğitim KENDİ --cikti klasörüne — eski modelin üzerine YAZMA (yoksa aynı-kesit karşılaştırması kapanır).'},
  {no:'7',komut:'python lewm_cli.py egit --devam --epoch 20 --cikti lewm_cikti_e12',
   acik:'Devam/uzatma: checkpoint\'tan kaldığı yerden sürer; scheduler yeni plana göre kurulur.'},
  {no:'8',komut:'python senaryo_karsilastir.py lewm_senaryo_cikti\\senaryo_tarama.csv lewm_senaryo_cikti_v2\\senaryo_tarama.csv --top 20',
   acik:'İki senaryo taramasını karşılaştır (Aynı kesitte koşulmuş olmalı!). Jaccard ve Spearman ≥0.5 ise model stabil; ikisi de düşükse eğitim uzatmayı düşün.'},
  {no:'9',komut:'python lewm_al_listesi.py --son-tarih 2026-10-16 --kontrol-tarih 2026-10-23',
   acik:'HINDSIGHT: geçmiş bir günün listesini bugünün kapanışlarıyla ölç → al_degerlendirme.csv. Bağımsız kanıt için haftalık koşturun.'},
  {no:'10',komut:'python lewm_harman.py',
   acik:'KURAL/MODEL HARMANI (ilk koşu): tüm hisselerde sigma geçişi + önbellek + 10 varyantın 250-bar kıyası → harman_karsilastirma.csv. Yavaş kısım yalnız ilk koşuda.'},
  {no:'11',komut:'python lewm_harman.py --onbellek-kullan --dogrula-bar 500',
   acik:'Harmanı yeniden tara: model/GPU AÇILMAZ, önbellekten saniyeler içinde. Pencereyi değiştirerek kazanan varyantın sağlamlığını sına (bar 250/500, ufuk 5/10).'},
  {no:'12',komut:'python lewm_cli.py egit --emb 256 --derinlik 5 --kafa 8 --epoch 30 --cikti lewm_cikti_buyuk',
   acik:'BÜYÜK MODEL: 2M→4.5M param, SIFIRDAN eğitim (başka klasöre!). Önceki bölümlerleme harmanı: egitim val eğrisi 12. epochta doygunluğa girer; ASIL kıyas downstream (harman/probe) üzerinden yapılır. OOM olursa --batch 128.'}]},
 {baslik:'NOTLAR',adimlar:[
  {no:'•',komut:'python lewm_cli.py --help',
   acik:'Ortak seçenekler (--epoch, --cikti, --periyot, --hisseler…) komuttan ÖNCE de SONRA da yazılabilir.'},
  {no:'•',komut:'python lewm_panel.py --dir lewm_cikti --dir lewm_senaryo_cikti --out lewm_panel.html',
   acik:'Panel üretici: varsayılan olarak lewm_cikti + lewm_senaryo_cikti (+ varsa lewm_al_cikti, lewm_radar_cikti, .) klasörlerini tarar.'},
  {no:'•',komut:'python lewm_panel.py --help',
   acik:'--out ile farklı dosya adı; --ac ile üretince otomatik aç.'}]}];
function kopyala(btn){const metin=btn.previousElementSibling.textContent;
 const basarili=()=>{btn.textContent='✓ Kopyalandı';setTimeout(()=>btn.textContent='📋 Kopyala',1500);};
 if(navigator.clipboard&&navigator.clipboard.writeText)
   navigator.clipboard.writeText(metin).then(basarili).catch(()=>yedekKopya(metin,basarili,btn));
 else yedekKopya(metin,basarili,btn);}
function yedekKopya(metin,basarili,btn){const ta=document.createElement('textarea');
 ta.value=metin;document.body.appendChild(ta);ta.select();
 try{document.execCommand('copy');basarili();}catch(e){btn.textContent='✗';}document.body.removeChild(ta);}
function bolumKomutlar(){return KOMUTLAR.map(b=>'<div class="kart"><h2>'+b.baslik+'</h2>'+
  b.adimlar.map(x=>'<div class="adim"><div class="no">'+x.no+'</div><div class="govde">'+
   '<div class="acik">'+ESC(x.acik)+'</div><div class="komut"><span>'+ESC(x.komut)+
   '</span><button onclick="kopyala(this)">📋 Kopyala</button></div></div></div>').join('')+'</div>').join('');}

/* ================= sözlük ================= */
const SOZLUK=[
 {grup:'İKİ LİSTENİN FARKI (en sık karışan konu)',maddeler:[
  ['Senaryo taraması','Latent uzayda k-NN analoji: "son 64 bar geçmişteki hangi durumlara benziyor, onlar SONRA ne yaşadı?" Keşif aracıdır; hindsight doğrulaması YOK. Düşmüş/toparlanmakta adlar öne çıkabilir.'],
  ['AL listesi','K1-K6 kural zinciri; 250-bar hindsight doğrulaması VAR. Trend ister: yükselmiş, MA üstünde, hacimli adlar. İki listenin kesişimi düşükse bu ZIT okuma değil — farklı sorular soruyorlar.'],
  ['gzσ','Gerçek-zamanlı sigma. Sigma serisi tahmin-tarihli: satır t\'nin değeri t+8 kapanışında bilinir → +8 bar kaydırılır. Karar yalnız buna bakmalı.'],
  ['Harman (lewm_harman)','AL kurallarının 10 varyantının aynı 250-bar pencerede kıyası: TABAN (skor 4/5/6), K1 sert/sakin, K7 sakin ek, hacim/uzama sert, rel20 ilk 15, ALARM-gün filtresi. Amaç: isabet%/fark%\'ı yükselten harmanı bulmak; kazananı farklı pencere/ufukta tekrar sınayın.'],
  ['Aynı kesit','İki taramayı karşılaştırmadan önce baglam_sonu (veya panelin kesit bilgisi) aynı gün mü diye bak; farklı günlerde piyasa hareketi farkı açıklar.']]},
 {grup:'SENARYO kolonları',maddeler:[
  ['medyan_senaryo_getiri_%','25 benzer geçmiş pencerenin SONRAKİ 8 bar getirisinin ortancası. +10% demek "benzer durumlar ortalama +%10 yapmış" demek — gelecek garantisi değil.'],
  ['p10_%','%10 dilimi: komşuların %10\'u bundan da KÖTÜ kapandı. Kötü senaryo sınırı. Derin negatif (örn. -15%) = kuyruk riski.'],
  ['p90_%','%90 dilimi: iyi senaryo sınırı. p90-p10 aralığı dar = model daha emin; geniş = belirsiz.'],
  ['yukselen_orani_%','25 komşudan kaçının 8 bar sonunda kârda olduğu (%).'],
  ['k','Komşu sayısı (25).']]},
 {grup:'AL kolonları',maddeler:[
  ['skor','Geçilen koşul sayısı 0-6. 6=GÜÇLÜ AL, 5=AL, 4=İZLE (eşik --esik-skor, varsayılan 5).'],
  ['K1_model','Model risk filtresi: gzσ ≤ +1 VE |gzσ| ≤ 3. Kanıt: pozitif sigma kuyruğu tarihsel olarak piyasa altı getirdi — modelin yön iddiası yok, anomali tespiti var.'],
  ['K2_ma20','Kapanış > MA20 — kısa trend yukarı.'],
  ['K3_trend','MA20 > MA50 — orta vade trend yukarı.'],
  ['K4_goreli','20-bar göreli güç > piyasa medyanı.'],
  ['K5_teyit','0 < r5 ≤ +%15 — teyit var ama aşırı uzama yok (kovalama filtresi).'],
  ['K6_hacim','5-bar ort hacim / 60-bar ort hacim ≥ 1 — hacim teyidi.'],
  ['r5_% / r20_%','Son 5 / 20 bar getiri. rel20_% = r20 − piyasa medyanı.'],
  ['hacim_x','Hacim teyit oranı.']]},
 {grup:'DOĞRULAMA kolonları',maddeler:[
  ['liste_ileri%','O günkü listedekilerin 5-bar ileri getirisi (medyan, kapanış→kapanış).'],
  ['piyasa_ileri%','TÜM hisselerin 5-bar ileri getiri medyanı — benchmark.'],
  ['fark%','Liste − piyasa (puan). Pozitif = kural o gün piyasa kazandı.'],
  ['isabet%','Listedekilerin pozitif kapananların oranı.'],
  ['"—" satırları','Son ~5 gün: 5-bar ufuk henüz dolmadı; ölçüm gelecekte otomatik gelir.']]},
 {grup:'SÜRPRİZ ve RADAR',maddeler:[
  ['sigma_skoru','Modelin "beklediğimden farklı" düzeyi (log-uzayda kayan medyan/MAD kalibrasyonu). |σ|>3 = OLAĞANDIŞI; tesadüfen beklenen oran ~%0.3.'],
  ['surprise','Ham sürpriz kare-mesafesi, 0-1 normalize.'],
  ['gz_alarm_oran_%','O gün gerçekten bilinen ±3σ dışı hisse oranı — karar için BU kolon.'],
  ['alarm_oran_%','Tahmin-tarihli oran; değeri +8 barda belli olur (geçmişe yazılır).'],
  ['ileri_5bar_%','O günden 5 bar sonra gerçekleşen piyasa getirisi — radarın isabet ölçümü.']]},
 {grup:'DÜRÜSTLÜK NOTLARI',maddeler:[
  ['Sızıntı kontrolü','sigma GZ kaydırılmış; AL koşulları yalnız D kapanışında bilinenlerle hesaplanır; hindsight modunda veri tarihe kadar FİZİKSEL kırpılır.'],
  ['Checkpoint','Bugünkü ağırlıklarla üretilir — ağırlık sızıntısı içerir (girdi sızıntısı yok).'],
  ['Keşif ≠ karar','Senaryo/sürpriz taramaları keşif; AL listesi doğrulanmış sinyal; hiçbiri yatırım tavsiyesi değil.']]}];
function bolumSozluk(){return SOZLUK.map(g=>'<div class="kart"><h2>'+g.grup+'</h2><dl class="sozluk">'+
  g.maddeler.map(m=>'<dt>'+ESC(m[0])+'</dt><dd>'+ESC(m[1])+'</dd>').join('')+'</dl></div>').join('');}

/* ================= CSV yükleme ================= */
const DOSYA_TIPI={'al_listesi.csv':'al','al_dogrulama.csv':'dogrulama',
 'al_degerlendirme.csv':'hindsight','senaryo_tarama.csv':'senaryo',
 'surprise_tarama.csv':'surpriz','piyasa_radari.csv':'radar'};
function csvSatirlari(metin){const satirlar=[];let satir=[],hane='',tirnak=false;
 for(let i=0;i<metin.length;i++){const ch=metin[i];
  if(tirnak){if(ch==='"'){if(metin[i+1]==='"'){hane+='"';i++;}else tirnak=false;}else hane+=ch;}
  else if(ch==='"')tirnak=true;
  else if(ch===','){satir.push(hane);hane='';}
  else if(ch==='\n'||ch==='\r'){if(ch==='\r'&&metin[i+1]==='\n')i++;
   satir.push(hane);hane='';if(satir.length>1||satir[0]!=='')satirlar.push(satir);satir=[];}
  else hane+=ch;}
 if(hane!==''||satir.length){satir.push(hane);satirlar.push(satir);}
 return satirlar;}
function csvNesneler(metin){const s=csvSatirlari(metin);if(!s.length)return[];
 const bas=s[0].map(x=>x.replace(/^\uFEFF/,'').trim());
 return s.slice(1).filter(r=>r.length>1||r[0]!=='').map(r=>{const o={};
  bas.forEach((b,i)=>{const v=(r[i]===undefined?'':r[i]).trim();
   o[b]=v===''?null:(v==='True'?true:v==='False'?false:(/^[-+]?\d+(\.\d+)?$/.test(v)?parseFloat(v):v));});
  return o;});}
function ozetYenile(){const tarihler=[];
 if(VERI.radar&&VERI.radar.length)tarihler.push(VERI.radar[VERI.radar.length-1].tarih);
 if(VERI.dogrulama&&VERI.dogrulama.length)tarihler.push(VERI.dogrulama[VERI.dogrulama.length-1].tarih);
 if(VERI.senaryo)VERI.senaryo.forEach(r=>{if(r.baglam_sonu)tarihler.push(r.baglam_sonu);});
 if(tarihler.length)OZET.kesit=tarihler.reduce((a,b)=>a>b?a:b);
 if(VERI.radar&&VERI.radar.length){const son=VERI.radar[VERI.radar.length-1];
  OZET.rejim={ad:son.gz_rejim||son.rejim||'?',neden:son.gz_neden||'',tarih:son.tarih||'',kaynak:'piyasa_radari.csv'};}
 else if(VERI.dogrulama&&VERI.dogrulama.length){const son=VERI.dogrulama[VERI.dogrulama.length-1];
  OZET.rejim={ad:son.gz_rejim||'?',neden:'',tarih:son.tarih||'',kaynak:'al_dogrulama.csv'};}
 if(VERI.al){const L=VERI.al,guclu=L.filter(r=>r.kategori==='GÜÇLÜ AL');
  OZET.al={n:L.length,guclu:guclu.length,top:L.length?L[0].hisse:'',top_skor:L.length?L[0].skor:null};}
 if(VERI.dogrulama){const olc=VERI.dogrulama.filter(r=>r['fark%']!==null&&r['fark%']!==undefined);
  if(olc.length){const f=olc.map(r=>r['fark%']);
   const alarm=olc.filter(r=>r.gz_rejim==='ALARM').map(r=>r['fark%']);
   OZET.dogrulama={n_gun:olc.length,bekleyen:VERI.dogrulama.length-olc.length,
    medyan_fark:Math.round(medyan(f)*100)/100,
    'pozitif_gun_%':Math.round(f.filter(x=>x>0).length/f.length*100),
    alarm_gun:alarm.length,
    alarm_fark:alarm.length?Math.round(medyan(alarm)*100)/100:null};}
  else OZET.dogrulama=null;}
 if(VERI.senaryo){const S=VERI.senaryo,poz=S.filter(r=>(r['medyan_senaryo_getiri_%']||0)>0);
  let kotu=null;S.forEach(r=>{if(r['p10_%']!=null&&(!kotu||r['p10_%']<kotu['p10_%']))kotu=r;});
  OZET.senaryo={n:S.length,pozitif:poz.length,top:S.length?S[0].hisse:'',
   top_medyan:S.length?S[0]['medyan_senaryo_getiri_%']:null,
   kotu_p10_hisse:kotu?kotu.hisse:'',kotu_p10:kotu?kotu['p10_%']:null};}
 if(VERI.surpriz){const P=VERI.surpriz.filter(r=>r.sigma_skoru!=null);
  const alarm=P.filter(r=>Math.abs(r.sigma_skoru)>3);
  OZET.surpriz={n:P.length,alarm:alarm.length,top:alarm.slice(0,3).map(r=>[r.hisse,r.sigma_skoru])};}}
function csvIsle(dosyaAdi,metin){
 const ad=dosyaAdi.split(/[\\/]/).pop().toLowerCase();
 const key=DOSYA_TIPI[ad];
 if(!key){toast('Tanınmayan dosya: '+ad+' — kabul edilenler: '+Object.keys(DOSYA_TIPI).join(', '));return;}
 const kayitlar=csvNesneler(metin);
 if(!kayitlar.length){toast(ad+' boş görünüyor.');return;}
 VERI[key]=kayitlar;ozetYenile();ciz();
 toast('✓ '+ad+' yüklendi → "'+key+'" bölümü güncellendi ('+kayitlar.length+' satır)');}
function csvYukle(f){const r=new FileReader();
 r.onload=e=>csvIsle(f.name,e.target.result);r.readAsText(f,'utf-8');}
document.addEventListener('dragover',e=>{e.preventDefault();
 $('#surukleOrtu').classList.add('gorunur');});
document.addEventListener('dragleave',e=>{if(!e.relatedTarget)
 $('#surukleOrtu').classList.remove('gorunur');});
document.addEventListener('drop',e=>{e.preventDefault();
 $('#surukleOrtu').classList.remove('gorunur');
 Array.from(e.dataTransfer.files).forEach(csvYukle);});
document.addEventListener('DOMContentLoaded',()=>{
 $('#csvDosya').addEventListener('change',e=>{
  Array.from(e.target.files).forEach(csvYukle);e.target.value='';});});

/* ================= sekme sistemi ================= */
const SEKMELER=[
 {id:'ozet',ad:'📊 Özet'},{id:'al',ad:'🎯 AL Listesi'},{id:'senaryo',ad:'🧭 Senaryo'},
 {id:'surpriz',ad:'⚡ Sürpriz'},{id:'dogrulama',ad:'✅ Doğrulama'},{id:'radar',ad:'📡 Radar'},
 {id:'hindsight',ad:'🧪 Hindsight'},{id:'komutlar',ad:'⌨️ Komutlar'},{id:'sozluk',ad:'📖 Sözlük'}];
let aktif='ozet';
function ciz(){
 $('#sekmeCubuk').innerHTML=SEKMELER.map(s=>{
  const bos=!['ozet','komutlar','sozluk'].includes(s.id)&&!(VERI[s.id]&&VERI[s.id].length);
  return '<button class="'+(s.id===aktif?'aktif':'')+'" onclick="sekmeGec(\''+s.id+'\')">'+s.ad+(bos?' ⚠':'')+'</button>';}).join('');
 let h='';
 if(aktif==='ozet')h=bolumOzet();
 else if(aktif==='komutlar')h=bolumKomutlar();
 else if(aktif==='sozluk')h=bolumSozluk();
 else h='<div class="kart">'+tabloKur(aktif)+'</div>';
 $('#icerik').innerHTML=h;}
function sekmeGec(id){aktif=id;ciz();window.scrollTo(0,0);}
function araGir(key,v){durumGetir(key).ara=v;ciz();
 const inp=$('#icerik .arama');
 if(inp){inp.focus();const n=inp.value.length;inp.setSelectionRange(n,n);}}

/* ================= tema (koyu/açık) ================= */
function temaUygula(t){document.documentElement.classList.toggle('koyu',t==='koyu');
 const b=$('#temaBtn');if(b)b.textContent=t==='koyu'?'☀️ Açık':'🌙 Koyu';
 try{localStorage.setItem('lewmTema',t);}catch(e){}}
function temaDegistir(){temaUygula(document.documentElement.classList.contains('koyu')?'acik':'koyu');}
(function(){let t=null;try{t=localStorage.getItem('lewmTema');}catch(e){}
 if(t!=='koyu'&&t!=='acik')t=(window.matchMedia&&window.matchMedia('(prefers-color-scheme: dark)').matches)?'koyu':'acik';
 temaUygula(t);})();

/* ================= başlat ================= */
$('#ustAlt').textContent='Üretim: '+META.uretim+(OZET.kesit?' • Kesit: '+OZET.kesit:'')+
 ' • Kaynak klasörler: '+META.dizinler.join(', ');
ciz();
</script>
</body>
</html>
"""
if __name__ == "__main__":
    ana()
