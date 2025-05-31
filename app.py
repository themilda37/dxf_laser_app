import streamlit as st
import ezdxf
import math
import tempfile
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

st.title("DXF Laser Kalkulačka")

# krok 1: upload souboru
uploaded = st.file_uploader("Nahraj DXF soubor", type="dxf")
if not uploaded:
    st.info("Prosím nahraj DXF soubor výše.")
    st.stop()

# krok 2: uložení uploadu do temp souboru a načtení DXF
with tempfile.NamedTemporaryFile(suffix=".dxf", delete=False) as tmp:
    tmp.write(uploaded.getbuffer())
    tmp_path = tmp.name

doc = ezdxf.readfile(tmp_path)
msp = doc.modelspace()

# Pomocná funkce pro výpočet plochy polygonu (shoelace formula)
def polygon_area(pts):
    if len(pts) < 3:
        return 0
    area = sum(
        pts[i].x * pts[(i + 1) % len(pts)].y - pts[(i + 1) % len(pts)].x * pts[i].y
        for i in range(len(pts))
    )
    return abs(area) / 2.0

# příprava proměnných
all_pts = []
total_length = 0.0

# krok 3: zpracování entit
for e in msp:
    try:
        path = ezdxf.path.make_path(e)
        pts = list(path.flattening(0.5))  # tolerance 0.5 mm
    except Exception:
        continue

    if not pts:
        continue

    all_pts.extend(pts)

    typ = e.dxftype()

    # Spočítáme délku křivky ručně z bodů
    length = sum(
        math.hypot(pts[i].x - pts[i - 1].x, pts[i].y - pts[i - 1].y)
        for i in range(1, len(pts))
    )

    total_length += length

# krok 4: výpočet bounding box z all_pts
if all_pts:
    # výpočet bounding boxu z all_pts
    xs = [pt.x for pt in all_pts]
    ys = [pt.y for pt in all_pts]
    bbox_result = {
        "xmin": min(xs), "ymin": min(ys),
        "xmax": max(xs), "ymax": max(ys)
    }
    # výpočet plochy bounding boxu
    width = bbox_result["xmax"] - bbox_result["xmin"]
    height = bbox_result["ymax"] - bbox_result["ymin"]
    bbox_area = width * height
else:
    bbox_result = None
    bbox_area = None
    st.warning(
        "Nebyl nalezen žádný bod pro výpočet bounding boxu – zkontrolujte typy entit v DXF."
    )
# krok 5: zobrazení výsledků
st.subheader("Celková délka křivek (m)")
st.write(f"{total_length/1000:.3f}")
# krok 6: vykreslení geometrií s posunutím 0 do pravého spodního rohu
fig, ax = plt.subplots()
ax.set_aspect('equal', 'box')
ax.set_title("Náhled geometrií z DXF (origin v pravém spodním rohu)")
ax.set_xlabel("X [mm]")
ax.set_ylabel("Y [mm]")
if bbox_result:
    dx = bbox_result["xmax"] - bbox_result["xmin"]
    dy = bbox_result["ymax"] - bbox_result["ymin"]
else:
    dx = dy = 0
for e in msp:
    try:
        path = ezdxf.path.make_path(e)
        pts = list(path.flattening(0.5))
        if not pts:
            continue
        xs = [dx - (pt.x - bbox_result["xmin"]) for pt in pts]
        ys = [(pt.y - bbox_result["ymin"]) for pt in pts]
        ax.plot(xs, ys, '-k')
    except Exception:
        continue
# nastavení rozsahu od 0 do dx a 0 do dy
ax.set_xlim(0, dx)
ax.set_ylim(0, dy)
if dx:
    ax.set_xticks([0, dx])
if dy:
    ax.set_yticks([0, dy])
st.pyplot(fig)

# --- Identifikace otvorů v DXF ---

from collections import defaultdict

# Parametry
TOLERANCE = 1.0  # mm, tolerance pro spojování koncových bodů

# Pomocná funkce: vzdálenost dvou bodů
def pt_dist(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)

# 1. Seskupení křivek do potenciálních kontur (s tolerancí)
def group_curves_to_loops(msp, tolerance=TOLERANCE):
    # Každý segment reprezentujeme jako seznam bodů (polyline)
    segments = []
    for e in msp:
        try:
            pts = list(ezdxf.path.make_path(e).flattening(0.5))
        except Exception:
            continue
        if len(pts) < 2:
            continue
        segments.append(pts)

    # Seskupování segmentů do smyček
    endpoints = defaultdict(list)
    for idx, seg in enumerate(segments):
        endpoints[(round(seg[0].x, 3), round(seg[0].y, 3))].append(('start', idx))
        endpoints[(round(seg[-1].x, 3), round(seg[-1].y, 3))].append(('end', idx))

    used = set()
    loops = []

    for i, seg in enumerate(segments):
        if i in used:
            continue
        loop = seg[:]
        used.add(i)
        changed = True
        while changed:
            changed = False
            for j, other in enumerate(segments):
                if j in used or i == j:
                    continue
                # Zkus napojit na konec
                if pt_dist(loop[-1], other[0]) < tolerance:
                    loop.extend(other[1:])
                    used.add(j)
                    changed = True
                    break
                elif pt_dist(loop[-1], other[-1]) < tolerance:
                    loop.extend(reversed(other[:-1]))
                    used.add(j)
                    changed = True
                    break
                # Zkus napojit na začátek
                elif pt_dist(loop[0], other[-1]) < tolerance:
                    loop = other[:-1] + loop
                    used.add(j)
                    changed = True
                    break
                elif pt_dist(loop[0], other[0]) < tolerance:
                    loop = list(reversed(other[1:])) + loop
                    used.add(j)
                    changed = True
                    break
        # Pokud je smyčka uzavřená (začátek a konec blízko)
        if pt_dist(loop[0], loop[-1]) < tolerance and len(loop) > 2:
            loops.append(loop)
    return loops

# 2. Výpočet obvodu smyčky
def loop_perimeter(pts):
    return sum(math.hypot(pts[i].x - pts[i-1].x, pts[i].y - pts[i-1].y) for i in range(1, len(pts))) + math.hypot(pts[0].x - pts[-1].x, pts[0].y - pts[-1].y)

# 3. Test bodu v polygonu (pro hierarchii otvorů)
def point_in_polygon(x, y, poly):
    n = len(poly)
    inside = False
    px, py = x, y
    for i in range(n):
        j = (i + 1) % n
        xi, yi = poly[i].x, poly[i].y
        xj, yj = poly[j].x, poly[j].y
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi + 1e-12) + xi):
            inside = not inside
    return inside

def polygon_centroid(pts):
    if len(pts) < 3:
        return pts[0].x, pts[0].y
    A = 0
    Cx = 0
    Cy = 0
    for i in range(len(pts)):
        j = (i + 1) % len(pts)
        cross = pts[i].x * pts[j].y - pts[j].x * pts[i].y
        A += cross
        Cx += (pts[i].x + pts[j].x) * cross
        Cy += (pts[i].y + pts[j].y) * cross
    A *= 0.5
    if abs(A) < 1e-12:
        return pts[0].x, pts[0].y
    Cx /= (6 * A)
    Cy /= (6 * A)
    return Cx, Cy

# 4. Identifikace otvorů podle even-odd pravidla
def identify_holes(loops):
    # Seřaď podle plochy (největší je vnější)
    loops_sorted = sorted(loops, key=lambda l: polygon_area(l), reverse=True)
    holes = []
    for idx, loop in enumerate(loops_sorted):
        cx, cy = polygon_centroid(loop)
        # Spočítej, kolik větších smyček tento centroid obsahuje
        inside_count = 0
        for j, outer in enumerate(loops_sorted):
            if j == idx:
                continue
            if polygon_area(outer) > polygon_area(loop):
                if point_in_polygon(cx, cy, outer):
                    inside_count += 1
        # Even-odd pravidlo: lichý počet vnoření = otvor
        if inside_count % 2 == 1:
            holes.append({
                "perimeter": loop_perimeter(loop),
                "points": loop
            })
    return holes

# --- Výpočet a výpis otvorů ---
loops = group_curves_to_loops(msp, tolerance=TOLERANCE)
holes = identify_holes(loops)

st.subheader(f"Otvorů: {len(holes)}")
if holes:
    import pandas as pd
    df_holes = pd.DataFrame([{"Obvod (m)": round(h["perimeter"]/1000, 3)} for h in holes])
    df_holes.index = df_holes.index + 1
    st.table(df_holes)
else:
    st.write("Žádné otvory nebyly rozpoznány.")

# Načti tabulku parametrů z Excelu
@st.cache_data
def load_param_table(path="parametry.xlsx"):
    # Ujisti se, že je nainstalován openpyxl
    try:
        import openpyxl  # noqa: F401
    except ImportError:
        st.error("Chybí knihovna 'openpyxl'. Nainstaluj ji příkazem: pip install openpyxl")
        return None
    return pd.read_excel(path, engine="openpyxl")

import os
excel_path = os.path.join(os.path.dirname(__file__), "parametry.xlsx")

try:
    param_table = load_param_table(excel_path)
except Exception as e:
    st.error(f"Chyba při načítání tabulky parametry: {e}")
    param_table = None

# Načti tabulku materiálů z Excelu
@st.cache_data
def load_material_table(path="materialy.xlsx"):
    try:
        import openpyxl  # noqa: F401
    except ImportError:
        st.error("Chybí knihovna 'openpyxl'. Nainstaluj ji příkazem: pip install openpyxl")
        return None
    return pd.read_excel(path, engine="openpyxl")

material_excel_path = os.path.join(os.path.dirname(__file__), "materialy.xlsx")
try:
    material_table = load_material_table(material_excel_path)
except Exception as e:
    st.error(f"Chyba při načítání tabulky materiálů: {e}")
    material_table = None

if param_table is not None and material_table is not None:
    # Rozbalovací volba pro materiál
    material_names = material_table["materiály"].dropna().unique()
    material = st.selectbox(
        "Materiál",
        sorted(material_names)
    )
    # Oprava: zjisti skutečný název sloupce pro sílu materiálu
    material_cols = [col.lower().replace(" ", "") for col in material_table.columns]
    if "sílamateriálu" in material_cols:
        sila_col = material_table.columns[material_cols.index("sílamateriálu")]
    elif "síla materiálu" in material_cols:
        sila_col = material_table.columns[material_cols.index("síla materiálu")]
    else:
        st.error("V tabulce materialy.xlsx nebyl nalezen sloupec 'síla materiálu'.")
        st.stop()

    # Rozbalovací volba pro Síla materiálu (mm) podle zvoleného materiálu (z tabulky materialy.xlsx)
    tloustky = material_table[material_table["materiály"] == material][sila_col].dropna().unique()
    tloustka = st.selectbox(
        "Síla materiálu (mm)",
        sorted(tloustky)
    )
    # Rozbalovací volba pro Pomocný plyn (zůstává z tabulky parametry)
    plyn = st.selectbox(
        "Pomocný plyn",
        sorted(param_table["Pomocný plyn"].dropna().unique())
    )
    material_row = material_table[(material_table["materiály"] == material) & (material_table[sila_col] == tloustka)].iloc[0]
    hustota = float(material_row["hustota kg/m³"])
    cena_kg = float(material_row["Kč/kg bez DPH"])

    # Najdi řádek odpovídající výběru v tabulce parametrů
    vybrane = param_table[
        (param_table["Pomocný plyn"] == plyn) &
        (param_table["Síla materiálu (mm)"] == tloustka)
    ]
    # Výpis rozměrů a plochy polotovaru
    if bbox_result:
        st.subheader("Polotovar")
        st.write({
            "šířka polotovaru (mm)": round(abs(bbox_result["xmax"] - bbox_result["xmin"]), 1),
            "výška polotovaru (mm)": round(abs(bbox_result["ymax"] - bbox_result["ymin"]), 1),
            "plocha polotovaru (m²)": round(bbox_area/1_000_000, 4)
        })

    # --- Parametry výpalku ---
    st.subheader("Parametry výpalku")
    vypalkova_plocha = 0
    if loops:
        sorted_loops = sorted(loops, key=lambda l: polygon_area(l), reverse=True)
        if sorted_loops:
            vnejsi = sorted_loops[0]
            vnejsi_plocha = polygon_area(vnejsi)
            otvory_plocha = sum(polygon_area(hole["points"]) for hole in holes)
            vypalkova_plocha = vnejsi_plocha - otvory_plocha
    vypalkova_plocha_m2 = vypalkova_plocha / 1_000_000

    tloustka_m = tloustka / 1000  # převod mm na m
    hmotnost_vypalku_kg = vypalkova_plocha_m2 * tloustka_m * hustota
    bbox_area_m2 = bbox_area / 1_000_000 if bbox_area else 0
    hmotnost_polotovaru_kg = bbox_area_m2 * tloustka_m * hustota if bbox_area else 0
    cena_materialu = hmotnost_polotovaru_kg * cena_kg

    st.write({
        "Plocha výpalku (m²)": round(vypalkova_plocha_m2, 4),
        "Hmotnost výpalku (kg)": round(hmotnost_vypalku_kg, 2),
        "Hmotnost polotovaru (kg)": round(hmotnost_polotovaru_kg, 2),
        "Cena materiálu (Kč)": round(cena_materialu, 2)
    })

    if not vybrane.empty:
        row = vybrane.iloc[0]
        # ...existing code for Parametry řezu...
        pocet_zapalu = 1 + len(holes)
        prodleva_zapal_s = float(row["Prodleva při zápalu (s)"])
        prodleva_celkem_s = pocet_zapalu * prodleva_zapal_s
        reznarychlost_mmin = float(row["Řezná rychlost (m/min)"])
        prutok_lmin = float(row["Průtok plynu (l/min)"])
        if reznarychlost_mmin > 0:
            cas_rezu_min = (total_length / 1000) / reznarychlost_mmin
            cas_rezu_s = cas_rezu_min * 60
        else:
            cas_rezu_min = 0
            cas_rezu_s = 0
        strojni_cas_s = cas_rezu_s + prodleva_celkem_s
        strojni_cas_min = strojni_cas_s / 60
        spotreba_plynu_l = prutok_lmin * strojni_cas_min

        st.subheader("Parametry řezu")
        st.write({
            "Řezná rychlost (m/min)": reznarychlost_mmin,
            "Prodleva při zápalu (s)": prodleva_zapal_s,
            "Počet zápalů": pocet_zapalu,
            "Prodleva celkem (s)": prodleva_celkem_s,
            "Průtok plynu (l/min)": prutok_lmin,
            "Tlak (bar)": float(row["Tlak (bar)"]),
            "Čas řezu (s)": round(cas_rezu_s, 1),
            "Celkový strojní čas (s)": round(strojni_cas_s, 1),
            "Spotřeba plynu (l)": round(spotreba_plynu_l, 1)
        })
    else:
        st.warning("Pro zvolený materiál nejsou v tabulce parametrů data - doplnit tabulku.")