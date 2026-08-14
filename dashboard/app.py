import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier
    XGB_AVAILABLE = False

try:
    import openpyxl  # noqa: F401
    OPENPYXL_AVAILABLE = True
except ImportError:
    OPENPYXL_AVAILABLE = False

try:
    import xlrd  # noqa: F401
    XLRD_AVAILABLE = True
except ImportError:
    XLRD_AVAILABLE = False


# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="Coding Stress ML Studio",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

FEATURE_COLUMNS = [
    'age', 'experience_years', 'daily_work_hours', 'sleep_hours',
    'caffeine_intake', 'bugs_per_day', 'commits_per_day',
    'meetings_per_day', 'screen_time', 'exercise_hours'
]
TARGET_COLUMN = 'burnout_level'
CLASS_ORDER = ['Low', 'Medium', 'High']  # urutan keparahan: Low=0, Medium=1, High=2

# Ikon per model, dipakai konsisten di semua tab supaya sesuai tema web
MODEL_ICONS = {
    'Random Forest': '🌲',
    'SVM': '🛡️',
    'XGBoost': '⚡',
    'XGBoost (fallback: GradientBoosting)': '⚡',
}

DEFAULT_DATASET_NAME = "developer_burnout_dataset_ervin.csv"

# =============================================================================
# GLOBAL STYLE
# =============================================================================
BG_MAIN = "#F3F6F3"
BG_CARD = "#FFFFFF"
INK = "#111315"
GREEN = "#17B978"
GREEN_DARK = "#0E8F5C"
GREEN_SOFT = "#E4F8EE"
AMBER = "#FFC94A"
RED = "#FF5A5F"

PALETTE_MAIN = ["#17B978", "#0E8F5C", "#7CD8B0", "#0B6E46", "#BFEFD8"]

st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=JetBrains+Mono:wght@500;700&display=swap');

    html, body, [data-testid="stAppViewContainer"] {{
        font-family: 'Inter', -apple-system, sans-serif;
        background: {BG_MAIN};
    }}
    [data-testid="stAppViewContainer"] > .main {{ background: {BG_MAIN}; }}
    [data-testid="stHeader"] {{ background: transparent; }}

    [data-testid="stSidebar"] {{
        background: #FFFFFF !important;
        border-right: 3px solid {INK} !important;
    }}
    [data-testid="stSidebar"] * {{ color: {INK}; }}

    .sidebar-logo-box {{
        background: linear-gradient(135deg, {GREEN} 0%, {GREEN_DARK} 100%);
        border: 3px solid {INK};
        border-radius: 14px;
        box-shadow: 4px 4px 0px {INK};
        padding: 16px 14px;
        text-align: center;
        margin: 6px 4px 22px 4px;
    }}
    .logo-mark {{
        width: 46px; height: 46px; margin: 0 auto;
        display: flex; align-items: center; justify-content: center;
        background: #ffffff; color: {GREEN_DARK};
        font-weight: 900; font-size: 17px; letter-spacing: -0.5px;
        border: 2.5px solid {INK}; border-radius: 12px;
        box-shadow: 2px 2px 0px rgba(17,19,21,0.5);
    }}
    .sidebar-logo-box h2 {{
        margin: 4px 0 0 0; font-weight: 900; color: #ffffff;
        font-size: 17px; line-height: 1.2; letter-spacing: -0.3px;
    }}
    .sidebar-logo-box p {{
        margin: 4px 0 0 0; color: rgba(255,255,255,0.9); font-size: 11px; font-weight: 600;
    }}

    .sidebar-section-label {{
        font-size: 11px; letter-spacing: 1px; color: #7a8082; font-weight: 800;
        margin: 4px 2px 8px 2px; text-transform: uppercase;
    }}

    div[data-testid="stRadio"] > label,
    div[data-testid="stRadio"] [data-testid="stWidgetLabel"] {{ display: none !important; }}
    div[data-testid="stRadio"] [role="radiogroup"] label div[data-checked] {{
        display: none !important; width: 0px !important; height: 0px !important;
    }}
    div[data-testid="stRadio"] [role="radiogroup"] label div:last-child {{
        padding-left: 0px !important; margin-left: 0px !important; width: 100% !important;
    }}
    div[data-testid="stRadio"] [role="radiogroup"] label p {{
        text-align: left !important; width: 100% !important; margin: 0 !important;
        font-size: 14px !important;
    }}
    div[data-testid="stRadio"] [role="radiogroup"] label {{
        background: #ffffff !important;
        border: 2px solid {INK} !important;
        border-radius: 10px !important;
        padding: 10px 14px !important;
        margin-bottom: 9px !important;
        width: 100% !important;
        transition: all 0.12s ease !important;
        cursor: pointer !important;
        box-shadow: 2px 2px 0px {INK} !important;
    }}
    div[data-testid="stRadio"] [role="radiogroup"] label:hover {{
        background: {GREEN_SOFT} !important;
        transform: translate(-1px, -1px);
        box-shadow: 3px 3px 0px {INK} !important;
    }}
    div[data-testid="stRadio"] [role="radiogroup"] label[data-checked="true"] {{
        background: {INK} !important;
        border-color: {INK} !important;
        box-shadow: 2px 2px 0px {GREEN} !important;
    }}
    div[data-testid="stRadio"] [role="radiogroup"] label[data-checked="true"] p {{
        color: #ffffff !important; font-weight: 700 !important;
    }}
    div[data-testid="stRadio"] [role="radiogroup"] label[data-checked="false"] p {{
        color: {INK} !important; font-weight: 600 !important;
    }}

    .sidebar-footer {{
        margin-top: 34px; font-size: 11.5px; color: #6b7072;
        border-top: 2px solid {INK}; padding-top: 12px;
    }}
    .sidebar-footer b {{ color: {INK}; }}

    div[data-testid="stVerticalBlockBorderWrapper"] {{
        background: {BG_CARD} !important;
        border: 2.5px solid {INK} !important;
        border-radius: 16px !important;
        padding: 6px !important;
        box-shadow: 4px 4px 0px rgba(17,19,21,0.85) !important;
        margin-bottom: 10px;
        transition: box-shadow 0.15s ease, transform 0.15s ease;
    }}
    div[data-testid="stVerticalBlockBorderWrapper"]:hover {{
        box-shadow: 5px 5px 0px rgba(17,19,21,0.9) !important;
        transform: translate(-1px, -1px);
    }}
    div[data-testid="stVerticalBlockBorderWrapper"] > div {{ padding: 20px 22px !important; }}

    h1, h2, h3, h4, h5, p, span, label, .stMarkdown {{ color: {INK}; }}

    /* ---------- STEP HEADER (badge nomor + judul, pengganti emoji angka) ---------- */
    .step-header-row {{
        display: flex; align-items: flex-start; gap: 12px;
        margin-bottom: 14px;
        padding-bottom: 12px;
        border-bottom: 2px solid rgba(17,19,21,0.08);
    }}
    .step-badge {{
        flex: none;
        width: 30px; height: 30px;
        display: flex; align-items: center; justify-content: center;
        background: {INK}; color: #ffffff;
        font-weight: 800; font-size: 14px;
        border-radius: 8px;
        margin-top: 1px;
    }}
    .step-title {{
        margin: 0; font-size: 18px; font-weight: 800; color: {INK};
        letter-spacing: -0.2px; line-height: 1.3;
    }}
    .step-caption {{
        margin: 3px 0 0 0; font-size: 13px; color: #6b7072; font-weight: 500;
    }}

    .page-header-wrap {{
        display: flex; align-items: center; justify-content: space-between;
        background: #ffffff; border: 2.5px solid {INK}; border-radius: 16px;
        padding: 16px 22px; margin-bottom: 18px;
        box-shadow: 4px 4px 0px rgba(17,19,21,0.85);
    }}
    .page-header-left {{ display: flex; align-items: center; gap: 14px; }}
    .page-header-icon {{
        width: 46px; height: 46px; min-width: 46px; border-radius: 12px;
        background: {GREEN_SOFT}; border: 2.5px solid {INK};
        display: flex; align-items: center; justify-content: center;
        font-size: 22px;
    }}
    .page-header-title {{ font-size: 21px; font-weight: 900; color: {INK}; margin: 0; letter-spacing: -0.4px; }}
    .page-header-breadcrumb {{ font-size: 12.5px; color: {GREEN_DARK}; font-weight: 700; margin: 0; }}

    .badge {{
        display: inline-block; padding: 6px 14px; border-radius: 20px;
        font-size: 12px; font-weight: 800; letter-spacing: 0.3px;
        text-transform: uppercase; border: 2px solid {INK};
    }}
    .badge-green {{ background: {GREEN}; color: #ffffff; box-shadow: 2px 2px 0px {INK}; }}
    .badge-amber {{ background: {AMBER}; color: #3a2600; box-shadow: 2px 2px 0px {INK}; }}
    .badge-dark  {{ background: {INK}; color: #ffffff; box-shadow: 2px 2px 0px {GREEN}; }}

    .subtitle-muted {{ color: #5c6163; font-size: 14.5px; margin-top: -6px; }}

    button[data-baseweb="tab"] {{
        color: #6b7072 !important; font-weight: 700 !important;
        border-radius: 10px 10px 0 0 !important;
    }}
    button[data-baseweb="tab"][aria-selected="true"] {{
        color: {INK} !important;
        background: {GREEN_SOFT} !important;
        border-bottom: 3px solid {GREEN} !important;
    }}

    [data-testid="stMetric"] {{
        background: {GREEN_SOFT};
        border: 2.5px solid {INK};
        border-radius: 14px; padding: 14px 16px;
        box-shadow: 3px 3px 0px {INK};
    }}
    [data-testid="stMetricLabel"] {{ color: {GREEN_DARK} !important; font-weight: 700 !important; }}
    [data-testid="stMetricValue"] {{ color: {INK} !important; font-weight: 900 !important; }}

    [data-testid="stDataFrame"] {{
        border-radius: 12px; overflow: hidden;
        border: 2.5px solid {INK} !important;
    }}

    div[data-testid="stAlert"] {{
        border-radius: 14px !important;
        border: 2.5px solid {INK} !important;
        box-shadow: 3px 3px 0px {INK} !important;
        font-weight: 600 !important;
    }}

    [data-testid="stFileUploaderDropzone"] {{
        background: {GREEN_SOFT} !important;
        border: 2.5px dashed {GREEN_DARK} !important;
        border-radius: 16px !important;
    }}

    div.stButton > button {{
        background: {GREEN};
        color: #fff; border: 2.5px solid {INK}; border-radius: 12px; font-weight: 800;
        padding: 10px 22px; box-shadow: 3px 3px 0px {INK};
        transition: transform 0.1s ease;
    }}
    div.stButton > button:hover {{
        transform: translate(-1px, -1px);
        box-shadow: 4px 4px 0px {INK};
        color: #fff;
    }}

    div.stDownloadButton > button {{
        background: {INK};
        color: #fff; border: 2.5px solid {INK}; border-radius: 12px; font-weight: 800;
        box-shadow: 3px 3px 0px {GREEN};
    }}
    div.stDownloadButton > button:hover {{
        transform: translate(-1px, -1px);
        box-shadow: 4px 4px 0px {GREEN};
        color: #fff;
    }}

    div[data-baseweb="select"] > div {{
        border: 2px solid {INK} !important; border-radius: 10px !important;
    }}
    span[data-baseweb="tag"] {{
        background: {GREEN} !important; border-radius: 8px !important;
    }}

    hr {{ border-color: {INK} !important; opacity: 0.15; }}
    </style>
""", unsafe_allow_html=True)


def page_header(icon, title, breadcrumb, badge_html=""):
    html = (
        f'<div class="page-header-wrap">'
        f'<div class="page-header-left">'
        f'<div class="page-header-icon">{icon}</div>'
        f'<div><p class="page-header-title">{title}</p>'
        f'<p class="page-header-breadcrumb">{breadcrumb}</p></div>'
        f'</div><div>{badge_html}</div></div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def step_header(number, title, icon="", caption=""):
    """Header langkah yang rapi: badge nomor kotak + ikon + judul, menggantikan
    tumpukan emoji angka (1️⃣2️⃣...) yang berantakan. Dibangun sebagai SATU baris
    HTML tanpa indentasi/baris kosong -- kalau ditulis multi-baris dengan indentasi,
    dan salah satu placeholder (mis. caption) kosong, baris kosong itu memecah HTML
    jadi dua blok di mata parser Markdown, dan sisanya (termasuk tag penutup </div>)
    ikut dirender sebagai teks/code block mentah alih-alih HTML."""
    icon_html = f"{icon} " if icon else ""
    caption_html = f'<p class="step-caption">{caption}</p>' if caption else ""
    html = (
        f'<div class="step-header-row">'
        f'<span class="step-badge">{number}</span>'
        f'<div><p class="step-title">{icon_html}{title}</p>{caption_html}</div>'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def style_fig(fig, ax_list):
    fig.patch.set_facecolor("#FFFFFF")
    if not isinstance(ax_list, (list, np.ndarray)):
        ax_list = [ax_list]
    for ax in np.ravel(ax_list):
        ax.set_facecolor("#FFFFFF")
        ax.tick_params(colors=INK)
        ax.xaxis.label.set_color(INK)
        ax.yaxis.label.set_color(INK)
        ax.title.set_color(INK)
        for spine in ax.spines.values():
            spine.set_color(INK)
            spine.set_linewidth(1.4)


# =============================================================================
# HELPER: baca file upload (CSV / XLSX / XLS)
# =============================================================================
def _read_csv_smart(uploaded_file):
    encodings = [None, "latin-1"]
    for enc in encodings:
        try:
            uploaded_file.seek(0)
            kwargs = {"sep": None, "engine": "python"}
            if enc:
                kwargs["encoding"] = enc
            df = pd.read_csv(uploaded_file, **kwargs)
            if df.shape[1] > 1:
                return df
        except Exception:
            pass
    for delim in [";", ",", "\t", "|"]:
        for enc in encodings:
            try:
                uploaded_file.seek(0)
                kwargs = {"sep": delim}
                if enc:
                    kwargs["encoding"] = enc
                df = pd.read_csv(uploaded_file, **kwargs)
                if df.shape[1] > 1:
                    return df
            except Exception:
                pass
    uploaded_file.seek(0)
    try:
        return pd.read_csv(uploaded_file)
    except UnicodeDecodeError:
        uploaded_file.seek(0)
        return pd.read_csv(uploaded_file, encoding="latin-1")


def read_uploaded_table(uploaded_file):
    name = uploaded_file.name.lower()
    try:
        if name.endswith(".csv"):
            df = _read_csv_smart(uploaded_file)
            if df.shape[1] <= 1:
                return None, (
                    "File CSV berhasil dibuka tapi hanya terbaca sebagai **1 kolom**. "
                    "Kemungkinan delimiter-nya tidak dikenali (bukan koma/titik-koma/tab). "
                    "Buka file di Excel/Notepad dan cek pemisah kolomnya, lalu upload ulang."
                )
            return df, None
        elif name.endswith(".xlsx") or name.endswith(".xls"):
            if name.endswith(".xlsx") and not OPENPYXL_AVAILABLE:
                return None, ("Library `openpyxl` belum terpasang, dibutuhkan untuk membaca file .xlsx. "
                               "Install dengan: `pip install openpyxl`.")
            if name.endswith(".xls") and not XLRD_AVAILABLE:
                return None, ("Library `xlrd` belum terpasang, dibutuhkan untuk membaca file .xls lama. "
                               "Install dengan: `pip install xlrd`.")
            xls = pd.ExcelFile(uploaded_file)
            sheet_names = xls.sheet_names
            if len(sheet_names) > 1:
                sheet = st.selectbox(
                    "File Excel ini punya beberapa sheet, pilih salah satu:",
                    options=sheet_names, key="excel_sheet_choice"
                )
            else:
                sheet = sheet_names[0]
            df = xls.parse(sheet)
            return df, None
        else:
            return None, "Format file tidak didukung. Gunakan .csv, .xlsx, atau .xls."
    except Exception as e:
        return None, f"Gagal membaca file: {e}"


def _find_default_dataset_path():
    """Cari dataset default di beberapa lokasi umum, karena app.py biasanya
    ada di folder 'dashboard/' terpisah dari folder 'dataset/'."""
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(here, DEFAULT_DATASET_NAME),
        os.path.join(here, "dataset", DEFAULT_DATASET_NAME),
        os.path.join(here, "..", "dataset", DEFAULT_DATASET_NAME),
        os.path.join(here, "..", DEFAULT_DATASET_NAME),
        DEFAULT_DATASET_NAME,
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    for folder in [os.path.join(here, "dataset"), os.path.join(here, "..", "dataset")]:
        if os.path.isdir(folder):
            csvs = [f for f in os.listdir(folder) if f.lower().endswith(".csv")]
            if csvs:
                return os.path.join(folder, csvs[0])
    return None


@st.cache_data(show_spinner=False)
def load_default_dataset():
    path = _find_default_dataset_path()
    if path is None:
        return None
    try:
        df = pd.read_csv(path, sep=";")
        if df.shape[1] <= 1:
            df = pd.read_csv(path)
        return df
    except Exception:
        return None


# =============================================================================
# PIPELINE INTI — SATU-SATUNYA SUMBER KEBENARAN
# =============================================================================
# Urutan PERSIS sama dengan notebook final:
#   1) Buang baris tanpa target
#   2) Buang baris duplikat (berdasar fitur) -> cegah data leakage
#   3) Split 80:20 (stratify)
#   4) Filter usia 20-35 diterapkan TERPISAH ke train & test (SETELAH split)
#   5) Verifikasi tidak ada overlap fitur train-test (leakage check)
#   6) Handling missing value (median dari TRAIN, dipakai juga untuk TEST)
#   7) Normalisasi Min-Max (fit dari TRAIN, dipakai juga untuk TEST)
#   8) Label encoding manual ordinal (Low=0, Medium=1, High=2)
#   9) SMOTE (hanya data training)
#  10) Training Random Forest, SVM, XGBoost dengan hyperparameter identik
#      dengan notebook
#
# Dipakai oleh SEMUA halaman (Overview, Data Preprocessing, Model Evaluation,
# Upload & Re-Kalkulasi) -> hasil DIJAMIN identik kalau dataset yang dipakai
# sama persis.

def _build_models(class_names):
    """Definisi model + hyperparameter — SAMA PERSIS dipakai untuk varian
    dengan SMOTE maupun tanpa SMOTE, supaya perbandingan adil (apple-to-apple)."""
    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=100, criterion="gini", random_state=42, n_jobs=-1
        ),
        'SVM': SVC(kernel='rbf', probability=True, random_state=42),
    }
    if XGB_AVAILABLE:
        models['XGBoost'] = XGBClassifier(
            objective="multi:softmax",
            num_class=len(class_names),
            eval_metric='mlogloss',
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
    else:
        models['XGBoost (fallback: GradientBoosting)'] = GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42
        )
    return models


def _train_models_safe(X_train, y_train, X_test, y_test, class_names):
    """Sama seperti training biasa, tapi SETIAP model dibungkus try/except
    sendiri-sendiri — kalau 1 model gagal (misal dataset terlalu kecil / kelas
    terlalu sedikit untuk SVM), model lain tetap jalan dan errornya dilaporkan,
    bukan menjatuhkan seluruh pipeline. Labels confusion matrix & classification
    report di-paksa eksplisit (0..n_class-1) supaya bentuknya selalu konsisten
    dengan class_names walau ada kelas yang kebetulan tidak muncul di y_test."""
    models = _build_models(class_names)
    label_ids = list(range(len(class_names)))

    results = {}
    errors = {}
    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred, labels=label_ids)
            report = classification_report(
                y_test, y_pred, labels=label_ids, target_names=class_names,
                output_dict=True, zero_division=0
            )
            results[name] = {'accuracy': acc, 'cm': cm, 'y_pred': y_pred, 'report': report, 'model': model}
        except Exception as e:
            errors[name] = f"{type(e).__name__}: {e}"
    return results, errors


def _apply_smote(X_train_scaled, y_train):
    """Coba SMOTE dengan k_neighbors default (5), fallback ke k lebih kecil
    kalau kelas minoritas terlalu sedikit, dan fallback akhir: skip SMOTE sama
    sekali (tanpa membuat pipeline error) kalau tetap tidak bisa."""
    min_train_class = pd.Series(y_train).value_counts().min()

    if not SMOTE_AVAILABLE:
        return X_train_scaled, y_train, False, "Library `imbalanced-learn` tidak terpasang — SMOTE dilewati."
    if min_train_class < 2:
        return X_train_scaled, y_train, False, "Ada kelas dengan <2 sampel di data training — SMOTE tidak bisa dijalankan, dilewati."

    try:
        smote = SMOTE(random_state=42)
        X_bal, y_bal = smote.fit_resample(X_train_scaled, y_train)
        return X_bal, y_bal, True, "SMOTE berhasil diterapkan (k_neighbors default = 5)."
    except Exception:
        try:
            k = max(1, min(5, min_train_class - 1))
            smote = SMOTE(random_state=42, k_neighbors=k)
            X_bal, y_bal = smote.fit_resample(X_train_scaled, y_train)
            return X_bal, y_bal, True, f"SMOTE berhasil diterapkan dengan k_neighbors={k} (fallback, kelas minoritas kecil)."
        except Exception as e2:
            return X_train_scaled, y_train, False, f"SMOTE gagal dijalankan ({e2}) — dilewati, model dilatih dari data asli."


def _preprocess_core(df_raw: pd.DataFrame, feature_cols: list, target_col: str, age_filter: bool = True):
    """Tahap 1-8 pipeline (SEBELUM SMOTE & training). Dipisah dari training
    supaya bisa dipakai untuk menghasilkan DUA varian (dengan & tanpa SMOTE)
    dari hasil preprocessing yang identik — perbandingan jadi adil."""
    df = df_raw.copy()

    # 1) Buang baris tanpa target
    n_mentah = len(df)
    df = df.dropna(subset=[target_col])
    n_setelah_drop_target = len(df)

    # 2) Buang duplikat berdasar fitur (cegah leakage baris kembar train/test)
    n_dup = int(df[feature_cols].duplicated().sum())
    df = df.drop_duplicates(subset=feature_cols).reset_index(drop=True)
    n_setelah_dedup = len(df)

    if n_setelah_dedup < 10:
        raise ValueError(
            f"Data bersih tersisa cuma {n_setelah_dedup} baris (setelah buang target kosong & duplikat). "
            "Minimal butuh ±10 baris untuk bisa displit train/test dan dilatih."
        )

    X_all = df[feature_cols].copy()
    y_all = df[target_col].copy()

    if y_all.nunique(dropna=True) < 2:
        raise ValueError("Kolom target yang dipilih cuma punya 1 kategori unik — tidak bisa dipakai untuk klasifikasi.")

    # 3) Split 80:20 SEBELUM filter usia
    min_class_count = y_all.value_counts(dropna=False).min()
    can_stratify = min_class_count >= 2
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42,
        stratify=y_all if can_stratify else None
    )
    n_train_before_age = len(X_train_raw)
    n_test_before_age = len(X_test_raw)

    # 4) Filter usia 20-35, diterapkan TERPISAH ke masing-masing subset
    age_filter_applied = bool(age_filter and 'age' in X_train_raw.columns)
    if age_filter_applied:
        mask_train = (X_train_raw['age'] >= 20) & (X_train_raw['age'] <= 35)
        mask_test = (X_test_raw['age'] >= 20) & (X_test_raw['age'] <= 35)
        X_train_raw = X_train_raw[mask_train]; y_train_raw = y_train_raw[mask_train]
        X_test_raw = X_test_raw[mask_test]; y_test_raw = y_test_raw[mask_test]
    n_train_after_age = len(X_train_raw)
    n_test_after_age = len(X_test_raw)

    if n_train_after_age < 5 or n_test_after_age < 2:
        raise ValueError(
            f"Setelah split & filter usia, data tersisa terlalu sedikit (train={n_train_after_age}, "
            f"test={n_test_after_age}). Coba matikan filter usia atau gunakan dataset lebih besar."
        )

    # 5) Verifikasi tidak ada overlap fitur train-test
    train_tuples = set(map(tuple, np.round(X_train_raw[feature_cols].values.astype(float), 6)))
    test_tuples = list(map(tuple, np.round(X_test_raw[feature_cols].values.astype(float), 6)))
    n_overlap = sum(1 for t in test_tuples if t in train_tuples)

    # 6) Handling missing value (median dari TRAIN)
    X_train_imp = X_train_raw.copy()
    X_test_imp = X_test_raw.copy()
    missing_info = []
    for c in feature_cols:
        n_missing_train = int(X_train_imp[c].isna().sum())
        n_missing_test = int(X_test_imp[c].isna().sum())
        if n_missing_train > 0 or n_missing_test > 0:
            median_val = X_train_imp[c].median()
            if pd.isna(median_val):
                median_val = 0.0
            missing_info.append([c, n_missing_train, n_missing_test, round(float(median_val), 4)])
            X_train_imp[c] = X_train_imp[c].fillna(median_val)
            X_test_imp[c] = X_test_imp[c].fillna(median_val)

    # 7) Normalisasi Min-Max (fit dari TRAIN)
    scaler = MinMaxScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_imp), columns=feature_cols, index=X_train_imp.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test_imp), columns=feature_cols, index=X_test_imp.index)

    # 8) Label encoding — manual ordinal kalau kategorinya persis Low/Medium/High,
    #    fallback ke LabelEncoder alfabetis untuk dataset lain (nama kelas apa pun).
    if pd.api.types.is_numeric_dtype(y_all):
        y_train = y_train_raw.astype(int).values
        y_test = y_test_raw.astype(int).values
        class_names = [str(c) for c in sorted(pd.Series(np.concatenate([y_train, y_test])).unique())]
    else:
        y_train_str = y_train_raw.astype(str).str.strip()
        y_test_str = y_test_raw.astype(str).str.strip()
        combined_unique = set(y_train_str.unique()) | set(y_test_str.unique())
        if combined_unique.issubset(set(CLASS_ORDER)):
            label_map = {label: idx for idx, label in enumerate(CLASS_ORDER)}
            y_train = y_train_str.map(label_map).astype(int).values
            y_test = y_test_str.map(label_map).astype(int).values
            class_names = list(CLASS_ORDER)
        else:
            le = LabelEncoder()
            le.fit(pd.concat([y_train_str, y_test_str]))
            y_train = le.transform(y_train_str)
            y_test = le.transform(y_test_str)
            class_names = list(le.classes_)

    if pd.Series(y_train).nunique() < 2:
        raise ValueError("Data training hanya punya 1 kelas target setelah split/filter — coba matikan filter usia atau pakai dataset lain.")

    summary = {
        "n_mentah": n_mentah,
        "n_setelah_drop_target": n_setelah_drop_target,
        "n_dup": n_dup,
        "n_setelah_dedup": n_setelah_dedup,
        "n_train_before_age": n_train_before_age,
        "n_test_before_age": n_test_before_age,
        "n_train_after_age": n_train_after_age,
        "n_test_after_age": n_test_after_age,
        "age_filter_applied": age_filter_applied,
        "n_overlap": n_overlap,
        "missing_info": missing_info,
        "class_names": class_names,
        "class_dist_test": pd.Series(y_test).value_counts().sort_index().to_dict(),
        "n_train": len(X_train_scaled),
        "n_test": len(X_test_scaled),
    }
    return X_train_scaled, X_test_scaled, y_train, y_test, class_names, summary


@st.cache_data(show_spinner=False)
def run_pipeline_full(df_raw: pd.DataFrame, feature_cols: list, target_col: str, age_filter: bool = True):
    """Pipeline lengkap untuk DATASET APA PUN: preprocessing -> SMOTE -> training
    3 model, dijalankan DUA KALI (dengan SMOTE & tanpa SMOTE) dari hasil
    preprocessing yang identik, supaya keduanya bisa dibandingkan apple-to-apple.
    Mengembalikan dict berisi kedua varian + ringkasan lengkap tiap tahap."""
    X_train_scaled, X_test_scaled, y_train, y_test, class_names, summary = _preprocess_core(
        df_raw, feature_cols, target_col, age_filter
    )

    class_dist_before_smote = pd.Series(y_train).value_counts().sort_index().to_dict()
    X_train_bal, y_train_bal, smote_applied, smote_note = _apply_smote(X_train_scaled, y_train)
    class_dist_after_smote = pd.Series(y_train_bal).value_counts().sort_index().to_dict()

    results_nosmote, errors_nosmote = _train_models_safe(X_train_scaled, y_train, X_test_scaled, y_test, class_names)
    results_smote, errors_smote = _train_models_safe(X_train_bal, y_train_bal, X_test_scaled, y_test, class_names)

    summary.update({
        "class_dist_before_smote": class_dist_before_smote,
        "class_dist_after_smote": class_dist_after_smote,
        "smote_applied": smote_applied,
        "smote_note": smote_note,
        "n_train_smote": len(X_train_bal),
        "errors_nosmote": errors_nosmote,
        "errors_smote": errors_smote,
    })

    return {
        "results_smote": results_smote,
        "results_nosmote": results_nosmote,
        "summary": summary,
        "X_test": X_test_scaled,
        "y_test": y_test,
        "class_names": class_names,
    }


# =============================================================================
# PERHITUNGAN MANUAL (RF, SVM, XGBoost) — PERSIS SAMA DENGAN NOTEBOOK
# =============================================================================
# Mengambil model yang SUDAH dilatih (dari _train_models), lalu menghitung
# ulang prediksi untuk 2 sampel data test dari nol (tanpa .predict_proba),
# dibandingkan dengan output model asli.

def manual_calc_random_forest(rf_model, sample_X, sample_y, class_names):
    """predict_proba Random Forest = rata-rata predict_proba SETIAP pohon."""
    n_trees = len(rf_model.estimators_)
    n_samples = len(sample_X)
    n_class = len(class_names)

    proba_manual = np.zeros((n_samples, n_class))
    vote_count = np.zeros((n_samples, n_class))
    for tree in rf_model.estimators_:
        tree_proba = tree.predict_proba(sample_X.values)
        proba_manual += tree_proba
        tree_pred = tree.predict(sample_X.values)
        for i, cls in enumerate(tree_pred):
            vote_count[i, int(cls)] += 1
    proba_manual /= n_trees

    proba_model = rf_model.predict_proba(sample_X.values)
    selisih = proba_manual - proba_model
    pred_manual = np.argmax(proba_manual, axis=1)

    return {
        "n_trees": n_trees,
        "vote_count": vote_count,
        "proba_manual": proba_manual,
        "proba_model": proba_model,
        "selisih": selisih,
        "pred_manual": pred_manual,
    }


def _rbf_kernel_row(x, SV, gamma):
    diff = SV - x
    sq = np.sum(diff ** 2, axis=1)
    return np.exp(-gamma * sq)


def _multiclass_probability(k, r):
    """Algoritma pairwise coupling Wu-Lin-Weng (persis dipakai libsvm)."""
    Q = np.zeros((k, k))
    for t in range(k):
        for j in range(k):
            if j != t:
                Q[t, t] += r[j, t] ** 2
                Q[t, j] = -r[j, t] * r[t, j]
    p = np.ones(k) / k
    eps = 0.005 / k
    for _ in range(100):
        Qp = Q.dot(p)
        pQp = p.dot(Qp)
        max_error = np.max(np.abs(Qp - pQp))
        if max_error < eps:
            break
        for t in range(k):
            diff = (-Qp[t] + pQp) / Q[t, t]
            p[t] += diff
            pQp = (pQp + diff * (diff * Q[t, t] + 2 * Qp[t])) / (1 + diff) ** 2
            Qp = (Qp + diff * Q[:, t]) / (1 + diff)
            p = p / p.sum()
    return p


def manual_calc_svm(svm_model, sample_X, sample_y, class_names):
    """Kernel RBF manual -> decision function OvO manual -> Platt scaling
    manual (pairwise coupling, algoritma Wu-Lin-Weng) -> probabilitas akhir."""
    gamma = svm_model._gamma
    sv = svm_model.support_vectors_
    dual_coef = svm_model.dual_coef_
    intercept = svm_model.intercept_
    n_support = svm_model.n_support_
    n_classes = len(class_names)
    starts = np.concatenate(([0], np.cumsum(n_support)))
    probA = svm_model.probA_
    probB = svm_model.probB_

    n_samples = len(sample_X)
    proba_manual = np.zeros((n_samples, n_classes))
    pred_manual = []
    decision_all = []
    pair_labels_all = None
    votes_all = []

    for s_i in range(n_samples):
        x = sample_X.values[s_i]
        K = _rbf_kernel_row(x, sv, gamma)

        decision_vals, pair_labels = [], []
        p = 0
        for i in range(n_classes):
            for j in range(i + 1, n_classes):
                coef1 = dual_coef[j - 1, starts[i]:starts[i + 1]]
                coef2 = dual_coef[i, starts[j]:starts[j + 1]]
                k1 = K[starts[i]:starts[i + 1]]
                k2 = K[starts[j]:starts[j + 1]]
                val = np.dot(coef1, k1) + np.dot(coef2, k2) + intercept[p]
                decision_vals.append(val)
                pair_labels.append((i, j))
                p += 1
        decision_vals = np.array(decision_vals)
        decision_all.append(decision_vals)
        pair_labels_all = pair_labels

        votes = np.zeros(n_classes)
        for (i, j), val in zip(pair_labels, decision_vals):
            if val > 0:
                votes[i] += 1
            else:
                votes[j] += 1
        votes_all.append(votes)

        r = np.zeros((n_classes, n_classes))
        p = 0
        for i in range(n_classes):
            for j in range(i + 1, n_classes):
                f = decision_vals[p]
                # Kasus khusus 2 kelas: konvensi tanda internal libsvm untuk
                # probA_/probB_ (Platt scaling) TERBALIK relatif terhadap tanda
                # publik decision_function() ketika hanya ada 1 pasangan kelas
                # (n_classes==2). Untuk OvO n_classes>=3 tandanya sudah konsisten
                # (diverifikasi cocok persis dengan predict_proba), jadi pembalikan
                # ini hanya diterapkan pada kasus biner.
                f_for_prob = -f if n_classes == 2 else f
                rij = 1.0 / (1.0 + np.exp(probA[p] * f_for_prob + probB[p]))
                r[i, j] = rij
                r[j, i] = 1.0 - rij
                p += 1

        p_manual = _multiclass_probability(n_classes, r)
        proba_manual[s_i] = p_manual
        pred_manual.append(int(np.argmax(p_manual)))

    proba_model = svm_model.predict_proba(sample_X.values)
    selisih = proba_manual - proba_model

    return {
        "decision_all": decision_all,
        "pair_labels": pair_labels_all,
        "votes_all": votes_all,
        "proba_manual": proba_manual,
        "proba_model": proba_model,
        "selisih": selisih,
        "pred_manual": np.array(pred_manual),
    }


def manual_calc_xgboost(xgb_model, sample_X, sample_y, class_names):
    """Raw margin (log-odds) -> softmax manual -> probabilitas akhir."""
    n_class = len(class_names)
    raw_margin = xgb_model.predict(sample_X.values, output_margin=True)
    raw_margin = np.asarray(raw_margin).reshape(len(sample_X), n_class)

    proba_manual = np.zeros((len(sample_X), n_class))
    exp_list, sigma_list = [], []
    for i in range(len(sample_X)):
        z = raw_margin[i]
        exp_z = np.exp(z)
        sigma = exp_z.sum()
        proba_manual[i] = exp_z / sigma
        exp_list.append(exp_z)
        sigma_list.append(sigma)

    proba_model = xgb_model.predict_proba(sample_X.values)
    selisih = proba_manual - proba_model
    pred_manual = np.argmax(proba_manual, axis=1)

    return {
        "raw_margin": raw_margin,
        "exp_list": exp_list,
        "sigma_list": sigma_list,
        "proba_manual": proba_manual,
        "proba_model": proba_model,
        "selisih": selisih,
        "pred_manual": pred_manual,
    }



# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.markdown(
        '<div class="sidebar-logo-box"><div class="logo-mark">CS</div>'
        '<h2>Coding Stress<br>ML Studio</h2>'
        '<p>Machine Learning Classification Platform</p></div>',
        unsafe_allow_html=True
    )

    st.markdown('<p class="sidebar-section-label">🗺️ Navigation</p>', unsafe_allow_html=True)
    menu = st.sidebar.radio(
        label="Pilih Halaman:",
        options=[
            "🏠 Dashboard Overview",
            "⚙️ Data Preprocessing",
            "📊 Model Evaluation",
            "📤 Upload & Re-Kalkulasi"
        ]
    )

    st.markdown(
        "<div class='sidebar-footer'><b>Peneliti:</b> Ervin<br>"
        "<b>Skripsi:</b> ITB AD Jakarta<br>© 2026 All Rights Reserved</div>",
        unsafe_allow_html=True
    )

# Muat dataset default & jalankan pipeline SEKALI di awal (di-cache),
# dipakai oleh Overview, Data Preprocessing, dan Model Evaluation.
# (Pipeline yang sama, run_pipeline_full, juga dipakai di halaman Upload &
# Re-Kalkulasi untuk dataset APA PUN yang diupload user.)
default_df = load_default_dataset()
default_pipeline_ok = False
default_pipeline_error = None

if default_df is not None:
    try:
        default_full = run_pipeline_full(default_df, FEATURE_COLUMNS, TARGET_COLUMN, age_filter=True)
        default_results = default_full['results_smote']
        default_summary = default_full['summary']
        default_X_test = default_full['X_test']
        default_y_test = default_full['y_test']
        if not default_results:
            default_pipeline_error = "Semua model gagal dilatih pada dataset default. Detail error: " + str(default_summary.get('errors_smote'))
        else:
            default_pipeline_ok = True
            default_df_komp = pd.DataFrame({
                'Algoritma': list(default_results.keys()),
                'Akurasi (%)': [round(v['accuracy'] * 100, 2) for v in default_results.values()]
            }).sort_values('Akurasi (%)', ascending=False).reset_index(drop=True)
    except Exception as e:
        default_pipeline_error = f"{type(e).__name__}: {e}"

# =============================================================================
# HALAMAN 1: DASHBOARD OVERVIEW
# =============================================================================
if menu == "🏠 Dashboard Overview":
    page_header(
        "🏠", "Dashboard Overview", "Admin / Ringkasan",
        badge_html='<span class="badge badge-green">Executive Summary</span>'
    )

    if not default_pipeline_ok:
        if default_df is None:
            st.error(
                f"File dataset default `{DEFAULT_DATASET_NAME}` tidak ditemukan. Taruh file tersebut di folder "
                "`dataset/` satu level di atas folder `dashboard/` (atau di folder yang sama dengan app.py), "
                "atau gunakan menu '📤 Upload & Re-Kalkulasi' untuk pakai dataset lain."
            )
        else:
            st.error(f"Pipeline gagal dijalankan pada dataset default. Detail: {default_pipeline_error}")
    else:
        best_model = default_df_komp.iloc[0]['Algoritma']
        best_acc = default_df_komp.iloc[0]['Akurasi (%)']

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Total Data Bersih", f"{default_summary['n_setelah_dedup']:,}", "baris (setelah dedup)")
        with col_b:
            st.metric("Model Terbaik", best_model, f"{best_acc:.2f}% akurasi")
        with col_c:
            st.metric("Jumlah Fitur", f"{len(FEATURE_COLUMNS)}", "atribut numerik")

        st.caption(
            "✅ Angka di atas dihitung LANGSUNG dari `developer_burnout_dataset_ervin.csv` memakai pipeline "
            "yang SAMA PERSIS dengan halaman Upload & Re-Kalkulasi dan notebook Jupyter (dedup → split → "
            "filter usia per-subset → missing value → normalisasi → encoding → SMOTE)."
        )

        st.write("")
        with st.container(border=True):
            st.subheader("📋 Ringkasan Tahapan Pipeline")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Data Mentah", f"{default_summary['n_mentah']:,}")
            c2.metric("Setelah Buang Target Kosong", f"{default_summary['n_setelah_drop_target']:,}")
            c3.metric("Duplikat Dibuang", f"{default_summary['n_dup']:,}")
            c4.metric("Data Bersih Final", f"{default_summary['n_setelah_dedup']:,}")

            c5, c6, c7 = st.columns(3)
            c5.metric("Data Training (setelah filter usia)", f"{default_summary['n_train_after_age']:,}")
            c6.metric("Data Testing (setelah filter usia)", f"{default_summary['n_test_after_age']:,}")
            c7.metric("Overlap Train-Test", f"{default_summary['n_overlap']}", "harus 0")

        st.write("")
        with st.container(border=True):
            st.subheader("📋 Ringkasan Topik Penelitian")
            st.markdown("""
                Sistem ini dirancang sebagai platform analitis interaktif untuk mengukur dan membandingkan performa algoritma
                **Random Forest, Support Vector Machine (SVM), dan XGBoost** dalam mendeteksi tingkatan *Coding Stress* pada mahasiswa informatika.

                Gunakan menu **Sidebar di sebelah kiri** untuk menavigasi alur metodologi penelitian, mulai dari pemrosesan data (preprocessing),
                evaluasi komparatif performa model, hingga fitur **upload dataset baru** untuk menghitung ulang seluruh pipeline secara langsung.
            """)
            st.info("💡 **Petunjuk:** Buka menu '📤 Upload & Re-Kalkulasi' untuk mencoba pipeline lengkap dengan data Anda sendiri. Jika file yang diupload sama persis dengan dataset default, hasilnya akan identik dengan halaman ini.")

# =============================================================================
# HALAMAN 2: DATA PREPROCESSING
# =============================================================================
elif menu == "⚙️ Data Preprocessing":
    page_header(
        "⚙️", "Analisis Pra-Pemrosesan Data", "Admin / Preprocessing",
        badge_html='<span class="badge badge-amber">Data Transformation Workflow</span>'
    )

    if not default_pipeline_ok:
        if default_df is None:
            st.error(f"File dataset default `{DEFAULT_DATASET_NAME}` tidak ditemukan.")
        else:
            st.error(f"Pipeline gagal dijalankan pada dataset default. Detail: {default_pipeline_error}")
        st.stop()

    s = default_summary

    # Urutan tab SUDAH DIPERBAIKI mengikuti urutan pipeline yang benar:
    # Bersihkan & Dedup -> Split & Filter Usia -> Missing Value -> Normalisasi
    # -> Label Encoding -> SMOTE. (Sebelumnya salah: age filter digabung
    # dengan missing value dan dilakukan SEBELUM split.)
    tab_clean, tab_split, tab_missing, tab_norm, tab_encode, tab_smote = st.tabs([
        "🧹 Bersihkan & Duplikat",
        "✂️ Split & Filter Usia",
        "❓ Missing Value",
        "⚖️ Normalisasi Data",
        "🏷️ Label Encoding",
        "🧬 SMOTE Balance"
    ])

    with tab_clean:
        with st.container(border=True):
            st.subheader("📌 Pembersihan Data: Target Kosong & Duplikat")
            st.markdown(
                "Baris tanpa label target dibuang lebih dulu (wajib, karena `train_test_split(stratify=y)` "
                "akan error jika `y` mengandung NaN). Baris duplikat (fitur sama persis) juga dibuang **sebelum split** "
                "untuk mencegah baris kembar bocor antara data training dan testing (data leakage)."
            )
            fig_vol, ax_vol = plt.subplots(figsize=(7, 4))
            tahapan_data = ['Data Mentah', 'Hapus Target\nKosong', 'Hapus\nDuplikat']
            volume_data = [s['n_mentah'], s['n_setelah_drop_target'], s['n_setelah_dedup']]
            sns.barplot(x=tahapan_data, y=volume_data, palette=PALETTE_MAIN[:3], ax=ax_vol, hue=tahapan_data, legend=False, edgecolor=INK, linewidth=1.2)
            ax_vol.set_ylabel('Jumlah Baris Data', fontweight='bold')
            for p in ax_vol.patches:
                ax_vol.annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2., p.get_height() + 100),
                                 ha='center', fontweight='bold', fontsize=9, color=INK)
            style_fig(fig_vol, ax_vol)
            sns.despine(left=True, bottom=True)
            plt.tight_layout()
            st.pyplot(fig_vol)
            plt.close(fig_vol)

            if s['n_dup'] > 0:
                st.warning(f"⚠️ Ditemukan **{s['n_dup']} baris duplikat** (fitur sama persis) — sudah dibuang.")
            else:
                st.success("✅ Tidak ada baris duplikat pada dataset ini.")
            st.success(f"✅ **Status Audit:** {s['n_mentah']:,} baris mentah → buang target kosong ({s['n_setelah_drop_target']:,}) → buang duplikat → **{s['n_setelah_dedup']:,} data bersih**.")

    with tab_split:
        with st.container(border=True):
            st.subheader(f"📌 Split Data (80:20) — DILAKUKAN DULU, Baru Filter Usia")
            st.markdown(
                "Split 80:20 dilakukan **lebih dulu** (memakai *stratified sampling* dari seluruh data bersih), "
                "baru filter usia 20–35 tahun diterapkan **secara terpisah** ke masing-masing subset train & test. "
                "Ini aman dari data leakage karena usia adalah atribut per-baris (tidak bergantung statistik lintas-baris)."
            )

            col1, col2 = st.columns(2)
            with col1:
                st.write("#### Sebelum Filter Usia")
                fig1, ax1 = plt.subplots(figsize=(5.5, 4))
                ax1.bar(['Train', 'Test'], [s['n_train_before_age'], s['n_test_before_age']],
                        color=[PALETTE_MAIN[0], PALETTE_MAIN[3]], edgecolor=INK, linewidth=1.2)
                for i, v in enumerate([s['n_train_before_age'], s['n_test_before_age']]):
                    ax1.annotate(str(v), (i, v), ha='center', va='bottom', fontweight='bold')
                ax1.set_ylabel('Jumlah Baris')
                style_fig(fig1, ax1)
                plt.tight_layout()
                st.pyplot(fig1)
                plt.close(fig1)

            with col2:
                st.write("#### Setelah Filter Usia 20-35")
                fig2, ax2 = plt.subplots(figsize=(5.5, 4))
                ax2.bar(['Train', 'Test'], [s['n_train_after_age'], s['n_test_after_age']],
                        color=[PALETTE_MAIN[0], PALETTE_MAIN[3]], edgecolor=INK, linewidth=1.2)
                for i, v in enumerate([s['n_train_after_age'], s['n_test_after_age']]):
                    ax2.annotate(str(v), (i, v), ha='center', va='bottom', fontweight='bold')
                ax2.set_ylabel('Jumlah Baris')
                style_fig(fig2, ax2)
                plt.tight_layout()
                st.pyplot(fig2)
                plt.close(fig2)

            c1, c2, c3 = st.columns(3)
            c1.metric("Data Training Final", s['n_train_after_age'])
            c2.metric("Data Testing Final", s['n_test_after_age'])
            c3.metric("Overlap Fitur Train-Test", s['n_overlap'], "harus 0 (aman)")

            if s['n_overlap'] == 0:
                st.success("✅ Tidak ada overlap fitur antara data training dan testing — aman dari leakage duplikasi baris.")
            else:
                st.error(f"🚨 Ditemukan {s['n_overlap']} baris testing yang fiturnya identik dengan data training!")

            st.write("")
            st.write("#### Distribusi Kelas: Training vs Testing (Setelah Filter Usia)")
            class_names = s['class_names']
            before = s['class_dist_before_smote']
            test_dist = s['class_dist_test']
            df_strat = pd.DataFrame({
                'Kelas': [class_names[k] if isinstance(k, (int, np.integer)) and k < len(class_names) else str(k) for k in before.keys()] +
                         [class_names[k] if isinstance(k, (int, np.integer)) and k < len(class_names) else str(k) for k in test_dist.keys()],
                'Jumlah': list(before.values()) + list(test_dist.values()),
                'Subset': ['Training'] * len(before) + ['Testing'] * len(test_dist)
            })
            fig3, ax3 = plt.subplots(figsize=(7, 4.2))
            sns.barplot(x='Kelas', y='Jumlah', hue='Subset', data=df_strat,
                        palette=[PALETTE_MAIN[0], PALETTE_MAIN[3]], ax=ax3, edgecolor=INK, linewidth=1.0)
            for p in ax3.patches:
                if p.get_height() > 0:
                    ax3.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width()/2., p.get_height()),
                                 ha='center', va='baseline', xytext=(0, 5), textcoords='offset points',
                                 fontsize=9, fontweight='bold', color=INK)
            style_fig(fig3, ax3)
            plt.tight_layout()
            st.pyplot(fig3)
            plt.close(fig3)

    with tab_missing:
        with st.container(border=True):
            st.subheader("📌 Handling Missing Value (Imputasi Median)")
            st.markdown(
                "Dilakukan **setelah** split & filter usia. Median dihitung HANYA dari data training "
                "(yang sudah difilter usia), lalu dipakai juga untuk mengisi nilai kosong di data testing."
            )
            missing_info = s['missing_info']
            if missing_info:
                df_missing = pd.DataFrame(missing_info, columns=["Fitur", "Kosong (Train)", "Kosong (Test)", "Median Training"])
                st.dataframe(df_missing, use_container_width=True, hide_index=True)

                fig, ax = plt.subplots(figsize=(8.5, 4.2))
                x_idx = np.arange(len(df_missing))
                width = 0.35
                ax.bar(x_idx - width/2, df_missing["Kosong (Train)"], width, label="Train", color=PALETTE_MAIN[0], edgecolor=INK, linewidth=1.0)
                ax.bar(x_idx + width/2, df_missing["Kosong (Test)"], width, label="Test", color=PALETTE_MAIN[3], edgecolor=INK, linewidth=1.0)
                ax.set_xticks(x_idx)
                ax.set_xticklabels(df_missing["Fitur"], rotation=45, ha="right")
                ax.set_title("Jumlah Missing Value per Fitur (Sebelum Imputasi)", fontweight="bold")
                ax.legend()
                style_fig(fig, ax)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.info("Tidak ada nilai kosong pada kolom fitur setelah split & filter usia.")

    with tab_norm:
        with st.container(border=True):
            st.subheader("📌 Penyetaraan Rentang Data Fitur (Min-Max Scaling)")
            st.markdown("Mentransformasikan seluruh variabel numerik ke rentang seragam [0, 1], di-fit dari data training saja.")
            fig_norm, ax_norm = plt.subplots(figsize=(10, 6))
            sns.boxplot(data=default_X_test, orient="h", palette=PALETTE_MAIN, ax=ax_norm)
            ax_norm.set_title("Hasil Normalisasi Seluruh Fitur pada Data Test (Min-Max Scaling, Rentang 0.0–1.0)", fontsize=11, fontweight='bold', pad=10)
            ax_norm.set_xlabel('Nilai Skala Hasil Normalisasi', fontsize=10)
            ax_norm.set_ylabel('Fitur / Atribut Dataset', fontsize=10)
            style_fig(fig_norm, ax_norm)
            plt.tight_layout()
            st.pyplot(fig_norm)
            plt.close(fig_norm)
            st.success("✅ Seluruh dimensi fitur berhasil dipetakan ke skala [0, 1].")

    with tab_encode:
        with st.container(border=True):
            st.subheader("📌 Label Encoding Target (Manual, Ordinal)")
            st.markdown(
                "Variabel target (`Low`, `Medium`, `High`) dikonversi memakai **mapping manual** "
                "(BUKAN `LabelEncoder` biasa, karena itu akan mengurutkan alfabetis: High=0, Low=1, Medium=2 — salah urutan). "
                "Dengan mapping manual, urutannya benar sesuai tingkat keparahan: **Low=0, Medium=1, High=2**."
            )
            col_enc1, col_enc2 = st.columns([1, 1.3])
            with col_enc1:
                st.write("#### 🗺️ Tabel Mapping Variabel Target")
                mapping_data = {'Kategori Asli': CLASS_ORDER, 'Nilai Encode': [0, 1, 2]}
                st.dataframe(pd.DataFrame(mapping_data), use_container_width=True, hide_index=True)
            with col_enc2:
                before = s['class_dist_before_smote']
                class_names = s['class_names']
                labels = [class_names[k] if isinstance(k, (int, np.integer)) and k < len(class_names) else str(k) for k in before.keys()]
                fig_le, ax_le = plt.subplots(figsize=(6.5, 4.5))
                ax_le.bar(labels, list(before.values()), color=PALETTE_MAIN[:len(before)], edgecolor=INK, linewidth=1.0)
                ax_le.set_title("Distribusi Kelas Target Data Latih (Pre-SMOTE / Imbalance)", fontsize=11, fontweight='bold', pad=15)
                ax_le.set_xlabel('Tingkat Stres Koding', fontsize=10)
                ax_le.set_ylabel('Jumlah Sampel Mahasiswa', fontsize=10)
                for p in ax_le.patches:
                    ax_le.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width()/2., p.get_height()),
                                   ha='center', va='baseline', xytext=(0, 5), textcoords='offset points',
                                   fontsize=9, fontweight='bold', color=INK)
                style_fig(fig_le, ax_le)
                plt.tight_layout()
                st.pyplot(fig_le)
                plt.close(fig_le)

    with tab_smote:
        with st.container(border=True):
            st.subheader("🧬 Penyeimbangan Kelas Menggunakan Teknik SMOTE")
            st.markdown("Algoritma *Synthetic Minority Over-sampling Technique* (SMOTE) diaplikasikan **hanya pada subset data training**.")

            before = s['class_dist_before_smote']
            after = s['class_dist_after_smote']
            class_names = s['class_names']
            labels = [class_names[k] if isinstance(k, (int, np.integer)) and k < len(class_names) else str(k) for k in before.keys()]

            df_before_st = pd.DataFrame({'Kelas': labels, 'Jumlah': list(before.values()), 'Kondisi': ['Sebelum SMOTE'] * len(before)})
            df_after_st = pd.DataFrame({'Kelas': labels, 'Jumlah': list(after.values()), 'Kondisi': ['Setelah SMOTE'] * len(after)})
            df_compare_st = pd.concat([df_before_st, df_after_st])

            fig_smote, ax_smote = plt.subplots(figsize=(8, 4.2))
            sns.barplot(x='Kelas', y='Jumlah', hue='Kondisi', data=df_compare_st,
                        palette=[PALETTE_MAIN[0], PALETTE_MAIN[3]], ax=ax_smote, edgecolor=INK, linewidth=1.0)
            for p in ax_smote.patches:
                if p.get_height() > 0:
                    ax_smote.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width()/2., p.get_height()),
                                      ha='center', va='baseline', xytext=(0, 5), textcoords='offset points',
                                      fontsize=9, fontweight='bold', color=INK)
            ax_smote.set_title("Perbandingan Distribusi Kelas Sebelum & Sesudah SMOTE (Data Training)", fontsize=11, fontweight='bold', pad=15)
            ax_smote.set_ylabel('Volume Sampel Data Latih')
            style_fig(fig_smote, ax_smote)
            plt.tight_layout()
            st.pyplot(fig_smote)
            plt.close(fig_smote)
            st.success(f"✨ **Hasil Akhir SMOTE:** Data latih kini seimbang, masing-masing kategori {max(after.values())} sampel (total {sum(after.values())} baris).")

# =============================================================================
# HALAMAN 3: MODEL EVALUATION
# =============================================================================
elif menu == "📊 Model Evaluation":
    page_header(
        "📊", "Evaluasi Komparatif Performa Model", "Admin / Evaluasi",
        badge_html='<span class="badge badge-green">Validation Results</span>'
    )

    if not default_pipeline_ok:
        if default_df is None:
            st.error(f"File dataset default `{DEFAULT_DATASET_NAME}` tidak ditemukan.")
        else:
            st.error(f"Pipeline gagal dijalankan pada dataset default. Detail: {default_pipeline_error}")
        st.stop()

    df_komp = default_df_komp
    best_model = df_komp.iloc[0]['Algoritma']
    best_acc = df_komp.iloc[0]['Akurasi (%)']

    with st.container(border=True):
        st.write("### 📈 Ringkasan Akurasi Global")
        col_glob1, col_glob2 = st.columns([1, 1.8])
        with col_glob1:
            st.write("#### 🏆 Algoritma Terbaik")
            st.metric(label="Akurasi Tertinggi", value=f"{best_acc:.2f}%", delta=f"Rekomendasi: {best_model}")
            st.write("#### 📋 Tabel Perbandingan Skor")
            st.dataframe(df_komp, use_container_width=True, hide_index=True)
        with col_glob2:
            st.write("#### 📉 Grafik Batang Akurasi Komparatif")
            fig_perf, ax_perf = plt.subplots(figsize=(7, 3.8))
            sns.barplot(x='Algoritma', y='Akurasi (%)', data=df_komp, palette=PALETTE_MAIN[:len(df_komp)], ax=ax_perf, hue='Algoritma', legend=False, edgecolor=INK, linewidth=1.2)
            ax_perf.set_ylim(0, 100)
            ax_perf.set_ylabel('Akurasi (%)')
            for p in ax_perf.patches:
                height = p.get_height()
                if height > 0:
                    ax_perf.annotate(f"{height:.2f}%", (p.get_x() + p.get_width() / 2., height - 8),
                                      ha='center', va='center', color='white', fontweight='bold', fontsize=11)
            style_fig(fig_perf, ax_perf)
            sns.despine(left=True, bottom=True)
            plt.tight_layout()
            st.pyplot(fig_perf)
            plt.close(fig_perf)

    st.write("")
    st.write(f"### 🔍 Detail Hasil Pengujian per Algoritma ({default_summary['n_test']} Data Uji)")
    tabs = st.tabs([f"{MODEL_ICONS.get(name, chr(0x1F4CC))} {name}" for name in default_results.keys()])
    for tab, (name, res) in zip(tabs, default_results.items()):
        with tab:
            col1, col2 = st.columns([1.2, 1])
            with col1:
                st.write("**Confusion Matrix**")
                fig_cm, ax_cm = plt.subplots(figsize=(5.5, 4.2))
                sns.heatmap(res['cm'], annot=True, fmt='d', cmap="Greens",
                            xticklabels=default_summary['class_names'], yticklabels=default_summary['class_names'],
                            ax=ax_cm, annot_kws={"size": 10, "weight": "bold"}, cbar=False, linewidths=1.2, linecolor=INK)
                ax_cm.set_xlabel('Prediksi Model', fontweight='bold')
                ax_cm.set_ylabel('Aktual (Data Asli)', fontweight='bold')
                style_fig(fig_cm, ax_cm)
                plt.tight_layout()
                st.pyplot(fig_cm)
                plt.close(fig_cm)
            with col2:
                st.write("**Metrik Klasifikasi Detail**")
                st.metric(label=f"Akurasi {name}", value=f"{res['accuracy']*100:.2f}%")
                report_df = pd.DataFrame(res['report']).transpose()
                report_df = report_df[~report_df.index.isin(['accuracy'])]
                st.dataframe(report_df.round(3), use_container_width=True)

    st.write("")
    st.success(
        f"💡 **Kesimpulan Analisis Skripsi:** Berdasarkan hasil pengujian nyata di atas, model **{best_model}** "
        f"menunjukkan kapabilitas paling optimal dalam mengklasifikasikan skala stres koding mahasiswa dengan akurasi sebesar **{best_acc:.2f}%**."
    )


# =============================================================================
# HALAMAN 4: UPLOAD DATA & RE-KALKULASI (SEMUA DATASET, CSV/XLSX/XLS)
# =============================================================================
# Halaman ini adalah satu-satunya tempat untuk:
#  - Menghitung ulang pipeline lengkap untuk DATASET APA PUN (bukan cuma
#    dataset skripsi default) — kolom fitur/target dikonfigurasi manual.
#  - Membandingkan hasil DENGAN SMOTE vs TANPA SMOTE (dua-duanya dilatih dari
#    hasil preprocessing yang identik, jadi perbandingannya adil).
#  - Perhitungan manual (Random Forest, SVM, XGBoost) untuk 2 sampel data
#    test — dihitung ulang dari nol lalu dicocokkan ke output asli model.
elif menu == "📤 Upload & Re-Kalkulasi":
    page_header(
        "📤", "Upload Data & Re-Kalkulasi Pipeline", "Admin / Live Recompute",
        badge_html='<span class="badge badge-amber">Live Recompute — Semua Dataset</span>'
    )

    with st.container(border=True):
        step_header(1, "Upload Dataset (.csv / .xlsx / .xls)", "📤")
        st.markdown(
            "Upload dataset **apa pun** (tidak harus dataset skripsi ini) untuk menjalankan ulang seluruh pipeline: "
            "buang target kosong → buang duplikat → split 80:20 → filter usia opsional (per-subset) → missing value "
            "→ normalisasi → label encoding → **training 2 varian: dengan SMOTE & tanpa SMOTE** → evaluasi lengkap "
            "→ **perhitungan manual** (RF/SVM/XGBoost) untuk 2 sampel data.\n\n"
            "Kolom fitur & target bisa dipilih bebas sesuai kolom yang ada di file Anda — tidak harus memakai nama "
            "kolom skripsi ini (`age`, `burnout_level`, dst)."
        )
        uploaded_file = st.file_uploader(
            "Pilih file CSV atau Excel",
            type=["csv", "xlsx", "xls"],
            help="Mendukung .csv, .xlsx (Excel 2007+), dan .xls (Excel lama)."
        )
        with st.expander("ℹ️ Contoh format kolom dataset skripsi (opsional, dataset lain juga bisa)"):
            st.code(
                "Fitur numerik : " + ", ".join(FEATURE_COLUMNS) + "\n"
                "Kolom target  : " + TARGET_COLUMN + " (nilai: Low / Medium / High, atau 0/1/2)",
                language="text"
            )
        if not OPENPYXL_AVAILABLE:
            st.warning("⚠️ Library `openpyxl` belum terpasang, sehingga upload **.xlsx** akan gagal. Jalankan `pip install openpyxl`.")
        if not XLRD_AVAILABLE:
            st.caption("ℹ️ Library `xlrd` belum terpasang — upload file **.xls** lama belum bisa diproses. Jalankan `pip install xlrd` bila perlu.")
        if not SMOTE_AVAILABLE:
            st.caption("ℹ️ Library `imbalanced-learn` belum terpasang — varian 'Dengan SMOTE' tidak akan tersedia. Jalankan `pip install imbalanced-learn` bila perlu.")

    if uploaded_file is None:
        st.info("⬆️ Silakan unggah file CSV atau Excel terlebih dahulu untuk memulai perhitungan ulang.")
        st.stop()

    # reset hasil lama kalau ganti file
    if st.session_state.get("last_uploaded_name") != uploaded_file.name:
        for k in ["ml_full", "ml_feature_cols", "ml_target_col", "excel_sheet_choice"]:
            st.session_state.pop(k, None)
        st.session_state["last_uploaded_name"] = uploaded_file.name

    df_raw, read_error = read_uploaded_table(uploaded_file)
    if read_error:
        st.error(read_error)
        st.stop()
    if df_raw is None or df_raw.empty:
        st.error("File berhasil dibaca tetapi tidak berisi data.")
        st.stop()

    with st.container(border=True):
        step_header(2, "Pratinjau Data Mentah", "📄")
        st.dataframe(df_raw.head(10), use_container_width=True)
        st.caption(f"Total baris: {len(df_raw):,} | Total kolom: {len(df_raw.columns)} | Sumber: {uploaded_file.name}")

    with st.container(border=True):
        step_header(3, "Konfigurasi Kolom", "🛠️")
        col_conf1, col_conf2 = st.columns(2)
        all_columns = list(df_raw.columns)
        numeric_columns = list(df_raw.select_dtypes(include=[np.number]).columns)

        if len(numeric_columns) == 0:
            st.error(
                "Tidak ada kolom numerik yang terdeteksi di file ini. Cek lagi apakah delimiter file sudah "
                "terbaca dengan benar di tabel pratinjau di atas (kalau kolom malah tergabung jadi satu, "
                "berarti delimiter belum cocok)."
            )
            st.stop()

        with col_conf1:
            target_col = st.selectbox(
                "Pilih kolom target (label yang mau diklasifikasikan):",
                options=all_columns,
                index=all_columns.index(TARGET_COLUMN) if TARGET_COLUMN in all_columns else 0
            )
        with col_conf2:
            default_features = [c for c in FEATURE_COLUMNS if c in numeric_columns] or \
                                [c for c in numeric_columns if c != target_col]
            feature_cols = st.multiselect(
                "Pilih kolom fitur numerik (input model):",
                options=[c for c in numeric_columns if c != target_col],
                default=default_features
            )

        has_age = 'age' in numeric_columns
        if has_age:
            age_filter = st.checkbox("Terapkan filter usia 20–35 tahun (kolom 'age' terdeteksi)", value=True)
        else:
            age_filter = False
            st.caption("ℹ️ Kolom 'age' tidak ditemukan di dataset ini — filter usia otomatis dilewati.")

        run_button = st.button("🚀 Jalankan Ulang Pipeline & Latih Model", use_container_width=True)

    if run_button:
        if not feature_cols:
            st.error("Pilih minimal satu kolom fitur sebelum menjalankan pipeline.")
            st.stop()
        if target_col in feature_cols:
            st.error("Kolom target tidak boleh ikut dipilih sebagai kolom fitur. Pilih ulang kolom fitur.")
            st.stop()
        if not XGB_AVAILABLE:
            st.info("Library `xgboost` tidak ditemukan — menggunakan `GradientBoostingClassifier` sebagai pengganti.")

        with st.spinner("Menjalankan pipeline lengkap: dedup → split → filter usia → preprocessing → training (dengan & tanpa SMOTE)..."):
            try:
                full = run_pipeline_full(df_raw, feature_cols, target_col, age_filter)
                st.session_state['ml_full'] = full
                st.session_state['ml_feature_cols'] = feature_cols
                st.session_state['ml_target_col'] = target_col
            except Exception as e:
                st.session_state.pop('ml_full', None)
                st.error(f"🚨 Pipeline gagal dijalankan: **{type(e).__name__}: {e}**")
                st.caption(
                    "Penyebab umum: kolom target/fitur salah pilih, data terlalu sedikit setelah filter, "
                    "atau kolom target cuma punya 1 kategori. Cek konfigurasi kolom di atas lalu coba lagi."
                )
                st.stop()

    # ---- semua tampilan hasil di bawah ini pakai data dari session_state,
    #      supaya tetap muncul walau halaman rerun (misal user ganti selectbox) ----
    if 'ml_full' not in st.session_state:
        st.info("⬆️ Atur konfigurasi kolom di atas, lalu klik **'Jalankan Ulang Pipeline & Latih Model'**.")
        st.stop()

    full = st.session_state['ml_full']
    summary = full['summary']
    class_names = full['class_names']
    X_test = full['X_test']
    y_test = full['y_test']
    results_smote = full['results_smote']
    results_nosmote = full['results_nosmote']

    # =========================================================================
    # 4️⃣ RINGKASAN PREPROCESSING
    # =========================================================================
    with st.container(border=True):
        step_header(4, "Ringkasan Preprocessing", "📋")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Data Mentah", f"{summary['n_mentah']:,}")
        c2.metric("Duplikat Dibuang", f"{summary['n_dup']:,}")
        c3.metric("Data Bersih", f"{summary['n_setelah_dedup']:,}")
        c4.metric("Overlap Train-Test", summary['n_overlap'], "harus 0")

        c5, c6, c7 = st.columns(3)
        c5.metric("Data Training (final)", f"{summary['n_train']:,}")
        c6.metric("Data Testing (final)", f"{summary['n_test']:,}")
        c7.metric("Jumlah Kelas Target", len(class_names))

        st.caption(f"Kelas target terdeteksi: {', '.join(class_names)}")
        if summary.get('age_filter_applied'):
            st.caption(f"Filter usia 20-35 diterapkan terpisah ke train & test (data sebelum filter: train={summary['n_train_before_age']}, test={summary['n_test_before_age']}).")

        if summary['n_overlap'] > 0:
            st.error(f"🚨 Ditemukan {summary['n_overlap']} baris testing yang fiturnya identik dengan data training (potensi leakage)!")
        else:
            st.success("✅ Tidak ada overlap fitur antara data training & testing.")

        if summary['missing_info']:
            with st.expander("❓ Detail Missing Value (diisi pakai median dari data training)"):
                st.dataframe(
                    pd.DataFrame(summary['missing_info'], columns=["Fitur", "Kosong (Train)", "Kosong (Test)", "Median Training"]),
                    use_container_width=True, hide_index=True
                )

        # laporkan kalau ada model yang gagal dilatih di salah satu varian
        errs_ns = summary.get('errors_nosmote') or {}
        errs_s = summary.get('errors_smote') or {}
        for name, err in errs_ns.items():
            st.warning(f"⚠️ Model **{name}** (Tanpa SMOTE) gagal dilatih: {err}")
        for name, err in errs_s.items():
            st.warning(f"⚠️ Model **{name}** (Dengan SMOTE) gagal dilatih: {err}")

        if not results_smote and not results_nosmote:
            st.error("Tidak ada model yang berhasil dilatih sama sekali. Periksa kembali dataset & konfigurasi kolom Anda.")
            st.stop()

    # =========================================================================
    # 5️⃣ SMOTE: BAGAIMANA CARA KERJANYA + DISTRIBUSI KELAS
    # =========================================================================
    with st.container(border=True):
        step_header(5, "SMOTE — Cara Kerja & Distribusi Kelas", "🧬")
        st.markdown(
            "**SMOTE (Synthetic Minority Over-sampling Technique)** menyeimbangkan kelas minoritas dengan cara "
            "membuat sampel sintetis baru — BUKAN menduplikat baris yang sudah ada. Untuk tiap sampel kelas "
            "minoritas, SMOTE mencari *k* tetangga terdekat (k-Nearest Neighbors) dari kelas yang sama di ruang "
            "fitur, lalu membuat titik baru di sepanjang garis penghubung antara sampel asli dan salah satu "
            "tetangganya secara acak. Proses ini diulang sampai semua kelas jumlahnya sama dengan kelas mayoritas. "
            "**SMOTE hanya diterapkan pada data training** — data testing tidak pernah disentuh, supaya evaluasi "
            "tetap mencerminkan data asli/nyata."
        )
        if summary['smote_applied']:
            st.success(f"✅ {summary['smote_note']}")
        else:
            st.warning(f"⚠️ {summary['smote_note']}")

        before = summary['class_dist_before_smote']
        after = summary['class_dist_after_smote']
        labels_before = [class_names[k] if isinstance(k, (int, np.integer)) and k < len(class_names) else str(k) for k in before.keys()]
        labels_after = [class_names[k] if isinstance(k, (int, np.integer)) and k < len(class_names) else str(k) for k in after.keys()]

        col_s1, col_s2 = st.columns(2)
        with col_s1:
            fig_b, ax_b = plt.subplots(figsize=(5.5, 4))
            ax_b.bar(labels_before, list(before.values()), color=PALETTE_MAIN[:len(before)], edgecolor=INK, linewidth=1.2)
            for i, v in enumerate(before.values()):
                ax_b.annotate(str(v), (i, v), ha='center', va='bottom', fontweight='bold')
            ax_b.set_title("Training — Sebelum SMOTE", fontweight='bold')
            ax_b.set_ylabel('Jumlah Sampel')
            style_fig(fig_b, ax_b)
            plt.tight_layout()
            st.pyplot(fig_b)
            plt.close(fig_b)
        with col_s2:
            fig_a, ax_a = plt.subplots(figsize=(5.5, 4))
            ax_a.bar(labels_after, list(after.values()), color=PALETTE_MAIN[:len(after)], edgecolor=INK, linewidth=1.2)
            for i, v in enumerate(after.values()):
                ax_a.annotate(str(v), (i, v), ha='center', va='bottom', fontweight='bold')
            ax_a.set_title("Training — Setelah SMOTE", fontweight='bold')
            ax_a.set_ylabel('Jumlah Sampel')
            style_fig(fig_a, ax_a)
            plt.tight_layout()
            st.pyplot(fig_a)
            plt.close(fig_a)

    # =========================================================================
    # 6️⃣ PERBANDINGAN AKURASI: DENGAN SMOTE vs TANPA SMOTE
    # =========================================================================
    with st.container(border=True):
        step_header(6, "Perbandingan Akurasi: Dengan SMOTE vs Tanpa SMOTE", "⚖️")
        st.markdown(
            "Kedua varian dilatih dari hasil preprocessing yang **identik** (split, filter usia, imputasi, "
            "normalisasi, dan label encoding sama persis) — satu-satunya perbedaan adalah data training-nya "
            "di-SMOTE atau tidak. Ini membuat perbandingannya adil (apple-to-apple)."
        )

        all_model_names = list(dict.fromkeys(list(results_nosmote.keys()) + list(results_smote.keys())))
        rows = []
        for name in all_model_names:
            acc_ns = results_nosmote[name]['accuracy'] * 100 if name in results_nosmote else None
            acc_s = results_smote[name]['accuracy'] * 100 if name in results_smote else None
            rows.append({
                "Algoritma": name,
                "Akurasi Tanpa SMOTE (%)": round(acc_ns, 2) if acc_ns is not None else None,
                "Akurasi Dengan SMOTE (%)": round(acc_s, 2) if acc_s is not None else None,
                "Selisih (SMOTE - Tanpa)": round(acc_s - acc_ns, 2) if (acc_ns is not None and acc_s is not None) else None,
            })
        df_compare_smote = pd.DataFrame(rows)
        st.dataframe(df_compare_smote, use_container_width=True, hide_index=True)

        chart_rows = []
        for name in all_model_names:
            if name in results_nosmote:
                chart_rows.append({"Algoritma": name, "Kondisi": "Tanpa SMOTE", "Akurasi (%)": round(results_nosmote[name]['accuracy'] * 100, 2)})
            if name in results_smote:
                chart_rows.append({"Algoritma": name, "Kondisi": "Dengan SMOTE", "Akurasi (%)": round(results_smote[name]['accuracy'] * 100, 2)})
        df_chart = pd.DataFrame(chart_rows)

        fig_cmp, ax_cmp = plt.subplots(figsize=(9, 4.2))
        sns.barplot(x='Algoritma', y='Akurasi (%)', hue='Kondisi', data=df_chart,
                    palette=[PALETTE_MAIN[3], PALETTE_MAIN[0]], ax=ax_cmp, edgecolor=INK, linewidth=1.2)
        ax_cmp.set_ylim(0, 100)
        for p in ax_cmp.patches:
            h = p.get_height()
            if h > 0:
                ax_cmp.annotate(f"{h:.1f}%", (p.get_x() + p.get_width()/2., h - 6), ha='center', va='center',
                                 color='white', fontweight='bold', fontsize=9)
        ax_cmp.legend(title='Kondisi', facecolor="#FFFFFF", labelcolor=INK)
        style_fig(fig_cmp, ax_cmp)
        sns.despine(left=True, bottom=True)
        plt.tight_layout()
        st.pyplot(fig_cmp)
        plt.close(fig_cmp)

        # model + kondisi terbaik secara keseluruhan
        best_overall = None
        for name in all_model_names:
            for kondisi, res_dict in [("Tanpa SMOTE", results_nosmote), ("Dengan SMOTE", results_smote)]:
                if name in res_dict:
                    acc = res_dict[name]['accuracy'] * 100
                    if best_overall is None or acc > best_overall[2]:
                        best_overall = (name, kondisi, acc)
        if best_overall:
            st.success(f"🏆 **Kombinasi terbaik:** {best_overall[0]} ({best_overall[1]}) — akurasi {best_overall[2]:.2f}%")

    # =========================================================================
    # 7️⃣ DETAIL EVALUASI PER MODEL (pilih varian)
    # =========================================================================
    with st.container(border=True):
        step_header(7, "Detail Evaluasi per Model", "🔎")
        varian_options = []
        if results_nosmote:
            varian_options.append("Tanpa SMOTE")
        if results_smote:
            varian_options.append("Dengan SMOTE")
        varian_pilihan = st.radio("Pilih varian yang ingin dilihat detailnya:", options=varian_options, horizontal=True, key="varian_detail")
        results_active = results_smote if varian_pilihan == "Dengan SMOTE" else results_nosmote

        tabs = st.tabs([f"{MODEL_ICONS.get(name, chr(0x1F4CC))} {name}" for name in results_active.keys()])
        for tab, (name, res) in zip(tabs, results_active.items()):
            with tab:
                col1, col2 = st.columns([1.2, 1])
                with col1:
                    st.write("**Confusion Matrix**")
                    fig_cm, ax_cm = plt.subplots(figsize=(5.5, 4.2))
                    sns.heatmap(res['cm'], annot=True, fmt='d', cmap="Greens",
                                xticklabels=class_names, yticklabels=class_names,
                                ax=ax_cm, annot_kws={"size": 10, "weight": "bold"}, cbar=False, linewidths=1.2, linecolor=INK)
                    ax_cm.set_xlabel('Prediksi Model', fontweight='bold')
                    ax_cm.set_ylabel('Aktual', fontweight='bold')
                    style_fig(fig_cm, ax_cm)
                    plt.tight_layout()
                    st.pyplot(fig_cm)
                    plt.close(fig_cm)
                with col2:
                    st.metric(f"Akurasi ({varian_pilihan})", f"{res['accuracy']*100:.2f}%")
                    report_df = pd.DataFrame(res['report']).transpose()
                    report_df = report_df[~report_df.index.isin(['accuracy'])]
                    st.dataframe(report_df.round(3), use_container_width=True)

    # =========================================================================
    # 9️⃣ UNDUH HASIL (gabungan prediksi Tanpa SMOTE & Dengan SMOTE)
    # =========================================================================
    with st.container(border=True):
        step_header(9, "Unduh Hasil", "⬇️")
        try:
            X_test_out = X_test.copy()
            X_test_out['actual'] = y_test
            for name, res in results_nosmote.items():
                X_test_out[f'pred_{name}_tanpa_smote'] = res['y_pred']
            for name, res in results_smote.items():
                X_test_out[f'pred_{name}_dengan_smote'] = res['y_pred']

            col_dl1, col_dl2 = st.columns(2)
            with col_dl1:
                csv_bytes = X_test_out.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "⬇️ Download Hasil Prediksi (CSV)",
                    data=csv_bytes, file_name="hasil_prediksi_data_baru.csv",
                    mime="text/csv", use_container_width=True
                )
            with col_dl2:
                if OPENPYXL_AVAILABLE:
                    from io import BytesIO
                    buffer = BytesIO()
                    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                        X_test_out.to_excel(writer, index=False, sheet_name="hasil_prediksi")
                    st.download_button(
                        "⬇️ Download Hasil Prediksi (XLSX)",
                        data=buffer.getvalue(), file_name="hasil_prediksi_data_baru.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )
                else:
                    st.caption("Download XLSX butuh `openpyxl` terpasang.")
        except Exception as e:
            st.error(f"Gagal menyiapkan file unduhan: {type(e).__name__}: {e}")