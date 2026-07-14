 #app.py
import os
import base64
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
import plotly.express as px
import streamlit as st

# ==============================================================
# 0) إعدادات الصفحة
# ==============================================================
st.set_page_config(page_title="Oman Smart Agriculture Recommendation System", page_icon="🇴🇲", layout="wide")

IMAGES_DIR = "images"
LOGO_PATH = os.path.join(IMAGES_DIR, "Oman-2040.png")
BACK_PATH = os.path.join(IMAGES_DIR, "back.png")


@st.cache_data(show_spinner=False)
def _file_to_data_uri(path: str):
    """يحوّل ملف صورة إلى data URI (base64) لتُستخدم داخل CSS url()، لأن المتصفح لا يستطيع
    قراءة مسار ملف محلي مباشرة داخل ورقة الأنماط."""
    if not path or not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f"data:image/png;base64,{b64}"


# ==============================================================
# 1) خلفية التطبيق — كتلة CSS واحدة ونظيفة تستهدف .stApp مباشرة
#    (هذه أول كتلة st.markdown في الملف، وتحتوي فقط على CSS وليس أي HTML)
# ==============================================================
_back_data_uri = _file_to_data_uri(BACK_PATH)
if _back_data_uri:
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image:
                linear-gradient(rgba(238,245,239,0.72), rgba(238,245,239,0.72)),
                url("{_back_data_uri}");
            background-size: cover;
            background-position: center center;
            background-attachment: fixed;
            background-repeat: no-repeat;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

# ==============================================================
# 2) دعم الخريطة التفاعلية (Folium) — استيراد آمن
# ==============================================================
try:
    import folium
    from streamlit_folium import st_folium
    MAP_LIBS_AVAILABLE = True
except ImportError:
    MAP_LIBS_AVAILABLE = False

OMAN_CENTER = (21.4735, 55.9754)

# الحدود التقريبية لكامل سلطنة عُمان [[جنوب، غرب], [شمال، شرق]] — تُستخدم لعرض الدولة بأكملها عبر fit_bounds
OMAN_BOUNDS = [[16.5, 51.8], [26.5, 60.0]]

# إحداثيات تقريبية لولايات سلطنة عُمان — عدّل المفاتيح لمطابقة تهجئة "Wilayah" في بياناتك بدقة إن اختلفت
WILAYAH_COORDS = {
    "مسقط": (23.6139, 58.5922), "مطرح": (23.6289, 58.5619), "بوشر": (23.5859, 58.4059),
    "السيب": (23.6703, 58.1889), "العامرات": (23.5464, 58.4106), "قريات": (23.2617, 58.9153),
    "صلالة": (17.0151, 54.0924), "طاقة": (17.0439, 54.3903), "مرباط": (16.9944, 54.6942),
    "رخيوت": (16.7333, 53.4167), "ضلكوت": (16.7333, 53.0833), "سدح": (17.7333, 55.1667),
    "ثمريت": (17.6167, 54.0333), "مقشن": (19.6333, 55.0833), "المزيونة": (18.5833, 52.4),
    "شليم وجزر الحلانيات": (17.5, 55.9333),
    "صحار": (24.3467, 56.7092), "شناص": (24.7397, 56.4661), "لوى": (24.6333, 56.5833),
    "صحم": (24.1667, 56.8833), "الخابورة": (23.9769, 57.0908), "السويق": (23.85, 57.4333),
    "الرستاق": (23.3908, 57.4256), "العوابي": (23.3333, 57.6167), "نخل": (23.3944, 57.8333),
    "وادي المعاول": (23.4333, 57.75), "بركاء": (23.6733, 57.8878), "المصنعة": (23.7667, 57.65),
    "نزوى": (22.9333, 57.5333), "بهلاء": (22.9667, 57.3), "منح": (23.0, 57.65),
    "الحمراء": (23.1167, 57.3), "أدم": (22.3833, 57.5167), "إزكي": (22.9333, 57.75),
    "سمائل": (23.3, 58.05), "بدبد": (23.4167, 58.1333),
    "إبراء": (22.6906, 58.5333), "القابل": (22.6167, 58.9833), "المضيبي": (22.6333, 58.1167),
    "وادي بني خالد": (22.6833, 59.2833), "بدية": (22.4667, 58.8), "دماء والطائيين": (22.9, 58.6167),
    "صور": (22.5667, 59.5289), "الكامل والوافي": (22.15, 59.35), "جعلان بني بو علي": (22.05, 59.2667),
    "جعلان بني بو حسن": (22.15, 59.2833), "مصيرة": (20.6667, 58.8833),
    "عبري": (23.2222, 56.5083), "ينقل": (23.5833, 56.5333), "ضنك": (23.55, 56.2667),
    "البريمي": (24.2508, 55.7933), "محضة": (24.35, 55.95), "السنينة": (24.35, 55.85),
    "خصب": (26.1833, 56.25), "بخا": (26.0167, 56.2667), "دباء البيعة": (25.6333, 56.2667),
    "مدحاء": (25.3, 56.3333),
    "هيماء": (19.9333, 56.2667), "الدقم": (19.6667, 57.7), "محوت": (20.8667, 57.7),
    "الجازر": (19.2333, 56.75),
}

# ==============================================================
# 3) قواميس المحاصيل
# ==============================================================
CROP_IMAGE_FILES = {
    "apple": "apple.png", "banana": "banana.png", "blackgram": "blackgram.png",
    "chickpea": "chickpea.png", "coconut": "coconut.png", "coffee": "coffee.png",
    "cotton": "cotton.png", "grapes": "grapes.png", "jute": "jute.png",
    "kidneybeans": "kidneybeans.png", "lentil": "lentil.png", "maize": "maize.png",
    "mango": "mango.png", "mothbeans": "mothbeans.png", "mungbean": "mungbean.png",
    "muskmelon": "muskmelon.png", "orange": "orange.png", "papaya": "papaya.png",
    "pigeonpeas": "pigeonpeas.png", "pomegranate": "pomegranate.png",
    "rice": "rice.png", "watermelon": "watermelon.png",
}
CROP_EMOJI_FALLBACK = {
    "apple": "🍎", "banana": "🍌", "blackgram": "🫘", "chickpea": "🫛", "coconut": "🥥",
    "coffee": "☕", "cotton": "🌱", "grapes": "🍇", "jute": "🌿", "kidneybeans": "🫘",
    "lentil": "🌰", "maize": "🌽", "mango": "🥭", "mothbeans": "🌱", "mungbean": "🌱",
    "muskmelon": "🍈", "orange": "🍊", "papaya": "🫛", "pigeonpeas": "🌱",
    "pomegranate": "🍎", "rice": "🌾", "watermelon": "🍉",
}
crop_ar = {
    "rice": "الأرز", "maize": "الذرة", "chickpea": "الحمص", "kidneybeans": "الفاصوليا",
    "pigeonpeas": "البازلاء", "mothbeans": "العنبية", "mungbean": "الماش",
    "blackgram": "العدس الأسود", "lentil": "العدس", "pomegranate": "الرمان",
    "banana": "الموز", "mango": "المانجا", "grapes": "العنب", "watermelon": "البطيخ",
    "muskmelon": "الشمام", "apple": "التفاح", "orange": "البرتقال", "papaya": "البابايا",
    "coconut": "النارجيل", "cotton": "القطن", "jute": "الكتان", "coffee": "القهوة",
    "wheat": "القمح العُماني",
}


def crop_icon(eng_name: str, size: int = 90):
    """يعرض أيقونة المحصول: صورة PNG إن وُجدت، وإلا إيموجي بديل — عبر st.image/st.markdown فقط، بدون HTML."""
    path = os.path.join(IMAGES_DIR, CROP_IMAGE_FILES.get(eng_name, ""))
    if os.path.exists(path):
        st.image(path, width=size)
    else:
        st.markdown(f"## {CROP_EMOJI_FALLBACK.get(eng_name, '🌱')}")


# ==============================================================
# 4) التنسيقات (CSS) — تستهدف فقط عناصر Streamlit الأصلية (containers بمفاتيح key=) والخطوط والاتجاه
#    لا يوجد أي وسم HTML يحمل محتوى هنا؛ فقط قواعد CSS نقية.
# ==============================================================
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@400;500;700;900&display=swap');

    html, body, [data-testid="stAppViewContainer"], .main {
        direction: rtl;
        text-align: right;
        font-family: 'Tajawal', sans-serif;
    }
    [data-testid="stHeader"] { background: rgba(0,0,0,0) !important; }
    section[data-testid="stSidebar"] { direction: rtl; text-align: right; }
    label, p, span, div[data-testid="stWidgetLabel"] p, .stMarkdown { font-family: 'Tajawal', sans-serif !important; }
    h1, h2, h3, h4 { font-weight: 800 !important; color: #1e3a2f !important; }

    /* ---------- الهيدر العلوي: st.container(key="top_header") ---------- */
    div[class*="st-key-top_header"] {
        background: linear-gradient(90deg, #0f172a 0%, #123a2e 55%, #15803d 100%) !important;
        border-radius: 16px !important; padding: 18px 26px !important;
        box-shadow: 0 10px 25px rgba(21, 128, 61, 0.15); margin-bottom: 22px;
    }
    div[class*="st-key-top_header"] h1, div[class*="st-key-top_header"] h2,
    div[class*="st-key-top_header"] h3, div[class*="st-key-top_header"] p,
    div[class*="st-key-top_header"] span, div[class*="st-key-top_header"] .stCaption { color: white !important; }
    div[class*="st-key-top_header"] img { opacity: 0.9; }

    /* ---------- لوحة المدخلات: st.container(key="env_panel") ---------- */
    div[class*="st-key-env_panel"] {
        background: rgba(255,255,255,0.92) !important; border-radius: 18px !important;
        padding: 22px 24px !important; box-shadow: 0 6px 20px rgba(0,0,0,0.06);
    }
    div[class*="st-key-env_panel"] div[data-testid="stSelectbox"],
    div[class*="st-key-env_panel"] div[data-testid="stRadio"],
    div[class*="st-key-env_panel"] div[data-testid="stSlider"] { margin-bottom: 18px !important; width: 100%; }
    div[class*="st-key-env_panel"] label, div[class*="st-key-env_panel"] div[data-testid="stWidgetLabel"] {
        width: 100%; text-align: right !important; direction: rtl !important;
    }
    div[class*="st-key-env_panel"] div[role="radiogroup"] { justify-content: flex-end !important; }
    div[class*="st-key-env_panel"] div[data-baseweb="select"] { direction: rtl; }

    /* ---------- شريط الـ Slider: BaseWeb داخليًا LTR لضمان سحب صحيح ---------- */
    div[data-baseweb="slider"] { direction: ltr !important; }
    div[data-testid="stSliderTickBarMin"], div[data-testid="stSliderTickBarMax"] { direction: ltr !important; }

    /* ---- تلوين شريط الـ Slider ديناميكيًا: أحمر حتى نسبة القيمة (--slider-pct)، ورمادي لما تبقّى ----
       يُحقن متغيّر CSS الخاص --slider-pct لكل حاوية slider عبر كتلة st.markdown صغيرة بعد كل تشغيل مباشرة
       (هذه هي كتلة CSS الوحيدة المرتبطة ديناميكيًا بقيمة الأداة). */
    div[class*="st-key-sl_"] { --slider-pct: 0%; }
    div[class*="st-key-sl_"] div[data-testid="stSlider"] div[data-baseweb="slider"] > div:first-of-type {
        background: linear-gradient(to right,
            #e5493f 0%, #e5493f var(--slider-pct),
            #dde5df var(--slider-pct), #dde5df 100%) !important;
    }
    div[class*="st-key-sl_"] div[data-testid="stSlider"] div[role="slider"] {
        background: #e5493f !important; border-color: #e5493f !important;
    }

    /* ---------- خريطة عُمان: st.container(key="map_panel") ---------- */
    div[class*="st-key-map_panel"] {
        background: rgba(255,255,255,0.94) !important; border-radius: 16px !important; padding: 10px !important;
        box-shadow: 0 4px 16px rgba(0,0,0,0.05) !important; overflow: hidden; margin-bottom: 18px;
    }

    /* ---------- بطاقات المؤشرات البيئية 2x2: st.container(key="ind_*", border=True) ---------- */
    div[class*="st-key-ind_"] {
        background: rgba(255,255,255,0.94) !important; border-radius: 14px !important;
        box-shadow: 0 4px 14px rgba(0,0,0,0.05) !important; min-height: 100px; box-sizing: border-box;
    }
    div[class*="st-key-ind_"] div[data-testid="stMetric"] { text-align: center; }
    div[class*="st-key-ind_"] div[data-testid="stMetricLabel"] { justify-content: center !important; font-weight: 700 !important; color: #1e3a2f !important; }
    div[class*="st-key-ind_"] div[data-testid="stMetricValue"] { justify-content: center !important; font-size: 20px !important; color: #15803d !important; }

    /* ---------- بطاقة النتيجة الرئيسية: st.container(key="hero_img_card" / "hero_text_card") ---------- */
    div[class*="st-key-hero_img_card"] {
        border-radius: 18px !important; overflow: hidden; box-shadow: 0 12px 28px rgba(21,128,61,0.2);
        min-height: 220px; display: flex; align-items: center; justify-content: center;
        background: white !important;
    }
    div[class*="st-key-hero_text_card"] {
        border-radius: 18px !important; min-height: 220px; box-sizing: border-box;
        background: linear-gradient(135deg, #15803d 0%, #0d9457 55%, #16a34a 100%) !important;
        box-shadow: 0 12px 28px rgba(21,128,61,0.25); padding: 26px 30px !important;
        display: flex; flex-direction: column; justify-content: center;
    }
    div[class*="st-key-hero_text_card"] * { color: white !important; }
    div[class*="st-key-hero_text_card"] div[data-testid="stMetricValue"] { color: white !important; }

    /* ---------- بطاقات المحاصيل البديلة: st.container(key="alt_*", border=True) ---------- */
    div[class*="st-key-alt_"] {
        background: rgba(255,255,255,0.95) !important; border-radius: 14px !important; text-align: center;
        box-shadow: 0 4px 14px rgba(0,0,0,0.05) !important;
    }

    /* ---------- بطاقة مؤشر الزراعة الذكية: st.container(key="kpi_panel") و kpi_* ---------- */
    div[class*="st-key-kpi_panel"] {
        background: rgba(255,255,255,0.92) !important; border-radius: 16px !important;
        padding: 18px 20px !important; box-shadow: 0 4px 16px rgba(0,0,0,0.04) !important; margin-bottom: 20px;
    }
    div[class*="st-key-kpi_panel"] p { text-align: center; }

    /* ---------- أزرار Streamlit ---------- */
    .stButton>button {
        background: linear-gradient(90deg, #0f172a 0%, #16a34a 100%) !important;
        color: white !important; font-family: 'Tajawal', sans-serif !important;
        font-weight: 800 !important; font-size: 16px !important; width: 100%;
        border: none !important; padding: 14px !important; border-radius: 12px !important;
        box-shadow: 0 4px 15px rgba(22, 163, 74, 0.25); transition: all .25s ease;
    }
    .stButton>button:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(15,23,42,0.3) !important; }
    </style>
""", unsafe_allow_html=True)


# ==============================================================
# 5) دوال مساعدة (بدون أي HTML — مكوّنات Streamlit أصلية فقط)
# ==============================================================
def styled_slider(label, min_value, max_value, value, key):
    """Slider بتلوين ديناميكي (أحمر حتى القيمة، رمادي بعدها) عبر متغيّر CSS مرتبط بحاوية key."""
    with st.container(key=key):
        val = st.slider(label, min_value=min_value, max_value=max_value, value=value)
    pct = 0.0 if max_value == min_value else (val - min_value) / (max_value - min_value) * 100
    st.markdown(
        f'<style>div[class*="st-key-{key}"] {{ --slider-pct: {pct:.2f}%; }}</style>',
        unsafe_allow_html=True,
    )
    return val


def indicator_metric_card(icon, label, value, unit, key):
    """بطاقة مؤشر بيئي واحدة باستخدام st.container + st.metric فقط — بدون أي HTML."""
    with st.container(key=key, border=True):
        st.metric(label=f"{icon} {label}", value=f"{value:g} {unit}".strip())


def render_offline_locator(selected_wilayah: str):
    """يعرض مؤشر موقع بسيط بالكامل عبر CSS/HTML — بدون أي اتصال إنترنت أو مكتبات خارجية.
    مضمون الظهور دائمًا، على عكس خريطة Leaflet التفاعلية التي تحتاج تحميل ملفات من CDN."""
    lat, lon = WILAYAH_COORDS.get(selected_wilayah, OMAN_CENTER)
    lats = [c[0] for c in WILAYAH_COORDS.values()]
    lons = [c[1] for c in WILAYAH_COORDS.values()]
    lat_min, lat_max = min(lats) - 0.4, max(lats) + 0.4
    lon_min, lon_max = min(lons) - 0.4, max(lons) + 0.4
    x_pct = max(3, min(97, (lon - lon_min) / (lon_max - lon_min) * 100))
    y_pct = max(3, min(97, (lat_max - lat) / (lat_max - lat_min) * 100))

    st.markdown(f"""
        <div style="position:relative; width:100%; height:220px; border-radius:14px; overflow:hidden;
                    background: linear-gradient(160deg, #d7ecd9 0%, #bfe0c6 55%, #a9d9b8 100%);
                    border:1px solid #cfe6d3;">
            <div style="position:absolute; inset:0;
                        background-image: linear-gradient(#ffffff40 1px, transparent 1px),
                                           linear-gradient(90deg, #ffffff40 1px, transparent 1px);
                        background-size: 12% 16%;"></div>
            <div style="position:absolute; left:{x_pct:.1f}%; top:{y_pct:.1f}%; transform:translate(-50%,-100%);
                        display:flex; flex-direction:column; align-items:center;">
                <div style="font-size:26px; line-height:1; filter:drop-shadow(0 2px 3px rgba(0,0,0,.35));">📍</div>
                <div style="background:#0f172a; color:white; font-size:11px; font-weight:700; padding:3px 8px;
                            border-radius:6px; margin-top:2px; white-space:nowrap;">{selected_wilayah}</div>
            </div>
        </div>
        <div style="font-size:11px; color:#6b7d74; text-align:center; margin-top:6px;">
            📌 إحداثيات تقريبية: {lat:.3f}°N, {lon:.3f}°E — خريطة مصغّرة توضيحية (لا تتطلب اتصال إنترنت)
        </div>
    """, unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def build_oman_map(selected_wilayah: str):
    """يبني خريطة Folium تعرض سلطنة عُمان بالكامل، مع تمييز الولاية المختارة بدبوس واضح.

    ملاحظتان مهمتان مقارنة بالنسخة السابقة:
    1) بدل الاعتماد على zoom_start ثابت (وهو تخمين قد لا يناسب كل أحجام الحاويات)، نستخدم
       fmap.fit_bounds(OMAN_BOUNDS) الذي يحسب مستوى التكبير المناسب تلقائيًا بحيث تظهر عُمان
       بأكملها دائمًا، بصرف النظر عن عرض/ارتفاع الحاوية في الصفحة.
    2) استُبدل `folium.Circle` (نصف قطره بالأمتار) بـ `folium.CircleMarker` (نصف قطره بالبكسل).
       عند تصغير الخريطة لعرض الدولة كاملة، أي دائرة بمقياس الأمتار (كانت 18 كم) تتقلص بصريًا حتى
       تختفي تقريبًا؛ أما CircleMarker فيحافظ على حجمه المرئي بالبكسل مهما كان مستوى التكبير.
    """
    center = WILAYAH_COORDS.get(selected_wilayah, OMAN_CENTER)

    fmap = folium.Map(location=OMAN_CENTER, zoom_start=6, tiles="CartoDB positron", control_scale=True)
    fmap.fit_bounds(OMAN_BOUNDS)

    # هالة بحجم ثابت بالبكسل حول الولاية المختارة — تبقى واضحة حتى مع ظهور الدولة كاملة
    folium.CircleMarker(
        location=center, radius=14, color="#e5493f", weight=2,
        fill=True, fill_color="#e5493f", fill_opacity=0.18,
    ).add_to(fmap)

    # دبوس مميز وواضح فوق الهالة، بلون مختلف (أحمر) ليبرز بوضوح على خلفية الخريطة الخضراء
    folium.Marker(
        location=center,
        tooltip=f"📍 {selected_wilayah}",
        popup=folium.Popup(f"<b>الولاية المختارة:</b> {selected_wilayah}", max_width=220),
        icon=folium.Icon(color="red", icon="map-marker", prefix="fa"),
    ).add_to(fmap)

    return fmap


# ==============================================================
# 6) تحميل ملفات النموذج
# ==============================================================
@st.cache_resource
def load_ml_assets():
    model = joblib.load("crop_model.pkl")
    le = joblib.load("label_encoder.pkl")
    scaler = joblib.load("scaler.pkl")
    oman_df = joblib.load("oman_data.pkl")
    return model, le, scaler, oman_df


try:
    rf_model, le, scaler, oman_df = load_ml_assets()
    assets_loaded = True
except Exception as e:
    assets_loaded = False
    st.error(f"⚠️ خطأ في تحميل ملفات النظام: {e}")

# ==============================================================
# 7) الهيدر العلوي — st.container(key="top_header") + مكوّنات أصلية فقط
# ==============================================================
now = datetime.now()
with st.container(key="top_header"):
    head_logo, head_title, head_date = st.columns([1, 5, 2])
    with head_logo:
        if os.path.exists(LOGO_PATH):
            st.image(LOGO_PATH, width=56)
    with head_title:
        st.markdown("### نظام عُمان الذكي لتوصية المحاصيل الزراعية")
        st.caption("Oman Smart Agriculture Recommendation System (Machine Learning)")
    with head_date:
        st.caption(now.strftime("%I:%M %p"))
        st.caption(now.strftime("%A %d %B %Y"))

if not assets_loaded:
    st.stop()

# ==============================================================
# 8) التخطيط الرئيسي
# ==============================================================
col_env, col_dashboard = st.columns([1, 1.35], gap="large")

with col_env:
    with st.container(key="env_panel"):
        st.markdown("#### ⚙️ المؤشرات والمدخلات البيئية")

        wilayah = st.selectbox("🗺️ اختر الولاية العُمانية", options=sorted(oman_df["Wilayah"].unique()))
        season_display = st.radio("🌤️ اختر الفصل الزراعي", options=["فصل الشتاء", "فصل الصيف"], index=0, horizontal=True)
        season_eng = "Winter" if season_display == "فصل الشتاء" else "Summer"

        # ---------------- الموقع الجغرافي للولاية المختارة ----------------
        st.markdown("##### 🗺️ الموقع الجغرافي للولاية المختارة")
        with st.container(key="map_panel"):
            # مؤشر موقع ثابت يعمل دائمًا بدون إنترنت — هذا ما يظهر افتراضيًا
            render_offline_locator(wilayah)

            if MAP_LIBS_AVAILABLE:
                show_interactive_map = st.checkbox(
                    "🌐 تفعيل الخريطة التفاعلية (Leaflet) — تتطلب اتصال إنترنت فعّال بـ cdn.jsdelivr.net",
                    value=False, key="toggle_interactive_map",
                )
                if show_interactive_map:
                    oman_map = build_oman_map(wilayah)
                    st_folium(oman_map, height=260, use_container_width=True, returned_objects=[], key=f"oman_map_{wilayah}")
            else:
                st.caption("💡 لتفعيل الخريطة التفاعلية لاحقًا، ثبّت: `pip install folium streamlit-folium`.")

        st.markdown("##### 🧪 مستويات تراكيز مغذيات التربة (NPK)")
        N = styled_slider("النيتروجين (N) — ppm", min_value=0, max_value=150, value=90, key="sl_N")
        P = styled_slider("الفسفور (P) — ppm", min_value=0, max_value=150, value=40, key="sl_P")
        K = styled_slider("البوتاسيوم (K) — ppm", min_value=0, max_value=150, value=40, key="sl_K")

        # قراءات مناخ الولاية/الفصل المختار (بطاقات 2x2 عبر st.metric فقط)
        row_preview = oman_df[(oman_df["Wilayah"] == wilayah) & (oman_df["Season"] == season_eng)]
        if len(row_preview) > 0:
            temperature_p = float(row_preview["Temperature_C"].values[0])
            humidity_p = float(row_preview["Humidity_%"].values[0])
            rainfall_p = float(row_preview["Rainfall_mm"].values[0])
            ph_p = float(row_preview["pH"].values[0])

            st.markdown("##### 🌍 مؤشرات البيئة الحالية في الولاية")
            ind_r1c1, ind_r1c2 = st.columns(2, gap="small")
            with ind_r1c1:
                indicator_metric_card("🌡️", "درجة الحرارة", temperature_p, "°C", key="ind_temp")
            with ind_r1c2:
                indicator_metric_card("💧", "الرطوبة", humidity_p, "%", key="ind_hum")

            ind_r2c1, ind_r2c2 = st.columns(2, gap="small")
            with ind_r2c1:
                indicator_metric_card("🌧️", "كمية الأمطار", rainfall_p, "mm", key="ind_rain")
            with ind_r2c2:
                indicator_metric_card("🧪", "حموضة التربة (pH)", ph_p, "", key="ind_ph")
        else:
            st.warning("⚠️ لا توجد بيانات مناخية مسجلة لهذه الولاية/الفصل.")

        run_prediction = st.button("🔮 احسب التوصية الآن")

with col_dashboard:
    st.markdown("#### 📊 لوحة تخطيط ذكية لأفضل الزراعات بناءً على بياناتك")

    if not run_prediction:
        st.info("💡 حدد الولاية والمؤشرات من اللوحة الجانبية، ثم اضغط 'احسب التوصية الآن' لعرض التحليل الكامل.")
    elif len(row_preview) == 0:
        st.warning("⚠️ لا يمكن حساب التوصية بدون بيانات مناخية لهذه الولاية/الفصل.")
    else:
        temperature, humidity, rainfall, ph = temperature_p, humidity_p, rainfall_p, ph_p

        features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        features_scaled = scaler.transform(features)

        probabilities = rf_model.predict_proba(features_scaled)[0]
        crops_classes = rf_model.classes_

        scored_results = []
        for i in range(len(crops_classes)):
            crop = crops_classes[i]
            suitability_score = probabilities[i] * 100
            if rainfall < 50 and crop in ["rice", "coconut"]:
                suitability_score -= 30
            if temperature > 35 and crop in ["apple", "grapes"]:
                suitability_score -= 20
            if humidity < 40 and crop in ["rice"]:
                suitability_score -= 15
            suitability_score = max(suitability_score, 0.0)

            crop_name_raw = le.inverse_transform([crop])[0]
            crop_translated = crop_ar.get(crop_name_raw, crop_name_raw)
            scored_results.append((crop_translated, suitability_score, crop_name_raw))

        scored_results = sorted(scored_results, key=lambda x: x[1], reverse=True)
        top_crop_name, top_crop_score, top_crop_english = scored_results[0]

        # ---------------- بطاقة النتيجة الرئيسية (Hero) — st.container فقط ----------------
        hc1, hc2 = st.columns([1, 1.6])
        with hc1:
            with st.container(key="hero_img_card", border=True):
                crop_icon(top_crop_english, size=160)
        with hc2:
            with st.container(key="hero_text_card", border=True):
                st.caption("🌱 المحصول المقترح الأنسب لك")
                st.markdown(f"## {top_crop_name}")
                st.metric("🎯 درجة الثقة", f"{top_crop_score:.0f}%")

        # ---------------- أفضل 3 محاصيل بديلة ----------------
        st.markdown("#### 🏅 أفضل 3 محاصيل بديلة مقترحة")
        alt1, alt2, alt3 = st.columns(3)
        for col, (name, score, eng) in zip([alt1, alt2, alt3], scored_results[1:4]):
            rank = scored_results.index((name, score, eng)) + 1
            with col:
                with st.container(key=f"alt_{rank}", border=True):
                    crop_icon(eng, size=70)
                    st.markdown(f"**{rank}. {name}**")
                    st.caption(f"درجة الصلاحية: {score:.0f}%")
                    st.progress(min(1.0, max(0.0, score / 100)))

        # ---------------- مؤشر الزراعة الذكية في عُمان ----------------
        soil_score = max(0, min(100, (N + P + K) / 3 / 1.2))
        climate_score = max(0, 100 - abs(temperature - 26) * 2.2)
        water_score = max(0, min(100, (rainfall / 1.3) + (humidity / 4)))
        sustainability_score = round((soil_score + water_score) / 2)
        productivity_score = round(top_crop_score)

        with st.container(key="kpi_panel"):
            st.markdown("#### 🌍 مؤشر الزراعة الذكية في عُمان")
            k1, k2, k3, k4, k5 = st.columns(5)
            for col, (label, val) in zip(
                [k1, k2, k3, k4, k5],
                [("التربة", round(soil_score)), ("المناخ", round(climate_score)),
                 ("المياه", round(water_score)), ("الاستدامة", sustainability_score),
                 ("الإنتاجية", productivity_score)]
            ):
                with col:
                    st.markdown(f"**{label}**")
                    st.progress(min(1.0, max(0.0, val / 100)))
                    st.caption(f"{val}/100")

        # ---------------- التوصيات — تنبيهات Streamlit الأصلية بدلاً من صناديق HTML ----------------
        soil_notes = []
        if N < 50:
            soil_notes.append("يُنصح بإضافة سماد الـ Urea لرفع مستوى النيتروجين وتحسين التهوية.")
        if P < 40:
            soil_notes.append("يُنصح بإضافة سماد الـ DAP لرفع مستوى الفسفور.")
        if K < 40:
            soil_notes.append("يُنصح بإضافة سماد الـ Potash لرفع مستوى البوتاسيوم.")
        if not soil_notes:
            soil_notes.append("يوصى بإضافة المواد العضوية وتحسين التهوية لزيادة الخصوبة.")

        st.warning(f"🌱 **توصيات تحسين التربة** — {soil_notes[0]}")
        st.success("🧪 **توصيات التسميد** — السماد المقترح: NPK (10-10-10) بمعدل 50 كجم/هكتار")
        st.error("❤️ **نصيحة** — احرص على الري المنتظم ومراقبة رطوبة التربة لضمان أفضل إنتاجية.")

        # ---------------- الرسم البياني التفاعلي ----------------
        st.markdown("#### 📊 التحليل البياني التفاعلي لمعدلات الملاءمة (أعلى 5 محاصيل)")
        top_5 = scored_results[:5]
        df_chart = pd.DataFrame({
            'المحصول': [c[0] for c in top_5],
            'نسبة الملاءمة (%)': [round(c[1], 2) for c in top_5],
        })
        fig = px.bar(df_chart, x='المحصول', y='نسبة الملاءمة (%)', text='نسبة الملاءمة (%)',
                     color_discrete_sequence=['#15803d'])
        fig.update_traces(textposition='outside', textfont_size=14, textfont_family='Tajawal', cliponaxis=False)
        fig.update_layout(
            font_family='Tajawal',
            margin=dict(l=20, r=20, t=30, b=20),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            yaxis=dict(range=[0, max(df_chart['نسبة الملاءمة (%)']) + 15]),
            height=340,
        )
        st.plotly_chart(fig, use_container_width=True)

        # ---------------- تقرير قابل للتحميل ----------------
        alternatives_txt = "\n".join([f"  {i+1}. {c} -> {s:.2f}%" for i, (c, s, _) in enumerate(scored_results[:3])])
        report_txt = f"""==================================================
🇴🇲 تقرير الاستدامة الزراعية الذكية - رؤية عُمان 2040 🇴🇲
==================================================
تاريخ التوليد: {now.strftime('%Y-%m-%d %H:%M')}

الولاية: {wilayah}   |   الفصل: {season_display}
NPK: N={N}, P={P}, K={K}
الطقس: {temperature}°م | رطوبة {humidity}% | أمطار {rainfall}مم | pH {ph}

المحصول المقترح: {top_crop_name} ({top_crop_score:.2f}%)
أفضل 3 بدائل:
{alternatives_txt}

مؤشرات الزراعة الذكية:
التربة {round(soil_score)}/100 | المناخ {round(climate_score)}/100 | المياه {round(water_score)}/100 | الاستدامة {sustainability_score}/100 | الإنتاجية {productivity_score}/100
=================================================="""

        st.download_button("📥 تحميل التقرير الشامل (.txt)", data=report_txt,
                            file_name=f"Oman_Agriculture_Report_{wilayah}.txt", mime="text/plain")
