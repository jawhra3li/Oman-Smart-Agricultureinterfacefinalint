# train.py
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder

# --- الجزء الأول: تدريب الموديل الأساسي ---
print("⏳ جاري تحميل بيانات المحاصيل العالمية...")
df = pd.read_csv("Crop_recommendation.csv")

X = df[["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]]
y = df["label"]

print("⚙️ جاري معالجة ترميز المحاصيل وتقييس البيانات...")
le = LabelEncoder()
y_encoded = le.fit_transform(y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42
)

print("🤖 جاري تدريب نموذج الذكاء الاصطناعي (SVC)...")
model = SVC(probability=True, random_state=42)
model.fit(X_train, y_train)

# --- الجزء الثاني: إنشاء وتوليد بيانات الولايات العُمانية برمجياً لضمان سلامتها ---
print("🇴🇲 جاري بناء قاعدة بيانات الطقس والولايات العُمانية...")
oman_data_dict = {
    "Wilayah": [
        "مسقط", "مطرح", "السيب", "بوشر", "العامرات", "قريات",
        "صلالة", "ثمريت", "طاقة", "مرباط", "شليم",
        "نزوى", "بهلاء", "إزكي", "سمائل", "أدم",
        "صحار", "الرستاق", "البريمي", "صور", "إبراء", "عبري"
    ] * 2,
    "Season": ["Winter"] * 22 + ["Summer"] * 22,
    "Temperature_C": [
        22, 22, 23, 22, 24, 23, 24, 20, 23, 24, 22, 20, 19, 21, 22, 23, 22, 21, 19, 23, 22, 21, # شتاء
        38, 37, 39, 39, 41, 38, 28, 42, 30, 31, 40, 43, 42, 41, 41, 44, 40, 42, 43, 40, 41, 44  # صيف
    ],
    "Humidity_%": [
        60, 62, 65, 60, 55, 58, 50, 40, 55, 52, 45, 45, 42, 48, 50, 35, 65, 55, 40, 60, 45, 38, # شتاء
        70, 72, 75, 68, 50, 60, 85, 25, 80, 78, 30, 25, 28, 30, 32, 20, 68, 45, 30, 55, 35, 25  # صيف
    ],
    "Rainfall_mm": [
        15, 12, 14, 15, 10, 18, 5,  2,  5,  4,  2,  20, 22, 18, 25, 10, 15, 20, 12, 14, 15, 10, # شتاء
        5,  4,  6,  5,  2,  8,  120,5,  80, 60, 10, 5,  8,  5,  3,  1,  8,  12, 5,  45, 30, 8   # صيف
    ],
    "pH": [
        7.2, 7.3, 7.1, 7.4, 7.5, 7.2, 7.6, 7.8, 7.5, 7.4, 7.9, 7.0, 6.9, 7.2, 7.1, 7.5, 7.3, 7.2, 7.4, 7.3, 7.2, 7.4, # شتاء
        7.3, 7.3, 7.2, 7.4, 7.6, 7.3, 7.5, 7.9, 7.4, 7.3, 7.8, 7.6, 7.0, 7.3, 7.2, 7.6, 7.4, 7.3, 7.5, 7.4, 7.3, 7.5  # صيف
    ]
}
oman_df = pd.DataFrame(oman_data_dict)

# --- الجزء الثالث: الحفظ المشترك والموحد ---
print("💾 جاري حفظ الملفات الأربعة بالتشفير الجديد الموحد...")
joblib.dump(model, "crop_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(le, "label_encoder.pkl")
joblib.dump(oman_df, "oman_data.pkl")

print("✅ تم تدريب النموذج وتوليد كافة ملفات النظام بنجاح وبصيغة موحدة! 🚀")