import streamlit as st
import cv2
import numpy as np
import onnxruntime as ort
import pandas as pd
import os
import time

# 1. إعداد واجهة المستخدم السحابية الاحترافية لمشروع التخرج
st.set_page_config(page_title="Arabic Sign Language System", layout="centered")
st.title("🤖 نظام الترجمة الفورية الحية لغة الإشارة العربية (ArSL)")
st.write("الرابط السحابي نشط 100%! التقط لقطات متتالية للإشارة لبناء جملة كاملة ومستمرة.")

# تحميل موديل ONNX والأسماء من ملف الإكسل مرة واحدة لحفظ الذاكرة السحابية
@st.cache_resource
def load_onnx_session():
    model_path = "arabic_sign_model.onnx"
    if os.path.exists(model_path):
        return ort.InferenceSession(model_path)
    return None

@st.cache_data
def load_labels():
    excel_path = "KARSL-502_Labels.xlsx"
    if os.path.exists(excel_path):
        try:
            df = pd.read_excel(excel_path)
            return df.iloc[:, 1].astype(str).str.strip().tolist()
        except: pass
    return [f"Word {i+1}" for i in range(502)]

ort_session = load_onnx_session()
class_labels = load_labels()

# تهيئة الذاكرة السحابية المستقرة للجملة التراكمية الكبيرة
if "translated_sentence" not in st.session_state:
    st.session_state.translated_sentence = []
if "frame_buffer" not in st.session_state:
    st.session_state.frame_buffer = []

# عرض الجملة الكبيرة المتسلسلة والمستمرة في المقدمة بشكل بارز
st.markdown("---")
st.markdown("### 📝 الجملة المترجمة الحالية (متسلسلة ومستمرة):")
sentence_placeholder = st.empty()

full_sentence = " ".join(st.session_state.translated_sentence)
if full_sentence:
    sentence_placeholder.markdown(f"<h1 style='text-align: center; color: #FF4B4B; direction: rtl;'>{full_sentence}</h1>", unsafe_allow_html=True)
else:
    sentence_placeholder.markdown("<h3 style='text-align: center; color: #777;'>التقط الإشارات المتتالية بالأسفل لبناء جملتك الحية...</h3>", unsafe_allow_html=True)

# زر مسح الجملة للبدء من جديد وتطهير الذاكرة السحابية فورا
if st.button("🗑️ مسح الجملة والبدء من جديد", type="secondary"):
    st.session_state.translated_sentence = []
    st.session_state.frame_buffer = []
    st.rerun()

st.markdown("---")

# 2. أداة الكاميرا المدمجة والآمنة سحابياً لمنع كراش الـ STUN
st.markdown("### 📸 التقط لقطة الحركة الحالية:")
camera_file = st.camera_input("وجه يدك للكاميرا وثبتها على الإشارة ثم اضغط على زر التقاط (Take Photo)")

if camera_file is not None and ort_session is not None:
    # تحويل الملف المكتشف مباشرة إلى مصفوفة صور رشيقة وتطبيع ألوانها للـ RGB
    file_bytes = np.asarray(bytearray(camera_file.read()), dtype=np.uint8)
    opencv_img = cv2.imdecode(file_bytes, 1)
    rgb_frame = cv2.cvtColor(opencv_img, cv2.COLOR_BGR2RGB)
    
    # تحجيم الإطار وتطبيعه ليتوافق مع أوزان الـ ONNX لـ 502 كلمة
    resized = cv2.resize(rgb_frame, (64, 64))
    normalized_frame = resized / 255.0
    
    # حشو الذاكرة المؤقتة بـ 5 إطارات متطابقة من نفس الوضعية لضمان ثبات رياضيات الموديل الحركي (GRU)
    st.session_state.frame_buffer = [normalized_frame] * 5
    
    # تشغيل التنبؤ الفوري والمستقر سحابياً وبدون أي تأخير برميجي
    input_data = np.expand_dims(np.array(st.session_state.frame_buffer, dtype=np.float32), axis=0)
    input_name = ort_session.get_inputs()[0].name
    outputs = ort_session.run(None, {input_name: input_data})
    
    predictions = np.squeeze(outputs)
    predicted_class_idx = np.argmax(predictions)
    confidence = predictions[predicted_class_idx]
    
    try:
        detected_word = class_labels[predicted_class_idx]
        st.success(f"🎉 إشارة مكتشفة بنجاح: **{detected_word}** (نسبة الثقة: {confidence*100:.1f}%)")
        
        # زر تفاعلي لحقن الكلمة الصائبة في الجملة الكبيرة بالأعلى مباشرة وثباتها
        if st.button(f"➕ أضف كلمة '{detected_word}' للجملة المتسلسلة"):
            if not st.session_state.translated_sentence or st.session_state.translated_sentence[-1] != detected_word:
                st.session_state.translated_sentence.append(detected_word)
            st.rerun()
    except Exception as e:
        st.error(f"تنبيه: {e}")
