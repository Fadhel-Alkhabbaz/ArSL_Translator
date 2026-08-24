import streamlit as st
import cv2
import numpy as np
import onnxruntime as ort
import pandas as pd
import os
import time

# 1. إعداد واجهة المستخدم الاحترافية لمشروع التخرج
st.set_page_config(page_title="Arabic Sign Language System", layout="centered")
st.title("🤖 نظام الترجمة الفورية الحية للغة الإشارة العربية (ArSL)")
st.write("أدِّ الحركات المتتالية بوضوح أمام الكاميرا لبناء جملة كاملة ومستمرة باللغة العربية.")

# تحميل موديل ONNX والأسماء من ملف الإكسل مرة واحدة لحفظ الذاكرة
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
    return [f"إشارة {i+1}" for i in range(502)]

ort_session = load_onnx_session()
class_labels = load_labels()

# تهيئة ذاكرة الجملة التراكمية في الـ Session State لكي لا تختفي الكلمات
if "translated_sentence" not in st.session_state:
    st.session_state.translated_sentence = []

# عرض الجملة التراكمية الكبيرة في المقدمة بخط عريض ولون جذاب
st.markdown("---")
st.markdown("### 📝 الجملة المترجمة الحالية (مستمرة):")
sentence_placeholder = st.empty()

def update_sentence_display():
    full_sentence = " ".join(st.session_state.translated_sentence)
    if full_sentence:
        sentence_placeholder.markdown(f"<h1 style='text-align: center; color: #FF4B4B; direction: rtl;'>{full_sentence}</h1>", unsafe_allow_html=True)
    else:
        sentence_placeholder.markdown("<h3 style='text-align: center; color: #777;'>في انتظار تتبع الحركات الحية لبناء الجملة...</h3>", unsafe_allow_html=True)

update_sentence_display()

# زر مسح الجملة للبدء من جديد
if st.button("🗑️ مسح الجملة والبدء من جديد", type="secondary"):
    st.session_state.translated_sentence = []
    st.rerun()

st.markdown("---")

# حجز مكان عرض الكاميرا لايف داخل الصفحة
frame_placeholder = st.empty()

# 2. زر تشغيل الكاميرا المباشرة والترجمة الفورية
if st.button("📸 ابدأ البث الحي والترجمة الفورية داخل الصفحة", type="primary"):
    if ort_session is not None:
        input_name = ort_session.get_inputs()[0].name
        sequence_buffers = []
        max_frames = 5
        target_size = (64, 64)
        
        last_added_word = ""
        stability_counter = 0
        current_detected_word = ""
        frame_counter = 0 # عداد داخلي لتوسيع الفارق الزمني بين اللقطات

        cap = cv2.VideoCapture(0)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: 
                st.error("⚠️ فشل الاتصال بكاميرا الويب الشخصية.")
                break
                
            frame = cv2.flip(frame, 1) # قلب الصورة كالمرآة لسهولة استخدام يدك
            frame_counter += 1
            
            # تحويل الألوان وتجهيز الإطار (RGB)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # ضبط الفارق الزمني: نأخذ لقطة للموديل كل 3 إطارات للحصول على دفق حركة متكامل وصحيح
            if frame_counter % 3 == 0:
                resized = cv2.resize(rgb_frame, target_size)
                normalized_frame = resized / 255.0
                sequence_buffers.append(normalized_frame)
                
                if len(sequence_buffers) > max_frames:
                    sequence_buffers.pop(0)
                    
                # استدعاء التنبؤ الفوري عبر ONNX عند اكتمال الـ 5 إطارات المتباعدة زمنياً
                if len(sequence_buffers) == max_frames:
                    input_data = np.expand_dims(np.array(sequence_buffers, dtype=np.float32), axis=0)
                    outputs = ort_session.run(None, {input_name: input_data})
                    
                    predictions = np.squeeze(outputs)
                    predicted_class_idx = np.argmax(predictions)
                    confidence = predictions[predicted_class_idx]
                    
                    # تحليل عتبة الثقة لاعتماد الكلمة
                    if confidence > 0.55:
                        try:
                            detected_word = class_labels[predicted_class_idx]
                            
                            if detected_word == current_detected_word:
                                stability_counter += 1
                            else:
                                current_detected_word = detected_word
                                stability_counter = 0
                            
                            # آلية الاستقرار: إذا ثبتت يدك على الحركة لثوانٍ تُضاف الجملة فوراً
                            if stability_counter >= 2 and detected_word != last_added_word:
                                st.session_state.translated_sentence.append(detected_word)
                                last_added_word = detected_word
                                stability_counter = 0
                                update_sentence_display()
                        except: pass

            # تحديث عرض الجملة التراكمية على الشاشة متزامناً مع البث
            full_sentence_text = " ".join(st.session_state.translated_sentence)
            if full_sentence_text:
                sentence_placeholder.markdown(f"<h1 style='text-align: center; color: #FF4B4B; direction: rtl;'>{full_sentence_text}</h1>", unsafe_allow_html=True)
            
            # عرض البث الحي الطبيعي والصافي ليدك داخل متصفح الويب
            frame_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)
            
            time.sleep(0.01)
            
        cap.release()
