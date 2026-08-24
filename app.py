import streamlit as st

if not hasattr(st, "experimental_rerun"):
    st.experimental_rerun = st.rerun

from streamlit_webrtc import webrtc_streamer
import cv2
import numpy as np
import onnxruntime as ort
import pandas as pd
import os
import av

# 1. إعداد واجهة المستخدم السحابية الاحترافية لمشروع التخرج
st.set_page_config(page_title="Arabic Sign Language System", layout="centered")
st.title("🤖 نظام الترجمة الفورية الحية للغة الإشارة العربية (ArSL)")
st.write("الرابط السحابي نشط! الترجمة الفورية تظهر الآن مباشرة مكتوبة فوق شاشة الفيديو لايف.")

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
    return [f"Word {i+1}" for i in range(502)]

ort_session = load_onnx_session()
class_labels = load_labels()

# 2. كلاس معالجة دفق الفيديو المباشر فائق السرعة
class SignLanguageTransformer:
    def __init__(self):
        self.sequence_buffers = []
        self.max_frames = 5
        self.target_size = (64, 64)
        self.frame_counter = 0
        self.display_text = "Scanning..." # النص الافتراضي على الشاشة
        if ort_session is not None:
            # 🔥 تم تصحيح الكشاف هنا برمجياً بإضافة [0] لقراءة اسم المدخلات من القائمة بنجاح
            self.input_name = ort_session.get_inputs()[0].name

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1) # قلب الصورة كالمرآة لسهولة الاستخدام
        self.frame_counter += 1
        
        rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # أخذ لقطة كل إطارين لتسريع المعالجة السحابية وتقليل الـ Lag
        if self.frame_counter % 2 == 0:
            resized = cv2.resize(rgb_frame, self.target_size)
            normalized_frame = resized / 255.0
            self.sequence_buffers.append(normalized_frame)
            
            if len(self.sequence_buffers) > self.max_frames:
                self.sequence_buffers.pop(0)
                
            # تشغيل التنبؤ الفوري عبر ONNX
            if len(self.sequence_buffers) == self.max_frames and ort_session is not None:
                input_data = np.expand_dims(np.array(self.sequence_buffers, dtype=np.float32), axis=0)
                outputs = ort_session.run(None, {self.input_name: input_data})
                
                predictions = np.squeeze(outputs)
                predicted_class_idx = np.argmax(predictions)
                confidence = predictions[predicted_class_idx]
                
                # إذا تخطت نسبة الثقة 50% نقوم بتحديث النص المكتوب فوراً
                if confidence > 0.50:
                    try:
                        self.display_text = f"Sign: {class_labels[predicted_class_idx]}"
                    except:
                        self.display_text = f"Sign ID: {predicted_class_idx + 1}"
                else:
                    self.display_text = "Scanning..."

        # كتابة الكلمة المترجمة مباشرة فوق فيديو المستخدم لايف وبشكل فوري باللون الأخضر
        cv2.putText(img, self.display_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# 3. تشغيل ملقم البث السحابي
if ort_session is not None:
    processor = SignLanguageTransformer()
    webrtc_streamer(
        key="sign-language-translator-cloud-final-v5",
        video_frame_callback=processor.recv,
        media_stream_constraints={
            "video": True,
            "audio": False
        }
    )
