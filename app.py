import streamlit as st

# حل مشكلة التوافقية للإصدارات الحديثة في Streamlit السحابي
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
st.write("الرابط السحابي نشط الآن! اسمح للمتصفح بفتح الكاميرا لبدء الترجمة وبناء الجمل.")

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
    return [f"إشارة {i+1}" for i in range(502)]

ort_session = load_onnx_session()
class_labels = load_labels()

# تهيئة ذاكرة الجملة التراكمية في الـ Session State
if "translated_sentence" not in st.session_state:
    st.session_state.translated_sentence = []

# عرض الجملة التراكمية الكبيرة في المقدمة
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

# 2. كلاس معالجة دفق الفيديو المباشر سحابياً عبر بروتوكول WebRTC
class SignLanguageTransformer:
    def __init__(self):
        self.sequence_buffers = []
        self.max_frames = 5
        self.target_size = (64, 64)
        self.frame_counter = 0
        if ort_session is not None:
            self.input_name = ort_session.get_inputs()[0].name

    def recv(self, frame):
        # سحب الإطار الحالي كمصفوفة NumPy مباشرة من بث متصفح المستخدم
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1) # قلب الصورة كالمرآة لراحة المستخدم
        self.frame_counter += 1
        
        # تحويل الألوان وتجهيز الإطار (RGB) المتطابق مع أوزان الموديل
        rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # المؤقت الحركي المتباعد: نأخذ لقطة كل 3 إطارات لتأمين الحركة كاملة للـ GRU
        if self.frame_counter % 3 == 0:
            resized = cv2.resize(rgb_frame, self.target_size)
            normalized_frame = resized / 255.0
            self.sequence_buffers.append(normalized_frame)
            
            if len(self.sequence_buffers) > self.max_frames:
                self.sequence_buffers.pop(0)
                
            # استدعاء التنبؤ الفوري سحابياً عبر ONNX Runtime الخفيف
            if len(self.sequence_buffers) == self.max_frames and ort_session is not None:
                input_data = np.expand_dims(np.array(self.sequence_buffers, dtype=np.float32), axis=0)
                outputs = ort_session.run(None, {self.input_name: input_data})
                
                predictions = np.squeeze(outputs)
                predicted_class_idx = np.argmax(predictions)
                confidence = predictions[predicted_class_idx]
                
                # عتبة الثقة المعتمدة سحابياً
                if confidence > 0.55:
                    try:
                        detected_word = class_labels[predicted_class_idx]
                        # حقن الكلمة المكتشفة في قائمة المتصفح لكي لا تختفي إطلاقاً
                        if detected_word not in st.session_state.translated_sentence:
                            st.session_state.translated_sentence.append(detected_word)
                    except: pass

        # إعادة الإطار المعالج للبث المباشر عبر مكتبة PyAV السحابية سريعة التدفق
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# 3. تشغيل ملقم البث المباشر المحدث والمقيد بخوادم STUN الرسمية لقوقل لضمان الاتصال السحابي
if ort_session is not None:
    processor = SignLanguageTransformer()
    webrtc_streamer(
        key="sign-language-translator-cloud",
        video_frame_callback=processor.recv,
        # خوادم الـ STUN ضرورية جداً سحابياً لتوصيل كاميرا اللابتوب بسيرفر لينكس عن بعد
        rtc_configuration={"iceServers": [{"urls": ["stun:://google.com"]}]},
        media_stream_constraints={
            "video": {
                "width": {"ideal": 640},
                "height": {"ideal": 480},
                "frameRate": {"ideal": 30}
            },
            "audio": False
        }
    )
