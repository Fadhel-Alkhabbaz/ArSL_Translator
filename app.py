import streamlit as st

# حل مشكلة التوافقية وإجبار الواجهة على التحديث الفوري سحابياً
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
st.title("🤖 نظام الترجمة الفورية الحية لبناء جمل لغة الإشارة العربية (ArSL)")
st.write("الرابط السحابي نشط! أدِّ الحركات متتالية أمام الكاميرا لبناء جملة كاملة ومستمرة.")

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

# تهيئة ذاكرة الجملة التراكمية السحابية لكي لا تختفي الكلمات نهائياً
if "translated_sentence" not in st.session_state:
    st.session_state.translated_sentence = []

# عرض الجملة الكبيرة المتسلسلة والمستمرة في المقدمة بشكل بارز
st.markdown("---")
st.markdown("### 📝 الجملة المترجمة الحالية (متسلسلة ومستمرة):")
sentence_placeholder = st.empty()

def refresh_sentence_ui():
    full_sentence = " ".join(st.session_state.translated_sentence)
    if full_sentence:
        sentence_placeholder.markdown(f"<h1 style='text-align: center; color: #FF4B4B; direction: rtl;'>{full_sentence}</h1>", unsafe_allow_html=True)
    else:
        sentence_placeholder.markdown("<h3 style='text-align: center; color: #777;'>قف أمام الكاميرا وابدأ بالإشارة لبناء الجملة...</h3>", unsafe_allow_html=True)

refresh_sentence_ui()

# زر مسح الجملة للبدء من جديد وتطهير الذاكرة السحابية فورا
if st.button("🗑️ مسح الجملة والبدء من جديد", type="secondary"):
    st.session_state.translated_sentence = []
    st.rerun()

st.markdown("---")

# 2. كلاس معالجة دفق الفيديو المباشر سحابياً وبناء الجمل التراكمية
class SignLanguageTransformer:
    def __init__(self):
        self.sequence_buffers = []
        self.max_frames = 5
        self.target_size = (64, 64)
        self.frame_counter = 0
        self.live_word = "Scanning..." # التوقع اللحظي على شاشة الكاميرا
        
        self.last_added_word = ""
        self.stability_counter = 0
        self.current_detected_word = ""
        
        if ort_session is not None:
            # 🟢 الحل البرمجي الجذري لفك القائمة وسحب الاسم بأمان تام على خوادم لينكس
            inputs = ort_session.get_inputs()
            self.input_name = inputs[0].name

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1) # قلب الصورة كالمرآة لسهولة استخدام اليدين
        self.frame_counter += 1
        
        rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # أخذ لقطة كل إطارين لتسريع الأداء السحابي ومنع الـ Lag تماماً
        if self.frame_counter % 2 == 0:
            resized = cv2.resize(rgb_frame, self.target_size)
            normalized_frame = resized / 255.0
            self.sequence_buffers.append(normalized_frame)
            
            if len(self.sequence_buffers) > self.max_frames:
                self.sequence_buffers.pop(0)
                
            # تشغيل التنبؤ الفوري والمستقر عبر ONNX
            if len(self.sequence_buffers) == self.max_frames and ort_session is not None:
                input_data = np.expand_dims(np.array(self.sequence_buffers, dtype=np.float32), axis=0)
                outputs = ort_session.run(None, {self.input_name: input_data})
                
                predictions = np.squeeze(outputs)
                predicted_class_idx = np.argmax(predictions)
                confidence = predictions[predicted_class_idx]
                
                # عتبة الثقة المعتمدة سحابياً لفلترة أي حركات عشوائية في الغرفة
                if confidence > 0.55:
                    try:
                        detected_word = class_labels[predicted_class_idx]
                        self.live_word = f"Sign: {detected_word} ({confidence*100:.0f}%)"
                        
                        # آلية تجميع الجمل المتسلسلة: يجب ثبات اليد لـ 3 لقطات متتالية للتأكيد
                        if detected_word == self.current_detected_word:
                            self.stability_counter += 1
                        else:
                            self.current_detected_word = detected_word
                            self.stability_counter = 0
                        
                        # التحديث المستمر الآمن: حقن الكلمة مباشرة في قائمة السيرفر المفتوحة لمنع القفل الأمني للمتصفح
                        if self.stability_counter >= 3 and detected_word != self.last_added_word:
                            if detected_word not in st.session_state.translated_sentence:
                                st.session_state.translated_sentence.append(detected_word)
                            self.last_added_word = detected_word
                            self.stability_counter = 0
                    except:
                        self.live_word = f"Sign ID: {predicted_class_idx + 1}"
                else:
                    self.live_word = "Scanning..."

        # كتابة الكلمة والـ ID اللحظي مباشرة فوق بث الكاميرا لايف باللون الأخضر الواضح
        cv2.putText(img, self.live_word, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# 3. تشغيل ملقم البث السحابي المحدث بكود كشاف مسح الكاش الصارم
if ort_session is not None:
    processor = SignLanguageTransformer()
    webrtc_streamer(
        key="sign-language-translator-cloud-final-v10", # تغيير الـ key لتطهير كاش المنصة حتماً وقراءة التحديث
        video_frame_callback=processor.recv,
        media_stream_constraints={
            "video": True,
            "audio": False
        }
    )
