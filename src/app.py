"""
تطبيق ويب (Streamlit) لترجمة لغة الإشارة العربية من الكاميرا الحية إلى جملة كاملة.

الملفات المطلوبة بجانب هذا السكربت (نفس ملفات live_inference.py):
    sign_lstm_model_best.pth
    idx_to_class.json
    karsl_labels.json
    pose_landmarker.task
    hand_landmarker.task

التثبيت (داخل نفس البيئة الافتراضية arsl_env التي أنشأناها):
    pip install streamlit streamlit-webrtc av

التشغيل:
    streamlit run app.py
"""

import json
import time
from collections import deque

import av
import cv2
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
from openai import OpenAI
from gtts import gTTS
import io

# ============================================================
# 0. الإعدادات الثابتة (يمكن التحكم ببعضها لاحقًا من الشريط الجانبي)
# ============================================================
MODEL_PATH = "sign_lstm_model_focused.pth"
IDX_TO_CLASS_PATH = "idx_to_class_focused.json"
LABELS_PATH = "karsl_labels.json"
POSE_MODEL_PATH = "pose_landmarker.task"
HAND_MODEL_PATH = "hand_landmarker.task"

TARGET_SEQ_LEN = 30
MIN_SEGMENT_FRAMES = 8
MAX_SEGMENT_FRAMES = 90
IDLE_FRAMES_TO_CLOSE = 8
DISPLAY_LANG_LABEL = "arabic"

RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})


# ============================================================
# 1. نفس دالة الاستخراج/التطبيع المستخدمة في التدريب بالضبط
# ============================================================
def extract_normalized_landmarks(pose_result, hand_result):
    if not pose_result.pose_landmarks or len(pose_result.pose_landmarks) == 0:
        return np.zeros((33 + 21 + 21) * 3, dtype=np.float32)

    pose_pts = np.array([[lm.x, lm.y, lm.z] for lm in pose_result.pose_landmarks[0]], dtype=np.float32)
    left_shoulder, right_shoulder = pose_pts[11], pose_pts[12]
    center = (left_shoulder + right_shoulder) / 2.0
    shoulder_dist = np.linalg.norm(left_shoulder - right_shoulder) or 1.0

    norm_pose = (pose_pts - center) / shoulder_dist
    norm_lh = np.zeros((21, 3), dtype=np.float32)
    norm_rh = np.zeros((21, 3), dtype=np.float32)

    if hand_result.hand_landmarks and hand_result.handedness:
        for hand_lms, handedness in zip(hand_result.hand_landmarks, hand_result.handedness):
            pts = np.array([[lm.x, lm.y, lm.z] for lm in hand_lms], dtype=np.float32)
            norm_pts = (pts - center) / shoulder_dist
            if handedness[0].category_name == "Left":
                norm_lh = norm_pts
            else:
                norm_rh = norm_pts

    return np.concatenate([norm_pose.flatten(), norm_lh.flatten(), norm_rh.flatten()])


def hands_motion_magnitude(prev_vec, curr_vec):
    if prev_vec is None:
        return 0.0
    return float(np.linalg.norm(curr_vec[99:] - prev_vec[99:]))


# ============================================================
# 2. النموذج + التحميل (مُخزَّن مؤقتًا Cached حتى لا يُعاد تحميله في كل تحديث)
# ============================================================
class SignLSTM(nn.Module):
    """يجب أن تطابق هذه البنية بالضبط بنية النموذج المحفوظ في .pth (BiLSTM + pooling)."""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True,
                             dropout=0.3, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2 * 2, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        mean_pool = out.mean(dim=1)
        max_pool, _ = out.max(dim=1)
        combined = torch.cat([mean_pool, max_pool], dim=1)
        return self.fc(combined)


@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model = SignLSTM(
        input_dim=checkpoint["input_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        output_dim=checkpoint["output_dim"],
        num_layers=checkpoint["num_layers"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, device


@st.cache_resource
def load_label_maps():
    with open(IDX_TO_CLASS_PATH, encoding="utf-8") as f:
        idx_to_class = json.load(f)
    with open(LABELS_PATH, encoding="utf-8") as f:
        class_labels = json.load(f)
    return idx_to_class, class_labels


def decode_prediction(predicted_idx, idx_to_class, class_labels):
    class_dir = idx_to_class[str(predicted_idx)]
    return class_labels[class_dir][DISPLAY_LANG_LABEL]


def describe_image_for_blind(image_bytes, api_key):
    """يرسل صورة مرفوعة من المستخدم لنموذج رؤية-لغة (VLM) ويطلب وصفًا تفصيليًا
    مناسبًا لشخص كفيف: العناصر الموجودة، الألوان، الأشخاص، أي نص مكتوب، والسياق العام.
    """
    import base64

    client = OpenAI(api_key=api_key)
    base64_image = base64.b64encode(image_bytes).decode("utf-8")

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "أنت مساعد وصفي متخصص لمساعدة الأشخاص المكفوفين أو ضعاف "
                    "البصر على فهم محتوى الصور. صف الصورة التالية بوضوح ودقة: "
                    "العناصر الرئيسية، الأشخاص وملامحهم العامة إن وُجدوا، الألوان، "
                    "المكان/السياق العام، وأي نص مكتوب داخل الصورة إن وُجد. "
                    "استخدم جملًا واضحة ومباشرة ومنظّمة، بدون مقدمات غير ضرورية."
                ),
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "صف هذه الصورة بالتفصيل."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                ],
            },
        ],
        max_tokens=500,
    )
    return response.choices[0].message.content.strip()


def text_to_speech_audio(text, lang="ar"):
    """يحوّل نصًا عربيًا إلى صوت مسموع (MP3 في الذاكرة، بدون حفظ ملف على القرص)."""
    buffer = io.BytesIO()
    tts = gTTS(text=text, lang=lang)
    tts.write_to_fp(buffer)
    buffer.seek(0)
    return buffer


def build_landmarkers():
    pose_options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=POSE_MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    hand_options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=HAND_MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return (
        vision.PoseLandmarker.create_from_options(pose_options),
        vision.HandLandmarker.create_from_options(hand_options),
    )


# ============================================================
# 3. معالج الفيديو (يعمل في Thread منفصل تديره streamlit-webrtc)
# ============================================================
class SignLanguageProcessor(VideoProcessorBase):
    def __init__(self):
        self.model, self.device = load_model()
        self.idx_to_class, self.class_labels = load_label_maps()
        self.pose_detector, self.hand_detector = build_landmarkers()

        self.segment_buffer = deque(maxlen=MAX_SEGMENT_FRAMES)
        self.prev_landmarks = None
        self.idle_streak = 0
        self.start_time = time.time()

        # هذه القيم يقرأها التطبيق الرئيسي مباشرة لعرضها في الواجهة
        self.sentence_words = []
        self.last_prediction_text = ""

        # قيم قابلة للتعديل حيًا من الشريط الجانبي (تُحدَّث من الـ main thread)
        self.motion_threshold = 0.010
        self.confidence_threshold = 0.6

    def clear_sentence(self):
        self.sentence_words = []

    def backspace_word(self):
        if self.sentence_words:
            self.sentence_words.pop()

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        timestamp_ms = int((time.time() - self.start_time) * 1000)

        resized = cv2.resize(img, (256, 256))
        image_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

        pose_result = self.pose_detector.detect_for_video(mp_image, timestamp_ms)
        hand_result = self.hand_detector.detect_for_video(mp_image, timestamp_ms)
        landmarks = extract_normalized_landmarks(pose_result, hand_result)

        # ---- تشخيص مؤقت: هل يتم رصد الجسم (Pose) فعليًا؟ ----
        pose_detected = bool(pose_result.pose_landmarks)
        print(f"Pose detected: {pose_detected} | Landmark vector mean abs: {np.abs(landmarks).mean():.5f}")

        motion = hands_motion_magnitude(self.prev_landmarks, landmarks)
        self.prev_landmarks = landmarks
        is_hand_visible = bool(hand_result.hand_landmarks)

        if is_hand_visible and (motion > self.motion_threshold or len(self.segment_buffer) < MIN_SEGMENT_FRAMES):
            self.segment_buffer.append(landmarks)
            self.idle_streak = 0
        else:
            self.idle_streak += 1

        segment_ready = (
            len(self.segment_buffer) >= MIN_SEGMENT_FRAMES
            and (self.idle_streak >= IDLE_FRAMES_TO_CLOSE or len(self.segment_buffer) >= MAX_SEGMENT_FRAMES)
        )

        if segment_ready:
            seq = np.array(self.segment_buffer, dtype=np.float32)
            resample_idx = np.linspace(0, len(seq) - 1, TARGET_SEQ_LEN, dtype=int)
            seq = seq[resample_idx]

            with torch.no_grad():
                x = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(self.device)
                logits = self.model(x)
                probs = torch.softmax(logits, dim=1)
                confidence, predicted_idx = probs.max(dim=1)
                confidence = confidence.item()
                predicted_idx = predicted_idx.item()

                # ---- تشخيص مؤقت: يطبع أعلى 3 احتمالات في الـ Terminal ----
                top3 = probs.topk(3)
                print("Top-3 confidences:", top3.values.tolist(), "| Classes:", top3.indices.tolist())

            if confidence >= self.confidence_threshold:
                word = decode_prediction(predicted_idx, self.idx_to_class, self.class_labels)
                self.sentence_words.append(word)
                self.last_prediction_text = f"{word} ({confidence*100:.0f}%)"
            else:
                self.last_prediction_text = f"? ({confidence*100:.0f}%) - تم تجاهلها"

            self.segment_buffer.clear()
            self.idle_streak = 0

        # الفيديو يبقى نظيفًا بدون أي نص فوقه - الجملة تُعرض بمكان منفصل في الواجهة (انظر الأسفل)
        display = cv2.flip(img, 1)
        return av.VideoFrame.from_ndarray(display, format="bgr24")


# ============================================================
# 4. واجهة Streamlit
# ============================================================
st.set_page_config(page_title="ArSL - ترجمة ووصول", page_icon="🤟", layout="centered")

# ---- تصميم مخصص: خط عربي أنيق (Cairo) + بطاقات + رأس متدرّج الألوان ----
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700;800&display=swap');

html, body, [class*="css"]  {
    font-family: 'Cairo', sans-serif !important;
    direction: rtl;
}

/* رأس الصفحة المتدرّج */
.hero-banner {
    background: linear-gradient(135deg, #7C3AED 0%, #4F46E5 50%, #0EA5E9 100%);
    border-radius: 20px;
    padding: 2.2rem 1.5rem;
    text-align: center;
    margin-bottom: 1.8rem;
    box-shadow: 0 10px 30px rgba(124, 58, 237, 0.35);
}
.hero-banner h1 {
    color: white;
    font-weight: 800;
    font-size: 2.1rem;
    margin: 0;
}
.hero-banner p {
    color: rgba(255,255,255,0.9);
    font-size: 1.05rem;
    margin-top: 0.5rem;
}

/* تبويبات أنيقة */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}
.stTabs [data-baseweb="tab"] {
    background-color: #1E293B;
    border-radius: 12px 12px 0 0;
    padding: 10px 20px;
    font-weight: 600;
    font-size: 1.05rem;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #7C3AED, #4F46E5) !important;
    color: white !important;
}

/* أزرار متدرّجة */
.stButton > button {
    background: linear-gradient(135deg, #7C3AED, #0EA5E9);
    color: white;
    border: none;
    border-radius: 12px;
    font-weight: 700;
    padding: 0.6rem 1.4rem;
    transition: transform 0.15s ease, box-shadow 0.15s ease;
    box-shadow: 0 4px 14px rgba(124, 58, 237, 0.35);
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(124, 58, 237, 0.5);
}

/* بطاقة الجملة المترجمة */
.sentence-card {
    background: linear-gradient(135deg, #1E293B, #0F172A);
    border: 1px solid #334155;
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
    margin-top: 0.5rem;
}
.sentence-card h2 {
    color: #A78BFA;
    font-size: 1.8rem;
    margin: 0;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero-banner">
    <h1>🤟 مساعد لغة الإشارة والوصول</h1>
    <p>ترجمة حية للغة الإشارة العربية، ووصف صوتي للصور لذوي الإعاقة البصرية</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### ⚙️ إعدادات الحساسية")
    motion_threshold = st.slider("عتبة الحركة (Motion Threshold)", 0.001, 0.05, 0.010, 0.001)
    confidence_threshold = st.slider("عتبة الثقة (Confidence Threshold)", 0.1, 0.95, 0.6, 0.05)
    st.caption("زد عتبة الحركة إذا كان يقطّع الإشارة الواحدة لعدة كلمات. قلّلها إذا كان لا يلتقط إشارات سريعة.")

# مفتاح OpenAI API يُقرأ تلقائيًا من .streamlit/secrets.toml — لا يُطلب من المستخدم إدخاله
openai_api_key = st.secrets.get("OPENAI_API_KEY", None)

tab1, tab2 = st.tabs(["🤟 ترجمة لغة الإشارة الحية", "🖼️ وصف الصور صوتيًا (للمكفوفين)"])

# ============================================================
# التبويب 1: الترجمة الحية للإشارة (كما كان سابقًا)
# ============================================================
with tab1:
    st.caption("اسمح للمتصفح بالوصول للكاميرا، ثم ابدأ بالإشارة. الجملة تظهر مباشرة فوق الفيديو.")

    ctx = webrtc_streamer(
        key="arsl-live-translation",
        video_processor_factory=SignLanguageProcessor,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
    )

    st.divider()
    st.subheader("📝 الجملة المترجمة")
    sentence_box = st.empty()
    last_word_box = st.empty()

    if ctx.video_processor:
        # مزامنة قيم الشريط الجانبي مع المعالج الفعلي في كل تحديث
        ctx.video_processor.motion_threshold = motion_threshold
        ctx.video_processor.confidence_threshold = confidence_threshold

        col1, col2 = st.columns(2)
        if col1.button("🗑️ مسح الجملة"):
            ctx.video_processor.clear_sentence()
        if col2.button("⌫ حذف آخر كلمة"):
            ctx.video_processor.backspace_word()

        # حلقة تحديث حية لصندوق الجملة طالما البث يعمل (بدون تحديث الصفحة كاملة)
        while ctx.state.playing:
            sentence_text = " ".join(ctx.video_processor.sentence_words) or "..."
            sentence_box.markdown(f'<div class="sentence-card"><h2>{sentence_text}</h2></div>', unsafe_allow_html=True)
            last_word_box.caption(f"آخر تنبؤ: {ctx.video_processor.last_prediction_text}")
            time.sleep(0.3)
    else:
        sentence_box.markdown('<div class="sentence-card"><h2>...</h2></div>', unsafe_allow_html=True)
        st.info("اضغط 'Start' أعلاه لتشغيل الكاميرا.")

# ============================================================
# التبويب 2: وصف الصور صوتيًا (للمكفوفين) — VLM + TTS
# ============================================================
with tab2:
    st.caption("ارفع صورة أو التقطها مباشرة، وسيصفها التطبيق لك بالتفصيل صوتيًا — مفيدة للأشخاص المكفوفين أو ضعاف البصر.")

    input_method = st.radio("طريقة إدخال الصورة", ["📁 رفع صورة", "📷 التقاط صورة بالكاميرا"], horizontal=True)

    if input_method == "📁 رفع صورة":
        image_source = st.file_uploader("اختر صورة", type=["jpg", "jpeg", "png"])
        caption_text = "الصورة المرفوعة"
    else:
        image_source = st.camera_input("التقط صورة")
        caption_text = "الصورة الملتقطة"

    if image_source is not None:
        st.image(image_source, caption=caption_text, use_container_width=True)

        if st.button("🔊 صف الصورة بصوت عالٍ"):
            if not openai_api_key:
                st.error("أدخل مفتاح OpenAI API أولًا من الشريط الجانبي.")
            else:
                with st.spinner("جارٍ تحليل الصورة..."):
                    try:
                        image_bytes = image_source.getvalue()
                        description = describe_image_for_blind(image_bytes, openai_api_key)
                    except Exception as e:
                        st.error(f"حدث خطأ أثناء تحليل الصورة: {e}")
                        description = None

                if description:
                    st.markdown(f"### الوصف:\n{description}")
                    with st.spinner("جارٍ تحويل الوصف إلى صوت..."):
                        try:
                            audio_buffer = text_to_speech_audio(description, lang="ar")
                            st.audio(audio_buffer, format="audio/mp3", autoplay=True)
                        except Exception as e:
                            st.error(f"حدث خطأ أثناء توليد الصوت: {e}")
    else:
        st.info("ارفع صورة أو التقطها من الأعلى للبدء.")