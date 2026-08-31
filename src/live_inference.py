"""
ترجمة لغة الإشارة العربية من الكاميرا الحية إلى جملة كاملة.

يعمل هذا الملف محليًا على جهازك (وليس داخل Colab) لأنه يحتاج وصول مباشر للكاميرا.

الملفات المطلوبة بجانب هذا السكربت (انسخها من Google Drive بعد التدريب):
    sign_lstm_model_best.pth
    idx_to_class.json
    karsl_labels.json
    pose_landmarker.task
    hand_landmarker.task

التثبيت:
    pip install opencv-python mediapipe torch numpy

التشغيل:
    python live_inference.py
"""

import json
import time
from collections import deque

import cv2
import numpy as np
import torch
import torch.nn as nn
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mediapipe as mp

# ============================================================
# 0. الإعدادات القابلة للتعديل
# ============================================================
MODEL_PATH = "sign_lstm_model_best.pth"
IDX_TO_CLASS_PATH = "idx_to_class.json"
LABELS_PATH = "karsl_labels.json"
POSE_MODEL_PATH = "pose_landmarker.task"
HAND_MODEL_PATH = "hand_landmarker.task"

TARGET_SEQ_LEN = 30        # يجب أن يطابق القيمة المستخدمة أثناء التدريب
CONFIDENCE_THRESHOLD = 0.6  # لا تُقبل الكلمة إلا إذا تجاوزت هذه الثقة
MIN_SEGMENT_FRAMES = 8      # أقل عدد إطارات لاعتبارها "إشارة" فعلية (وليس ضجيج)
MAX_SEGMENT_FRAMES = 90     # سقف أعلى لطول الإشارة الواحدة (لتفادي انتظار لا نهائي)
MOTION_THRESHOLD = 0.010    # عتبة حركة اليدين لتمييز "يتحرك" من "ثابت"
IDLE_FRAMES_TO_CLOSE = 8    # عدد الإطارات الثابتة المتتالية لاعتبار الإشارة انتهت
LANGUAGE = "arabic"         # أو "english"

DISPLAY_LANG_LABEL = "arabic" if LANGUAGE == "arabic" else "english"


# ============================================================
# 1. نفس دالة الاستخراج/التطبيع المستخدمة في التدريب بالضبط
#    (لا تُغيّر هذه الدالة إلا إذا غيّرتها أيضًا في نوت بوك التدريب)
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
    """مقياس بسيط لحركة اليدين فقط (يتجاهل الـ pose) بين إطارين متتاليين."""
    if prev_vec is None:
        return 0.0
    hands_prev = prev_vec[99:]  # أول 99 قيمة = pose (33*3)، والباقي = يدين (21+21)*3
    hands_curr = curr_vec[99:]
    return float(np.linalg.norm(hands_curr - hands_prev))


# ============================================================
# 2. تعريف النموذج (يطابق بنية التدريب)
# ============================================================
class SignLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def load_model(device):
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model = SignLSTM(
        input_dim=checkpoint["input_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        output_dim=checkpoint["output_dim"],
        num_layers=checkpoint["num_layers"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def load_label_maps():
    with open(IDX_TO_CLASS_PATH, encoding="utf-8") as f:
        idx_to_class = json.load(f)  # {"0": "0006", ...}
    with open(LABELS_PATH, encoding="utf-8") as f:
        class_labels = json.load(f)  # {"0006": {"arabic": "...", "english": "..."}}
    return idx_to_class, class_labels


def decode_prediction(predicted_idx, idx_to_class, class_labels):
    class_dir = idx_to_class[str(predicted_idx)]
    return class_labels[class_dir][DISPLAY_LANG_LABEL]


# ============================================================
# 3. تجهيز MediaPipe (وضع VIDEO لأداء أفضل مع الكاميرا الحية)
# ============================================================
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
    pose_detector = vision.PoseLandmarker.create_from_options(pose_options)
    hand_detector = vision.HandLandmarker.create_from_options(hand_options)
    return pose_detector, hand_detector


# ============================================================
# 4. الحلقة الرئيسية: كاميرا + تقطيع الإشارات + تجميع الجملة
# ============================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    model = load_model(device)
    idx_to_class, class_labels = load_label_maps()
    pose_detector, hand_detector = build_landmarkers()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("تعذر فتح الكاميرا. تأكد أنها متصلة وغير مستخدمة من برنامج آخر.")

    start_time = time.time()
    segment_buffer = deque(maxlen=MAX_SEGMENT_FRAMES)  # إطارات الإشارة الحالية قيد التجميع
    prev_landmarks = None
    idle_streak = 0

    sentence_words = []
    last_prediction_text = ""

    print("[INFO] اضغط 'c' لمسح الجملة، 'b' لحذف آخر كلمة، 'q' للخروج.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        timestamp_ms = int((time.time() - start_time) * 1000)

        # مهم: نستخرج الـ landmarks من الإطار الأصلي (غير المقلوب) حتى تتطابق
        # تسمية اليد اليسرى/اليمنى مع بيانات التدريب (KArSL غير مقلوبة).
        # القلب (flip) يُطبَّق فقط على نسخة العرض في الأسفل، وليس هنا.
        resized = cv2.resize(frame, (256, 256))
        image_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

        pose_result = pose_detector.detect_for_video(mp_image, timestamp_ms)
        hand_result = hand_detector.detect_for_video(mp_image, timestamp_ms)
        landmarks = extract_normalized_landmarks(pose_result, hand_result)

        motion = hands_motion_magnitude(prev_landmarks, landmarks)
        prev_landmarks = landmarks

        is_hand_visible = bool(hand_result.hand_landmarks)

        if is_hand_visible and (motion > MOTION_THRESHOLD or len(segment_buffer) < MIN_SEGMENT_FRAMES):
            # المستخدم لا يزال يشير (أو بداية إشارة جديدة) -> أضف الإطار واستمر بالتجميع
            segment_buffer.append(landmarks)
            idle_streak = 0
        else:
            idle_streak += 1

        segment_ready = (
            len(segment_buffer) >= MIN_SEGMENT_FRAMES
            and (idle_streak >= IDLE_FRAMES_TO_CLOSE or len(segment_buffer) >= MAX_SEGMENT_FRAMES)
        )

        if segment_ready:
            seq = np.array(segment_buffer, dtype=np.float32)
            resample_idx = np.linspace(0, len(seq) - 1, TARGET_SEQ_LEN, dtype=int)
            seq = seq[resample_idx]

            with torch.no_grad():
                x = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)  # (1, 30, 225)
                logits = model(x)
                probs = torch.softmax(logits, dim=1)
                confidence, predicted_idx = probs.max(dim=1)
                confidence = confidence.item()
                predicted_idx = predicted_idx.item()

            if confidence >= CONFIDENCE_THRESHOLD:
                word = decode_prediction(predicted_idx, idx_to_class, class_labels)
                sentence_words.append(word)
                last_prediction_text = f"{word} ({confidence*100:.0f}%)"
            else:
                last_prediction_text = f"? ({confidence*100:.0f}%) - تم تجاهلها"

            segment_buffer.clear()
            idle_streak = 0

        # ---- عرض الحالة على الشاشة ----
        sentence_display = " ".join(sentence_words) if sentence_words else "..."
        display_frame = cv2.flip(frame, 1)  # عرض مرآوي فقط، لا تأثير له على الاستدلال
        cv2.putText(display_frame, f"Sentence: {sentence_display}", (15, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display_frame, f"Last: {last_prediction_text}", (15, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
        cv2.putText(display_frame, f"Buffer: {len(segment_buffer)} frames", (15, 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(display_frame, "c=clear  b=backspace  q=quit", (15, display_frame.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        cv2.imshow("ArSL Live Translation", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("c"):
            sentence_words.clear()
        elif key == ord("b") and sentence_words:
            sentence_words.pop()

    cap.release()
    cv2.destroyAllWindows()
    pose_detector.close()
    hand_detector.close()


if __name__ == "__main__":
    main()
