# ArSL — مساعد لغة الإشارة العربية والوصول
# ArSL — Arabic Sign Language & Accessibility Assistant

---

## 📖 نظرة عامة | Overview

### بالعربية

يهدف هذا المشروع إلى بناء نموذج ذكاء اصطناعي يتعرّف على إشارات لغة الإشارة العربية من خلال كاميرا حية ويترجمها إلى نص مباشر، باستخدام MediaPipe لاستخراج نقاط الجسم واليدين، ونموذج BiLSTM مدرَّب على بيانات KArSL. يتضمن المشروع أيضًا ميزة إضافية لوصف الصور صوتيًا لذوي الإعاقة البصرية.

### In English

This project aims to build an AI model that recognizes Arabic Sign Language gestures via a live camera and translates them into text in real time, using MediaPipe to extract body and hand landmarks, and a BiLSTM model trained on the KArSL dataset. The project also includes an additional feature that generates spoken descriptions of images for visually impaired users.

---

## ⚙️ الإعداد والتشغيل | Setup & Run

### بالعربية

```bash
python -m venv arsl_env
arsl_env\Scripts\activate
pip install -r requirements.txt
```

أنشئ ملف `.streamlit/secrets.toml` محليًا (لا تَرفعه أبدًا لـ GitHub):
```toml
OPENAI_API_KEY = "ضع_مفتاحك_هنا"
```

تشغيل التطبيق:
```bash
streamlit run src/app.py
```

### In English

```bash
python -m venv arsl_env
arsl_env\Scripts\activate
pip install -r requirements.txt
```

Create `.streamlit/secrets.toml` locally (never commit this file):
```toml
OPENAI_API_KEY = "your-key-here"
```

Run the app:
```bash
streamlit run src/app.py
```

---

## 📊 البيانات | Dataset

### بالعربية

يعتمد هذا المشروع على بيانات [KArSL](https://hamzah-luqman.github.io/KArSL/) - لم تُرفَع هنا بسبب حجمها الكبير وشروط الترخيص. حمّلها بشكل منفصل وضعها في `data/raw/`، ثم شغّل خلايا الاستخراج في نوت بوك التدريب لملء `data/processed/`.

### In English

This project uses the [KArSL](https://hamzah-luqman.github.io/KArSL/) dataset — not redistributed here due to size and licensing. Download it separately, place raw videos under `data/raw/`, then run the extraction cells in the training notebook to populate `data/processed/`.

---

## ⚠️ القيود المعروفة | Known Limitations

### بالعربية

يُظهر النموذج فجوة واضحة بين دقته على بيانات KArSL نفسها (المصوَّرة باستوديو Kinect V2 بإضاءة وخلفية وأشخاص ثابتين) وأدائه الفعلي أمام كاميرا ويب حقيقية - وهي فجوة تعميم (Domain Gap) متوافقة مع ما وثّقته أبحاث منشورة على نفس البيانات. لهذا السبب، يركّز المشروع حاليًا على عدد محدود من الكلمات.

### In English

The model shows a clear gap between its accuracy on KArSL's own data (studio-recorded with Kinect V2, fixed lighting/background/signers) and real-world webcam performance — a domain-generalization gap consistent with findings reported in published KArSL research. For this reason, the project currently focuses on a limited number of words.

