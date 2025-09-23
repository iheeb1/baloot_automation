# 🎮 Robust Baloot Automation Bot

بوت تلقائي ذكي للعب **بلوت** و **جاكارو** على موقع [kammelna.com/baloot](https://www.kammelna.com/baloot).

يضغط على الأزرار نيابة عنك، يتعرف على النصوص على الشاشة، ويتعامل مع التأخير أو الأخطاء.
يحتوي على **لوحة تحكم عائمة** للتحكم بالتشغيل والإيقاف بسهولة.

---

## 🧪 ملفات الاختبار

يحتوي المشروع على ملفين للاختبار:

### `test_claim_button.py`
- **الغرض**: اختبار وظيفة النقر على الأزرار
- **الاستخدام**: للتأكد من عمل آلية النقر بشكل صحيح
- **التشغيل**:
  ```bash
  python test_claim_button.py
  ```

### `test_path_detection.py`
- **الغرض**: اختبار كشف مسارات الملفات والصور
- **الاستخدام**: للتأكد من وجود جميع ملفات الصور المطلوبة
- **التشغيل**:
  ```bash
  python test_path_detection.py
  ```

> 💡 **نصيحة**: شغّل ملفات الاختبار قبل تشغيل البوت الرئيسي للتأكد من أن كل شيء يعمل بشكل صحيح.



---

## ⚠️ تنبيه مهم

- هذا البرنامج لأغراض **تعليمية وتجريبية فقط**.
- تشغيله قد يخالف شروط الموقع، وقد يؤدي إلى حظر حسابك.
- الموقع أحيانًا لا يعمل ويظهر رسالة: *"حصل خطأ فادح!"* — وقتها البوت لن يعمل حتى يعود الموقع.

---

## 📦 متطلبات النظام

### النظام المدعوم
- **Windows 10 أو 11** (مطلوب)

### البرامج المطلوبة

#### 1. Python 3.9 أو أحدث
- نزّل من [python.org](https://www.python.org/downloads/windows/)
- **مهم**: أثناء التثبيت اختر ✅ **Add Python to PATH**

#### 2. Google Chrome
- نزّل من [google.com/chrome](https://www.google.com/chrome/)

#### 3. Tesseract OCR (للتعرف على النصوص العربية)
- نزّل من [Tesseract UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki)
- أثناء التثبيت اختر اللغة: **Arabic**
- بعد التثبيت، يكون عادةً في: `C:\Program Files\Tesseract-OCR\tesseract.exe`

---

## ⚙️ إعداد البيئة الافتراضية (مستحسن)

للتأكد من عدم وجود مشاكل مع المكتبات، من الأفضل استخدام **Virtual Environment**:

### 1. إنشاء البيئة الافتراضية
افتح **Command Prompt** أو **PowerShell** داخل مجلد المشروع:

```bash
python -m venv baloot_env
```

### 2. تفعيل البيئة
```bash
baloot_env\Scripts\activate
```

بعد تفعيل البيئة، سترى اسم البيئة في بداية السطر `(baloot_env)`.

### 3. تثبيت المكتبات المطلوبة
```bash
pip install opencv-python selenium webdriver-manager pytesseract Pillow pyautogui
```

---

## 🖼️ ملفات الصور المطلوبة

يجب أن تكون الملفات التالية موجودة في نفس مجلد `baloot_automation.py`:

| اسم الملف | الوصف |
|-----------|--------|
| `play_baloot_template.png` | زر "العب بلوت" |
| `return_template.png` | زر "عودة" (أخضر) |
| `return_grey_template.png` | زر "عودة" (رمادي) |
| `leave_game_template.png` | زر "مغادرة" |
| `green_participate_button.png` | زر "شارك / مشاركة" (أخضر) |
| `claim_button_template.png` | زر "استلم" |
| `mouwafeq_template.png` | زر "موافق" |

> 📸 **ملاحظة**: إذا لم يتعرف البرنامج على الأزرار، التقط صورًا من الموقع (عند عمله) واحفظها بنفس الأسماء.

---

## ▶️ تشغيل البوت

### 1. فتح Command Prompt
افتح **Command Prompt** داخل مجلد المشروع.

### 2. تفعيل البيئة الافتراضية (إذا استخدمتها)
```bash
baloot_env\Scripts\activate
```

### 3. تشغيل البوت
```bash
python baloot_automation.py
```

### 4. استخدام لوحة التحكم
- سينفتح المتصفح تلقائيًا ويذهب إلى موقع البلوت
- ستظهر لوحة تحكم عائمة على الشاشة:
  - ▶️ اضغط **START** لبدء التشغيل
  - ⏹️ اضغط **STOP** للإيقاف

---

## 🛑 طرق إيقاف البوت

يمكنك إيقاف البوت بإحدى الطرق التالية:

1. من لوحة التحكم اضغط ⏹️ **STOP**
2. اضغط `Ctrl + C` في Command Prompt
3. أغلق نافذة المتصفح

---

## 📁 مجلد التصحيح

يتم حفظ لقطات الشاشة التلقائية في المجلد:
```
📁 final_debug_screenshots/
```

**الهدف**: تتبع الأخطاء ورؤية ما حصل في اللحظة نفسها

💡 **نصيحة**: من الجيد تنظيف هذا المجلد من وقت لآخر حتى لا يكبر حجمه

---

## 🔧 خيارات متقدمة

### تخصيص مسارات الصور
يمكن تعديل مسارات الصور عبر Command-line arguments:

```bash
python baloot_automation.py --claim claim_button_template.png --agree mouwafeq_template.png --back return_grey_template.png
```

هذا يسمح بتخصيص الصور بدون تعديل الكود مباشرة.

---

## 🗂️ هيكل المشروع

```
📁 baloot-automation/
├── 📄 baloot_automation.py          # الملف الرئيسي للبوت
├── 📄 test_claim_button.py          # ملف اختبار النقر على الأزرار
├── 📄 test_path_detection.py        # ملف اختبار كشف المسارات
├── 📄 README.md                     # هذا الملف
├── 📁 baloot_env/                   # البيئة الافتراضية (اختياري)
├── 📁 final_debug_screenshots/      # لقطات الشاشة للتصحيح
├── 🖼️ play_baloot_template.png
├── 🖼️ return_template.png
├── 🖼️ return_grey_template.png
├── 🖼️ leave_game_template.png
├── 🖼️ green_participate_button.png
├── 🖼️ claim_button_template.png
└── 🖼️ mouwafeq_template.png
```

---

## 🚀 البدء السريع

1. **تثبيت Python** مع تفعيل "Add to PATH"
2. **تثبيت Chrome و Tesseract**
3. **فتح Command Prompt** في مجلد المشروع
4. **إنشاء البيئة الافتراضية**:
   ```bash
   python -m venv baloot_env
   baloot_env\Scripts\activate
   ```
5. **تثبيت المكتبات**:
   ```bash
   pip install opencv-python selenium webdriver-manager pytesseract Pillow pyautogui
   ```
6. **وضع ملفات الصور** في نفس المجلد
7. **تشغيل البوت**:
   ```bash
   python baloot_automation.py
   ```

---

## ❓ استكشاف الأخطاء

### البوت لا يتعرف على الأزرار
- تأكد من وجود جميع ملفات الصور
- التقط صورًا جديدة من الموقع إذا تغير التصميم

### البوت لا يبدأ
- تأكد من تثبيت جميع المكتبات المطلوبة
- تأكد من تفعيل البيئة الافتراضية

### الموقع لا يعمل
- انتظر حتى يعود الموقع للعمل
- تحقق من اتصالك بالإنترنت

---

## 📞 الدعم

إذا واجهت مشاكل، تحقق من:
- مجلد `final_debug_screenshots/` لرؤية لقطات الشاشة
- رسائل الأخطاء في Command Prompt
- التأكد من أن الموقع يعمل بشكل طبيعي

---

**✅ كل شيء جاهز الآن!**

عند تشغيل البوت لأول مرة قد يستغرق بضع ثوانٍ، وبعد ذلك سيعمل بسلاسة.