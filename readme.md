# 🎮 Robust Baloot Automation Bot

بوت تلقائي ذكي للعب **بلوت** و **جاكارو** على موقع [kammelna.com/baloot](https://www.kammelna.com/baloot).  
يضغط على الأزرار نيابة عنك، يتعرف على النصوص على الشاشة، ويتعامل مع التأخير أو الأخطاء.  
يحتوي على **لوحة تحكم عائمة** للتحكم بالتشغيل والإيقاف بسهولة.

---

## ⚠️ تنبيه مهم
- هذا البرنامج لأغراض **تعليمية وتجريبية فقط**.  
- تشغيله قد يخالف شروط الموقع، وقد يؤدي إلى حظر حسابك.  
- الموقع أحيانًا لا يعمل ويظهر رسالة: *"حصل خطأ فادح!"* — وقتها البوت لن يعمل حتى يعود الموقع.

---

## 📦 ما تحتاجه قبل البدء (Windows فقط)

1. **جهاز Windows 10 أو 11**  
2. **Python 3.9 أو أحدث**  
   - نزّل من [python.org](https://www.python.org/downloads/windows/)  
   - أثناء التثبيت اختر: ✅ **Add Python to PATH**  

3. **Google Chrome** مثبت  
   - نزّل من [google.com/chrome](https://www.google.com/chrome/)  

4. **Tesseract OCR** (للتعرف على النصوص العربية من الصور)  
   - نزّل من [Tesseract UB Mannheim (Windows installer)](https://github.com/UB-Mannheim/tesseract/wiki)  
   - أثناء التثبيت اختر اللغة: **Arabic**  
   - بعد التثبيت، يكون عادةً في:  
     ```
     C:\Program Files\Tesseract-OCR\tesseract.exe
     ```

---

## ⚙️ إعداد البيئة الافتراضية (مستحسن)
للتأكد من عدم وجود مشاكل مع المكتبات، من الأفضل استخدام **Virtual Environment**:

1. افتح **Command Prompt** أو **PowerShell** داخل مجلد المشروع.  
2. إنشاء البيئة الافتراضية:

```bash
python -m venv baloot_env
تفعيل البيئة:

bash
Copy code
baloot_env\Scripts\activate
بعد تفعيل البيئة، سترى اسم البيئة في بداية السطر (baloot_env).

تثبيت المكتبات المطلوبة داخل البيئة:

bash
Copy code
pip install opencv-python selenium webdriver-manager pytesseract Pillow pyautogui
🖼️ ملفات البوت
يجب أن يكون لديك ملفات الصور التالية بجانب baloot_automation.py:

play_baloot_template.png → زر "العب بلوت"

return_template.png → زر "عودة" (أخضر)

return_grey_template.png → زر "عودة" (رمادي)

leave_game_template.png → زر "مغادرة"

green_participate_button.png → زر "شارك / مشاركة" (أخضر)

claim_button_template.png → زر "استلم"

mouwafeq_template.png → زر "موافق"

📸 ملاحظة: إذا لم يتعرف البرنامج على الأزرار، التقط صورًا من الموقع (عند عمله) واحفظها بنفس الأسماء.

▶️ تشغيل البوت
افتح Command Prompt داخل مجلد المشروع.

إذا استخدمت البيئة الافتراضية، فعلها أولاً:

bash
Copy code
baloot_env\Scripts\activate
شغّل البوت:

bash
Copy code
python baloot_automation.py
سينفتح المتصفح تلقائيًا ويذهب إلى موقع البلوت.

ستظهر لوحة تحكم عائمة على الشاشة:

▶️ اضغط START لبدء التشغيل.

⏹️ اضغط STOP للإيقاف.

🛑 طرق إيقاف البوت
من لوحة التحكم اضغط ⏹️ STOP

أو اضغط Ctrl + C في Command Prompt

أو أغلق نافذة المتصفح

📁 مجلد التصحيح (Debug Screenshots)
يتم حفظ لقطات الشاشة التلقائية في المجلد:

Copy code
📁 final_debug_screenshots/
الهدف: تتبع الأخطاء ورؤية ما حصل في اللحظة نفسها

💡 من الجيد تنظيفه من وقت لآخر حتى لا يكبر حجمه

✅ كل شيء جاهز الآن!
عند تشغيل البوت لأول مرة قد يستغرق بضع ثوانٍ، وبعد ذلك سيعمل بسلاسة.

📝 ملاحظة إضافية
الملف الرئيسي للبوت: baloot_automation.py

يمكن تعديل مسارات الصور أو إعدادات الأزرار عند الحاجة عبر Command-line arguments:

bash
Copy code
python baloot_automation.py --claim claim_button_template.png --agree mouwafeq_template.png --back return_grey_template.png
هذا يسمح بتخصيص الصور بدون تعديل الكود مباشرة.