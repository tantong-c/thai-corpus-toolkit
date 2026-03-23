# 🐜 Thai Corpus Tools — Multipage App

ชุดเครื่องมือวิเคราะห์คลังข้อมูลภาษาไทย พัฒนาด้วย Python + Streamlit + PyThaiNLP

---

## 📁 โครงสร้างโปรเจกต์

```
corpus_app/
│
├── Home.py                      ← หน้าแรก + อธิบาย workflow
│
├── pages/
│   ├── 1_Word_Segmenter.py      ← ตัดคำภาษาไทย (batch)
│   ├── 2_POS_Tagger.py          ← กำกับชนิดคำ (ORCHID tagset)
│   └── 3_Concordance.py         ← วิเคราะห์คลังข้อมูล
│
├── requirements.txt
├── README.md
├── .gitignore
└── Sarabun-Regular.ttf          ← font ภาษาไทย (ดาวน์โหลดเอง)
```

---

## ⚠️ ข้อกำหนดระบบ

| รายการ | เวอร์ชันที่แนะนำ |
|--------|----------------|
| Python | **3.10 – 3.11** (แนะนำ 3.11) |
| PyThaiNLP | 5.1.x ขึ้นไป |

> **หมายเหตุ:** Python 3.13 ยังไม่รองรับ dependency บางตัวของ PyThaiNLP
> แนะนำให้ใช้ Python 3.11 เพื่อความเสถียรสูงสุด

---

## 🚀 การติดตั้ง

```bash
# 1. สร้าง virtual environment (Python 3.11)
py -3.11 -m venv venv311       # Windows
python3.11 -m venv venv311     # Mac/Linux

# 2. Activate
venv311\Scripts\activate       # Windows
source venv311/bin/activate    # Mac/Linux

# 3. ติดตั้ง dependencies
pip install -r requirements.txt

# 4. รัน
streamlit run Home.py
```

---

## 📄 รูปแบบไฟล์และ Workflow

```
ไฟล์ดิบ (.txt)
    ↓  [✂️ Word Segmenter]
ไฟล์ตัดคำ:  คำ|คำ|คำ
    ↓  [🏷️ POS Tagger]
ไฟล์มี POS: คำ/TAG|คำ/TAG
    ↓  แก้ไขด้วยมือได้ก่อนนำไปใช้
    ↓  [🐜 Concordance]
ผลวิเคราะห์ + Export CSV
```

---

## 🏷️ POS Tagset — ORCHID

โปรแกรมใช้ **ORCHID tagset** ผ่าน PyThaiNLP `engine="perceptron", corpus="orchid"`

> PyThaiNLP 5.x บน Python 3.13 รองรับเฉพาะ ORCHID tagset
> LST20 และ UD จะรองรับเมื่ออัปเกรด Python เป็น 3.10–3.11

### Tags ที่พบบ่อย

| Tag | ความหมาย | ตัวอย่าง |
|-----|---------|---------|
| `NCMN` | คำนามทั่วไป | บ้าน, รถ, ความรัก |
| `NPRP` | คำนามเฉพาะ | กรุงเทพ, ไทย |
| `VACT` | กริยาการกระทำ | กิน, วิ่ง, เขียน |
| `VSTA` | กริยาสภาวะ | เป็น, อยู่, คือ |
| `VAUX` | กริยาช่วย | จะ, ต้อง, ควร |
| `ADJE` | คำคุณศัพท์ | สวย, ใหญ่, แดง |
| `ADVN` | คำวิเศษณ์ | เร็ว, ช้า, มาก |
| `PRON` | คำสรรพนาม | ฉัน, เขา, เรา |
| `RPRE` | คำบุพบท | ใน, บน, ของ, กับ |
| `CONJ` | คำสันธาน | และ, หรือ, แต่ |
| `NEGA` | คำปฏิเสธ | ไม่, ไม่ได้ |
| `PART` | คำอนุภาค/ลงท้าย | ครับ, ค่ะ, นะ |
| `CLAS` | คำลักษณนาม | คน, ตัว, เล่ม |

ดูตารางอ้างอิงทั้งหมดได้ในหน้า **🏷️ POS Tagger** ของโปรแกรม

---

## 🔍 POS Search Syntax (ใน Concordance)

| รูปแบบ | ความหมาย |
|--------|----------|
| `กิน` | Exact match ปกติ |
| `กิน/VACT` | คำว่า "กิน" ที่เป็น VACT เท่านั้น |
| `/NCMN` | ทุกคำที่เป็น Noun |
| `รัก*/VACT` | คำขึ้นต้น "รัก" ที่เป็น VACT |
| `กิน/VACT /NCMN` | "กิน" (VACT) ตามด้วย Noun ใดก็ได้ |
| `/VACT /NCMN` | Verb ใดก็ได้ ตามด้วย Noun |

---

## 🔧 ฟีเจอร์ทั้งหมด

| Tab | ฟีเจอร์ | คำอธิบาย |
|-----|---------|----------|
| 🔍 Concordance | KWIC | ค้นหาคำในบริบท + POS Search |
| 📊 Word List | Word Frequency | ความถี่ + % + Cumulative % + POS |
| 🔗 N-Grams | Cluster Analysis | วิเคราะห์กลุ่มคำ 2–5 คำ |
| ☁️ Word Cloud | Visual Cloud | ภาพแสดงความถี่คำ + Download PNG |
| 🏷️ Tag Cloud | Interactive Tags | Tag Cloud แบบ Interactive |
| 📐 Collocations | MI Score | Mutual Information + ระบุไฟล์ที่พบ |
| 📈 Corpus Stats | Statistics | TTR, Hapax, POS Distribution |
| 📄 File Content | Tokenized View | แสดงเนื้อหาไฟล์ที่ผ่านการประมวลผล |

---

## 📦 Font ภาษาไทย

ดาวน์โหลด **Sarabun** จาก https://fonts.google.com/specimen/Sarabun
วางไฟล์ `Sarabun-Regular.ttf` ไว้ใน root folder ของโปรเจกต์

---

## ☁️ Deploy บน Streamlit Community Cloud

1. Push โค้ดขึ้น GitHub (รวม `requirements.txt` และ `Sarabun-Regular.ttf`)
2. ไปที่ [share.streamlit.io](https://share.streamlit.io)
3. เชื่อมต่อ repository และเลือกไฟล์ `Home.py`
4. กด **Deploy**

> Streamlit Cloud ใช้ Python 3.11 โดย default ซึ่งรองรับ PyThaiNLP ได้เต็มที่

---

## 📄 License

MIT License — ใช้งานและดัดแปลงได้อย่างอิสระ