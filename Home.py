import streamlit as st

st.set_page_config(
    page_title="Thai Corpus Tools",
    page_icon="🐜",
    layout="wide"
)

st.markdown("""
<style>
    .stApp { background-color: #F8F9FA; color: #212529; }
    [data-testid="stSidebar"] { background-color: #E9ECEF; }
    .workflow-box {
        background: white;
        border-radius: 10px;
        padding: 20px 25px;
        border: 1px solid #dee2e6;
        margin-bottom: 10px;
    }
    .step-num {
        font-size: 2em;
        font-weight: bold;
        color: #0d6efd;
    }
    .arrow {
        text-align: center;
        font-size: 2em;
        color: #adb5bd;
        margin: 0;
    }
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.title("🐜 Thai Corpus Tools")
st.subheader("ชุดเครื่องมือวิเคราะห์คลังข้อมูลภาษาไทย")
st.caption("พัฒนาด้วย Python + Streamlit + PyThaiNLP 5.x")

st.divider()

# --- Workflow Overview ---
st.header("📌 ขั้นตอนการใช้งาน (Workflow)")
st.write("ใช้งานตามลำดับขั้นตอน หรือเริ่มจากขั้นตอนที่ต้องการได้เลย")

col1, col2, col3, col4, col5 = st.columns([3, 1, 3, 1, 3])

with col1:
    st.markdown("""
    <div class="workflow-box">
        <div class="step-num">✂️ ขั้นที่ 1</div>
        <h4>Word Segmenter</h4>
        <p>ตัดคำภาษาไทยอัตโนมัติ รองรับหลายไฟล์พร้อมกัน ดาวน์โหลดเป็น ZIP</p>
        <hr/>
        <b>Input:</b> ไฟล์ .txt ดิบ<br/>
        <b>Output:</b> ไฟล์ .txt ตัดคำด้วย | คั่น
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown('<div class="arrow">→</div>', unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="workflow-box">
        <div class="step-num">🏷️ ขั้นที่ 2</div>
        <h4>POS Tagger</h4>
        <p>กำกับชนิดคำ (POS) อัตโนมัติด้วยชุด ORCHID (33 tags) ของ PyThaiNLP</p>
        <hr/>
        <b>Input:</b> ไฟล์ตัดคำแล้ว (| คั่น)<br/>
        <b>Output:</b> ไฟล์ คำ/TAG | คั่น
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown('<div class="arrow">→</div>', unsafe_allow_html=True)

with col5:
    st.markdown("""
    <div class="workflow-box">
        <div class="step-num">🐜 ขั้นที่ 3</div>
        <h4>Concordance</h4>
        <p>วิเคราะห์คลังข้อมูล KWIC, Word List, N-Grams, Word Cloud, MI Score</p>
        <hr/>
        <b>Input:</b> ไฟล์ดิบ / ตัดคำ / มี POS<br/>
        <b>Output:</b> ผลวิเคราะห์ + CSV
    </div>
    """, unsafe_allow_html=True)

st.divider()

# --- File Format Guide ---
st.header("📄 รูปแบบไฟล์ที่รองรับ")

col_a, col_b, col_c = st.columns(3)

with col_a:
    st.markdown("#### ไฟล์ดิบ (Raw Text)")
    st.code("ฉันกินข้าวกับเพื่อน\nวันนี้อากาศดีมาก", language=None)
    st.caption("ข้อความธรรมดา ยังไม่ตัดคำ")

with col_b:
    st.markdown("#### ไฟล์ตัดคำแล้ว (Segmented)")
    st.code("ฉัน|กิน|ข้าว|กับ|เพื่อน\nวันนี้|อากาศ|ดี|มาก", language=None)
    st.caption("ใช้ | คั่นระหว่างคำ")

with col_c:
    st.markdown("#### ไฟล์มี POS (POS Tagged)")
    st.code("ฉัน/PRON|กิน/VACT|ข้าว/NCMN|กับ/RPRE|เพื่อน/NCMN", language=None)
    st.caption("รูปแบบ คำ/TAG คั่นด้วย | (ORCHID tagset)")

st.divider()

# --- POS Tagset Info ---
st.header("🏷️ ชุด POS Tag ที่ใช้ — ORCHID Tagset")
st.caption("PyThaiNLP 5.x รองรับ ORCHID tagset ผ่าน engine: perceptron")

col_p1, col_p2, col_p3, col_p4 = st.columns(4)

with col_p1:
    st.markdown("""
    **คำนาม (Nouns)**
    - `NCMN` — นามทั่วไป
    - `NPRP` — นามเฉพาะ
    - `NTTL` — ตำแหน่ง/คำนำหน้าชื่อ
    - `NCNM` — จำนวนนับ
    - `CLAS` — ลักษณนาม
    """)

with col_p2:
    st.markdown("""
    **กริยา (Verbs)**
    - `VACT` — กริยาการกระทำ
    - `VSTA` — กริยาสภาวะ (เป็น, อยู่)
    - `VAUX` — กริยาช่วย (จะ, ต้อง)
    """)

with col_p3:
    st.markdown("""
    **คุณศัพท์ / วิเศษณ์**
    - `ADJE` — คุณศัพท์
    - `ADVN` — วิเศษณ์
    - `ADVI` — วิเศษณ์บอกระดับ
    - `ADVP` — อนุภาควิเศษณ์
    """)

with col_p4:
    st.markdown("""
    **อื่น ๆ**
    - `PRON` — สรรพนาม
    - `RPRE` — บุพบท (ใน, ของ)
    - `CONJ` — สันธาน (และ, หรือ)
    - `NEGA` — ปฏิเสธ (ไม่)
    - `PART` — คำลงท้าย (ครับ, ค่ะ)
    """)

st.info("📖 ดูตารางอ้างอิง ORCHID tags ทั้งหมดได้ที่หน้า **🏷️ POS Tagger**")

st.divider()

# --- POS Search Syntax ---
st.header("🔍 POS Search Syntax (ใน Concordance)")
st.caption("ใช้ได้เฉพาะไฟล์ที่ผ่านการกำกับ POS แล้ว")

col_s1, col_s2 = st.columns(2)

with col_s1:
    st.markdown("""
    | รูปแบบ | ความหมาย |
    |--------|----------|
    | `กิน` | Exact match ปกติ |
    | `กิน/VACT` | คำว่า "กิน" ที่เป็น VACT เท่านั้น |
    | `/NCMN` | ทุกคำที่เป็น Noun (NCMN) |
    | `รัก*/VACT` | คำขึ้นต้น "รัก" ที่เป็น VACT |
    """)

with col_s2:
    st.markdown("""
    | รูปแบบ | ความหมาย |
    |--------|----------|
    | `กิน/VACT /NCMN` | "กิน" (VACT) ตามด้วย Noun ใดก็ได้ |
    | `กิน /NCMN` | "กิน" ตามด้วย Noun |
    | `กิน <2> /NCMN` | "กิน" ห่างจาก Noun 2 คำ |
    | `/VACT /NCMN` | Verb ใดก็ได้ ตามด้วย Noun |
    """)

st.divider()

# --- Tools Summary ---
st.header("🔧 สรุปเครื่องมือทั้งหมด")

st.markdown("""
| หน้า | เครื่องมือ | ฟังก์ชันหลัก |
|------|-----------|-------------|
| ✂️ Word Segmenter | ตัดคำภาษาไทย | newmm / longest / multi_cut · batch process · download ZIP |
| 🏷️ POS Tagger | กำกับชนิดคำ | ORCHID tagset · ตารางอ้างอิง · download ZIP |
| 🐜 Concordance | วิเคราะห์คลังข้อมูล | KWIC · Word List · N-Grams · Word Cloud · Tag Cloud · MI Score · Corpus Stats |
""")

st.divider()
st.caption("💡 เริ่มต้นใช้งานโดยเลือกเครื่องมือจาก Sidebar ทางด้านซ้ายมือ")
