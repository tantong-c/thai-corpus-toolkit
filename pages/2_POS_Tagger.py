import streamlit as st
import pandas as pd
import io
import zipfile
import re
import os
from pythainlp.tag import pos_tag

st.set_page_config(page_title="POS Tagger", page_icon="🏷️", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #F8F9FA; color: #212529; }
    [data-testid="stSidebar"] { background-color: #E9ECEF; }
    th { background-color: #DEE2E6 !important; color: #212529 !important; }
</style>
""", unsafe_allow_html=True)

st.title("🏷️ POS Tagger")
st.caption(
    "กำกับชนิดคำ (Part-of-Speech Tagging) สำหรับภาษาไทย — ใช้ชุด ORCHID (PyThaiNLP)")

# --- ORCHID Tag Reference ---
ORCHID_TAGS = [
    ("NCMN", "Common Noun",           "คำนามทั่วไป",              "บ้าน, รถ, หนังสือ"),
    ("NPRP", "Proper Noun",           "คำนามเฉพาะ",
     "กรุงเทพ, สมชาย, ไทย"),
    ("NTTL", "Title Noun",            "คำนามที่เป็นตำแหน่ง",      "นาย, นาง, ดร., ศ."),
    ("NLBL", "Label Noun",            "คำนามที่เป็นป้าย/สัญลักษณ์", "ก., ข., 1., 2."),
    ("NCNM", "Cardinal Numeral",      "จำนวนนับ",
     "หนึ่ง, สอง, สาม, 1, 2"),
    ("NONM", "Ordinal Numeral",       "จำนวนเรียงลำดับ",           "ที่หนึ่ง, ที่สอง"),
    ("NMNM", "Monetary Numeral",      "จำนวนเงิน",                 "หนึ่งบาท, ร้อยบาท"),
    ("VACT", "Active Verb",           "กริยาแสดงการกระทำ",
     "กิน, วิ่ง, เขียน, พูด"),
    ("VSTA", "Stative Verb",          "กริยาแสดงสภาวะ",            "เป็น, อยู่, คือ"),
    ("VAUX", "Auxiliary Verb",        "กริยาช่วย",
     "จะ, ควร, ต้อง, อาจ"),
    ("ADVN", "Adverb",                "คำวิเศษณ์",
     "เร็ว, ช้า, ดี, มาก"),
    ("ADVP", "Adverbial Particle",    "อนุภาควิเศษณ์",
     "ขึ้น, ลง, ออก, เข้า, มา"),
    ("ADVI", "Intensifier Adverb",
     "คำวิเศษณ์บอกระดับ",         "มาก, น้อย, เกิน, พอ"),
    ("ADJE", "Adjective",             "คำคุณศัพท์",
     "สวย, ใหญ่, เล็ก, แดง"),
    ("AJTV", "Attributive Adjective", "คำคุณศัพท์ขยายนาม",        "ดี, สูง, ต่ำ"),
    ("PRON", "Pronoun",               "คำสรรพนาม",
     "ฉัน, เขา, เรา, มัน, ท่าน"),
    ("RPRE", "Preposition",           "คำบุพบท",
     "ใน, บน, ของ, กับ, โดย"),
    ("CONJ", "Conjunction",           "คำสันธาน",
     "และ, หรือ, แต่, เพราะ"),
    ("PUNC", "Punctuation",           "เครื่องหมายวรรคตอน",        ", . ! ? ฯ ฯลฯ"),
    ("INTJ", "Interjection",          "คำอุทาน",
     "โอ้, อ้าว, เฮ้, โธ่"),
    ("FIXN", "Nominal Prefix",        "คำนำหน้านาม",               "การ, ความ"),
    ("FIXV", "Verbal Prefix",         "คำนำหน้ากริยา",             "นัก, ผู้, ความ"),
    ("DONM", "Nominal Suffix",        "คำต่อท้ายนาม",              "ๆ (ซ้ำนาม)"),
    ("PART", "Particle",              "คำอนุภาค / คำลงท้าย",
     "ครับ, ค่ะ, นะ, สิ, เถอะ"),
    ("DETM", "Determiner",            "คำแสดงความชี้เฉพาะ",
     "นี้, นั้น, โน้น, ดัง"),
    ("CLAS", "Classifier",            "คำลักษณนาม",
     "คน, ตัว, เล่ม, ใบ, คัน"),
    ("NEGA", "Negation",              "คำปฏิเสธ",
     "ไม่, ไม่ได้, มิ, หา...ไม่"),
    ("RANK", "Rank",                  "คำแสดงลำดับ",
     "แรก, สุดท้าย, ต่อไป"),
    ("QUANM", "Quantifier",            "คำบอกปริมาณ",
     "ทุก, บาง, หลาย, ทั้งหมด"),
    ("SUBR", "Subordinator",          "คำเชื่อมอนุประโยค",         "ที่, ซึ่ง, อัน, ว่า"),
    ("CMPM", "Comparison Marker",     "คำเปรียบเทียบ",
     "กว่า, เท่า, เหมือน, คล้าย"),
    ("XVAM", "Aspect Marker",         "คำบอกลักษณะการณ์",
     "แล้ว, อยู่, กำลัง, เคย"),
    ("POSTM", "Postmodifier",          "คำขยายหลังนาม",             "เอง, ทั้งหมด"),
]

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ ตั้งค่า POS Tagger")

    st.info("""
    **ℹ️ PyThaiNLP 5.x**
    รองรับ corpus: **ORCHID**
    Engine: **perceptron**
    """)

    st.divider()
    st.markdown("""
    **Output format:**
    ```
    คำ/TAG|คำ/TAG|คำ/TAG
    ```
    ตัวอย่าง:
    ```
    ฉัน/PRON|กิน/VACT|ข้าว/NCMN
    ```
    """)

# --- POS Reference Table ---
st.header("📚 ตารางอ้างอิงชนิดคำ — ORCHID Tagset")

with st.expander("📖 คลิกเพื่อดูตารางอ้างอิง POS Tags ทั้งหมด", expanded=True):
    df_orchid = pd.DataFrame(
        ORCHID_TAGS,
        columns=["Tag", "ชื่อภาษาอังกฤษ", "ความหมาย", "ตัวอย่าง"]
    )
    df_orchid.index += 1

    def highlight_orchid(row):
        content = ["VACT", "VSTA", "NCMN", "NPRP",
                   "ADJE", "AJTV", "ADVN", "ADVI"]
        func = ["RPRE", "CONJ", "SUBR", "NEGA", "VAUX", "PART", "XVAM"]
        if row["Tag"] in content:
            return ["background-color: #d4edda"] * len(row)
        elif row["Tag"] in func:
            return ["background-color: #fff3cd"] * len(row)
        return [""] * len(row)

    st.dataframe(
        df_orchid.style.apply(highlight_orchid, axis=1),
        use_container_width=True,
        height=400
    )
    st.caption("🟢 สีเขียว = Content Words | 🟡 สีเหลือง = Function Words")

st.divider()

# --- Quick Reference ---
with st.expander("⚡ Quick Reference — Tags ที่พบบ่อยที่สุด"):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        **คำนาม (Nouns)**
        - `NCMN` — นามทั่วไป
        - `NPRP` — นามเฉพาะ
        - `NTTL` — ตำแหน่ง/คำนำหน้าชื่อ
        - `CLAS` — ลักษณนาม
        """)
    with col2:
        st.markdown("""
        **กริยา (Verbs)**
        - `VACT` — กริยาการกระทำ
        - `VSTA` — กริยาสภาวะ (เป็น, อยู่)
        - `VAUX` — กริยาช่วย (จะ, ต้อง)
        """)
    with col3:
        st.markdown("""
        **อื่น ๆ ที่สำคัญ**
        - `ADVN` — วิเศษณ์
        - `PRON` — สรรพนาม
        - `NEGA` — ปฏิเสธ (ไม่)
        - `PART` — คำลงท้าย (ครับ, ค่ะ)
        - `RPRE` — บุพบท (ใน, ของ)
        """)

st.divider()

# --- File Upload & Processing ---
st.header("📂 อัปโหลดและประมวลผล")

st.info("📌 ไฟล์ที่อัปโหลดต้องเป็นไฟล์ที่ **ตัดคำแล้ว** (คั่นด้วย `|`) จากขั้นตอน ✂️ Word Segmenter")

uploaded_files = st.file_uploader(
    "เลือกไฟล์ที่ตัดคำแล้ว (.txt, UTF-8, คั่นด้วย |)",
    type="txt",
    accept_multiple_files=True
)

if uploaded_files:
    st.info(f"📂 เลือกไฟล์ทั้งหมด {len(uploaded_files)} ไฟล์")

    # Preview ไฟล์แรก
    with st.expander("👁️ ดูตัวอย่างไฟล์แรก"):
        preview_content = uploaded_files[0].read().decode("utf-8")[:500]
        uploaded_files[0].seek(0)
        st.code(preview_content, language=None)

    st.warning("⚠️ การ POS Tagging อาจใช้เวลานาน ขึ้นอยู่กับขนาดไฟล์")

    if st.button("▶️ เริ่มกำกับ POS", type="primary"):

        zip_buffer = io.BytesIO()
        preview_text = ""  # เก็บไว้แสดง preview ไฟล์แรก

        with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
            progress_bar = st.progress(0)
            status = st.empty()

            for i, uploaded_file in enumerate(uploaded_files):
                try:
                    status.text(f"กำลังประมวลผล: {uploaded_file.name} ...")
                    content = uploaded_file.read().decode("utf-8")

                    # แยกคำด้วย |
                    lines = content.split("\n")
                    output_lines = []

                    for line in lines:
                        tokens = [t.strip()
                                  for t in line.split("|") if t.strip()]
                        if not tokens:
                            output_lines.append("")
                            continue

                        # POS Tag — perceptron + orchid
                        tagged = pos_tag(
                            tokens,
                            engine="perceptron",
                            corpus="orchid"
                        )

                        # รูปแบบ คำ/TAG|คำ/TAG
                        tagged_str = "|".join(
                            [f"{word}/{tag}" for word, tag in tagged]
                        )
                        output_lines.append(tagged_str)

                    output_text = "\n".join(output_lines)

                    # เก็บไฟล์แรกไว้ preview
                    if i == 0:
                        preview_text = output_text

                    # บันทึกลง ZIP
                    name_part, extension = os.path.splitext(uploaded_file.name)
                    new_filename = f"{name_part}_orchid{extension}"
                    zip_file.writestr(new_filename, output_text)

                    progress_bar.progress((i + 1) / len(uploaded_files))

                except Exception as e:
                    st.error(
                        f"❌ เกิดข้อผิดพลาดกับไฟล์ {uploaded_file.name}: {e}"
                    )

        status.empty()
        st.success(
            f"✅ กำกับ POS เสร็จสิ้น! {len(uploaded_files)} ไฟล์ | Tagset: ORCHID"
        )

        # Preview ผลลัพธ์
        with st.expander("👁️ ดูตัวอย่างผลลัพธ์ (ไฟล์แรก)"):
            st.caption("รูปแบบ: คำ/TAG|คำ/TAG|...")
            st.code(preview_text[:500], language=None)

        st.download_button(
            label="📥 ดาวน์โหลดผลลัพธ์ทั้งหมด (ZIP)",
            data=zip_buffer.getvalue(),
            file_name="pos_tagged_orchid.zip",
            mime="application/zip"
        )

        st.info(
            "💡 ขั้นตอนถัดไป: นำไฟล์ที่ได้ไปใช้ใน **🐜 Concordance** "
            "เพื่อค้นหาด้วย POS ได้ เช่น `กิน/VACT` หรือ `/NCMN`"
        )

    st.divider()
    st.markdown("""
    > **📝 หมายเหตุ:** POS Tagging เป็นการประมาณการอัตโนมัติ
    > ความแม่นยำขึ้นอยู่กับประเภทข้อความ
    > แนะนำให้ตรวจสอบและแก้ไขผลลัพธ์ก่อนนำไปใช้ในงานวิจัยอย่างเป็นทางการ
    """)

else:
    st.info("👆 กรุณาเลือกไฟล์ที่ตัดคำแล้วเพื่อเริ่มต้น")
