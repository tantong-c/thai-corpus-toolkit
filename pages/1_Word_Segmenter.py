import streamlit as st
import os
from pythainlp import word_tokenize
import io
import zipfile

st.set_page_config(page_title="Word Segmenter", page_icon="✂️", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #F8F9FA; color: #212529; }
    [data-testid="stSidebar"] { background-color: #E9ECEF; }
    th { background-color: #DEE2E6 !important; color: #212529 !important; }
</style>
""", unsafe_allow_html=True)

st.title("✂️ Word Segmenter")
st.caption("ตัดคำภาษาไทยแบบ Batch — รองรับหลายไฟล์พร้อมกัน")

# --- Sidebar Settings ---
with st.sidebar:
    st.header("⚙️ ตั้งค่าการตัดคำ")

    engine = st.selectbox(
        "Engine",
        ["newmm", "longest", "multi_cut"],
        help="newmm: แนะนำสำหรับใช้งานทั่วไป / longest: ตัดคำยาวที่สุด / multi_cut: หลายแนวทาง"
    )

    separator = st.text_input("ตัวคั่นคำ (Separator)", value="|")

    st.divider()
    st.markdown("""
    **Engine แนะนำ:**
    - `newmm` — ทั่วไป ✅
    - `longest` — เน้นคำยาว
    - `multi_cut` — ละเอียดที่สุด
    """)

# --- Engine Info ---
with st.expander("ℹ️ ข้อมูล Engine และรูปแบบ Output"):
    st.markdown("""
    **รูปแบบไฟล์ Output:**
    - ชื่อไฟล์จะมี `_seg` ต่อท้าย เช่น `text01_seg.txt`
    - คำแต่ละคำคั่นด้วยสัญลักษณ์ที่เลือก (default: `|`)
    - encoding: UTF-8

    **ตัวอย่าง:**
    ```
    Input:  ฉันกินข้าวกับเพื่อนที่โรงเรียน
    Output: ฉัน|กิน|ข้าว|กับ|เพื่อน|ที่|โรงเรียน
    ```

    **ขั้นตอนถัดไป:** นำไฟล์ที่ได้ไปใช้ใน 🏷️ POS Tagger หรือ 🐜 Concordance ได้เลย
    """)

# --- File Uploader ---
uploaded_files = st.file_uploader(
    "เลือกไฟล์ข้อความ (.txt, UTF-8)",
    type="txt",
    accept_multiple_files=True
)

if uploaded_files:
    st.info(f"📂 เลือกไฟล์ทั้งหมด {len(uploaded_files)} ไฟล์")

    # Preview
    with st.expander("👁️ ดูตัวอย่างไฟล์แรก (50 ตัวอักษรแรก)"):
        preview = uploaded_files[0].read().decode("utf-8-sig")[:300]
        uploaded_files[0].seek(0)
        st.text(preview + "...")

    if st.button("▶️ เริ่มตัดคำทั้งหมด", type="primary"):
        zip_buffer = io.BytesIO()

        with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
            progress_bar = st.progress(0)
            status = st.empty()

            for i, uploaded_file in enumerate(uploaded_files):
                try:
                    status.text(f"กำลังประมวลผล: {uploaded_file.name} ...")
                    content = uploaded_file.read().decode("utf-8-sig")

                    tokens = word_tokenize(content, engine=engine, keep_whitespace=False)

                    # กรองช่องว่างและ | ออก
                    tokens = [t.strip() for t in tokens if t.strip() and t.strip() != "|"]
                    segmented_text = separator.join(tokens)

                    name_part, extension = os.path.splitext(uploaded_file.name)
                    new_filename = f"{name_part}_seg{extension}"
                    zip_file.writestr(new_filename, segmented_text)

                    progress_bar.progress((i + 1) / len(uploaded_files))

                except Exception as e:
                    st.error(f"❌ เกิดข้อผิดพลาดกับไฟล์ {uploaded_file.name}: {e}")

        status.empty()
        st.success(f"✅ ตัดคำเสร็จสิ้น! {len(uploaded_files)} ไฟล์")

        st.download_button(
            label="📥 ดาวน์โหลดผลลัพธ์ทั้งหมด (ZIP)",
            data=zip_buffer.getvalue(),
            file_name="segmented_files.zip",
            mime="application/zip"
        )

        st.info("💡 ขั้นตอนถัดไป: นำไฟล์ที่ได้ไปใช้ใน **🏷️ POS Tagger** หรือ **🐜 Concordance**")

else:
    st.info("👆 กรุณาเลือกไฟล์ .txt เพื่อเริ่มต้น")
