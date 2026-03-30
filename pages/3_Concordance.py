import streamlit as st
import pandas as pd
from collections import Counter
import re
import io
import math
from pythainlp import word_tokenize
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

st.set_page_config(page_title="Concordance", page_icon="🐜", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #F8F9FA; color: #212529; }
    [data-testid="stSidebar"] { background-color: #E9ECEF; }
    th { background-color: #DEE2E6 !important; color: #212529 !important; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# STOP WORDS
# ============================================================

ENGLISH_STOP_WORDS = set([
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours",
    "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers",
    "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves",
    "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are",
    "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does",
    "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until",
    "while", "of", "at", "by", "for", "with", "about", "against", "between", "into",
    "through", "during", "before", "after", "above", "below", "to", "from", "up", "down",
    "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here",
    "there", "when", "where", "why", "how", "all", "both", "each", "few", "more", "most",
    "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than",
    "too", "very", "s", "t", "can", "will", "just", "don", "should", "now",
])

# ============================================================
# FILE TYPE DETECTION
# ============================================================


def detect_file_type(content: str) -> str:
    """ตรวจสอบประเภทไฟล์อัตโนมัติ"""
    sample = content[:500]
    # มี POS tag รูปแบบ คำ/TAG
    if re.search(r'\w+/[A-Z]{2,5}', sample):
        return "pos_tagged"
    # มี | คั่น แต่ไม่มี POS
    elif "|" in sample:
        return "segmented"
    # ข้อความดิบ
    else:
        return "raw"

# ============================================================
# PROCESS CORPUS
# ============================================================


@st.cache_data(show_spinner=False)
def process_corpus(uploaded_files, auto_tokenize=True):
    files_data = []
    all_tokens_flat = []

    for uploaded_file in uploaded_files:
        content = uploaded_file.read().decode("utf-8")
        file_type = detect_file_type(content)
        has_pos = False

        if file_type == "pos_tagged":
            # แยกคำและ POS
            has_pos = True
            raw_pairs = [t.strip()
                         for t in re.split(r'[|\n]', content) if t.strip()]
            tokens = []
            pos_tokens = []  # list of (word, tag)
            for pair in raw_pairs:
                if "/" in pair:
                    parts = pair.rsplit("/", 1)
                    word, tag = parts[0].strip(), parts[1].strip()
                    tokens.append(word)
                    pos_tokens.append((word, tag))
                else:
                    tokens.append(pair)
                    pos_tokens.append((pair, "XX"))

        elif file_type == "segmented":
            raw_tokens = re.split(r'[|\n]', content)
            tokens = [t.strip()
                      for t in raw_tokens if t.strip() and t.strip() != "|"]
            pos_tokens = []

        else:  # raw
            if auto_tokenize:
                raw_tokens = word_tokenize(
                    content, engine='newmm', keep_whitespace=False)
            else:
                raw_tokens = content.replace("\n", "|").split("|")
            tokens = [t.strip()
                      for t in raw_tokens if t.strip() and t.strip() != "|"]
            pos_tokens = []

        display_text = "|".join(tokens)
        files_data.append({
            "filename": uploaded_file.name,
            "tokens": tokens,
            "pos_tokens": pos_tokens,
            "has_pos": has_pos,
            "file_type": file_type,
            "text": display_text
        })
        all_tokens_flat.extend(tokens)

    return files_data, all_tokens_flat

# ============================================================
# SEARCH FUNCTIONS
# ============================================================


def check_token_match(token, tag, pattern):
    """Match คำ + รองรับ /TAG pattern"""
    token_lower = token.lower()

    # แยก pattern กับ TAG ถ้ามี
    pos_filter = None
    word_pattern = pattern

    if "/" in pattern:
        parts = pattern.rsplit("/", 1)
        word_pattern = parts[0].strip()
        pos_filter = parts[1].strip().upper()

    # ตรวจ TAG ก่อน
    if pos_filter:
        if tag is None or tag.upper() != pos_filter:
            return False
        # ถ้า word_pattern ว่าง = ค้นหาทุกคำใน POS นั้น
        if word_pattern == "":
            return True

    word_pattern_lower = word_pattern.lower()

    if word_pattern_lower.startswith("*") and word_pattern_lower.endswith("*") and len(word_pattern_lower) > 2:
        clean = word_pattern_lower[1:-1]
        return clean in token_lower[1:-1]
    elif word_pattern_lower.startswith("*") and len(word_pattern_lower) > 1:
        clean = word_pattern_lower[1:]
        return token_lower.endswith(clean) and len(token_lower) > len(clean)
    elif word_pattern_lower.endswith("*") and len(word_pattern_lower) > 1:
        clean = word_pattern_lower[:-1]
        return token_lower.startswith(clean) and len(token_lower) > len(clean)
    else:
        return token_lower == word_pattern_lower


def parse_search_query(query):
    query = query.strip()
    gap_pattern = re.search(r'^(\S+)\s+<(\d+)(?:-(\d+))?>\s+(\S+)$', query)
    sequence_pattern = re.search(r'^(\S+)\s+(\S+)$', query)

    if gap_pattern:
        return "gap", (gap_pattern.group(1), int(gap_pattern.group(2)),
                       int(gap_pattern.group(3)) if gap_pattern.group(
                           3) else int(gap_pattern.group(2)),
                       gap_pattern.group(4))
    elif sequence_pattern:
        return "gap", (sequence_pattern.group(1), 0, 0, sequence_pattern.group(2))
    else:
        return "single", query


def generate_kwic(files_data, keyword, window_size=7, show_pos=False):
    results = []
    search_type, search_params = parse_search_query(keyword)

    for file_info in files_data:
        filename = file_info['filename']
        tokens = file_info['tokens']
        pos_tokens = file_info.get('pos_tokens', [])
        has_pos = file_info.get('has_pos', False)
        len_tokens = len(tokens)

        def get_tag(idx):
            if has_pos and pos_tokens and idx < len(pos_tokens):
                return pos_tokens[idx][1]
            return None

        def format_token(idx):
            w = tokens[idx]
            if show_pos and has_pos and pos_tokens and idx < len(pos_tokens):
                return f"{w}/{pos_tokens[idx][1]}"
            return w

        i = 0
        while i < len_tokens:
            match_found = False
            match_start = i
            match_end = i

            if search_type == "single":
                if check_token_match(tokens[i], get_tag(i), search_params):
                    match_found = True
            elif search_type == "gap":
                start_pat, min_gap, max_gap, end_pat = search_params
                if check_token_match(tokens[i], get_tag(i), start_pat):
                    for j in range(i + 1 + min_gap, min(i + 1 + max_gap + 1, len_tokens)):
                        if check_token_match(tokens[j], get_tag(j), end_pat):
                            match_found = True
                            match_end = j
                            break

            if match_found:
                left = [format_token(x) for x in range(
                    max(0, match_start - window_size), match_start)]
                node = [format_token(x)
                        for x in range(match_start, match_end + 1)]
                right = [format_token(x) for x in range(
                    match_end + 1, min(len_tokens, match_end + window_size + 1))]
                results.append({
                    "Left": " ".join(left),
                    "Node": " ".join(node),
                    "Right": " ".join(right),
                    "File": filename
                })
            i += 1
    return pd.DataFrame(results)


def get_content_words(tokens, extra_stopwords=None):
    stop = ENGLISH_STOP_WORDS.copy()
    if extra_stopwords:
        stop |= set(extra_stopwords)
    return [t for t in tokens if t not in stop and not re.fullmatch(r'[\d\s\W]+', t) and len(t) >= 2]


def generate_ngrams(tokens, n=2, min_freq=1):
    if len(tokens) < n:
        return pd.DataFrame()
    ngrams = zip(*[tokens[i:] for i in range(n)])
    counts = Counter([" ".join(g) for g in ngrams])
    df = pd.DataFrame(counts.items(), columns=['Cluster', 'Frequency'])
    return df[df['Frequency'] >= min_freq].sort_values('Frequency', ascending=False).reset_index(drop=True)


def calculate_mi_score(word1, tokens, min_freq=2, files_data=None):
    N = len(tokens)
    bigrams = [(tokens[i], tokens[i+1]) for i in range(len(tokens)-1)]
    bigram_counts = Counter(bigrams)
    word_counts = Counter(tokens)
    f_word1 = word_counts.get(word1, 0)
    if f_word1 == 0:
        return pd.DataFrame()

    bigram_files = {}
    if files_data:
        for fi in files_data:
            fname = fi['filename']
            ft = fi['tokens']
            for i in range(len(ft)-1):
                bg = (ft[i], ft[i+1])
                bigram_files.setdefault(bg, set()).add(fname)

    results = []
    for (w1, w2), f_pair in bigram_counts.items():
        if w1 == word1 and f_pair >= min_freq:
            f_word2 = word_counts.get(w2, 0)
            if f_word2 > 0:
                mi = math.log2((f_pair * N) / (f_word1 * f_word2))
                found_in = ", ".join(sorted(bigram_files.get(
                    (w1, w2), set()))) if files_data else "-"
                results.append({"Collocate": w2, "Frequency": f_pair,
                               "MI Score": round(mi, 3), "พบในไฟล์": found_in})

    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values("MI Score", ascending=False).reset_index(drop=True)
        df.index += 1
    return df


def generate_wordcloud(freq_dict, colormap="viridis", bg_color="white", max_words=150, font_path="Sarabun-Regular.ttf"):
    if not freq_dict:
        return None
    wc = WordCloud(font_path=font_path, width=900, height=500,
                   background_color=bg_color, colormap=colormap,
                   max_words=max_words, prefer_horizontal=0.85,
                   min_font_size=10, max_font_size=120, collocations=False)
    wc.generate_from_frequencies(freq_dict)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis("off")
    plt.tight_layout(pad=0)
    return fig


def generate_tag_cloud_html(word_freq_dict, max_words=80):
    if not word_freq_dict:
        return ""
    items = sorted(word_freq_dict.items(),
                   key=lambda x: x[1], reverse=True)[:max_words]
    max_freq = items[0][1]
    min_freq = items[-1][1]
    colors = ["#0d6efd", "#6610f2", "#6f42c1", "#d63384", "#dc3545",
              "#fd7e14", "#ffc107", "#198754", "#20c997", "#0dcaf0"]
    html = '<div style="background:white;border-radius:12px;padding:25px;border:1px solid #dee2e6;line-height:2.8;text-align:center;">'
    for i, (word, freq) in enumerate(items):
        size = int(14 + (freq - min_freq) / (max_freq - min_freq + 1) * 38)
        color = colors[i % len(colors)]
        html += f'<span style="font-size:{size}px;color:{color};margin:4px 8px;display:inline-block;font-weight:{"bold" if size > 30 else "normal"};" title="ความถี่: {freq}">{word}</span>'
    html += "</div>"
    return html


def calculate_corpus_stats(files_data, all_tokens):
    total_tokens = len(all_tokens)
    total_types = len(set(all_tokens))
    ttr = round(total_types / total_tokens * 100, 2) if total_tokens > 0 else 0
    file_stats = []
    for f in files_data:
        t = f['tokens']
        file_stats.append({
            "ไฟล์": f['filename'],
            "ประเภท": f.get('file_type', '-'),
            "มี POS": "✅" if f.get('has_pos') else "❌",
            "Tokens": len(t),
            "Types": len(set(t)),
            "TTR (%)": round(len(set(t)) / len(t) * 100, 2) if len(t) > 0 else 0,
        })
    return {"total_tokens": total_tokens, "total_types": total_types, "ttr": ttr,
            "file_stats": pd.DataFrame(file_stats)}


def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')


# ============================================================
# UI
# ============================================================
st.title("🐜 Concordance & Corpus Analysis")
st.caption("วิเคราะห์คลังข้อมูลภาษา — รองรับไฟล์ดิบ / ตัดคำแล้ว / มี POS Tag")

# --- Sidebar ---
with st.sidebar:
    st.header("📂 Upload & Settings")

    use_auto_tokenize = st.checkbox(
        "Auto Tokenize (สำหรับไฟล์ดิบ)",
        value=True,
        help="เลือกถ้าไฟล์ยังไม่ได้ตัดคำ"
    )

    uploaded_files = st.file_uploader(
        "เลือกไฟล์ (.txt, UTF-8)",
        type=['txt'],
        accept_multiple_files=True
    )

    st.divider()
    st.subheader("🛑 Stop Words เพิ่มเติม")
    custom_sw_input = st.text_area(
        "ใส่คำที่ต้องการกรองออก (หนึ่งคำต่อบรรทัด):", height=80)
    custom_stopwords = [w.strip()
                        for w in custom_sw_input.splitlines() if w.strip()]

    st.divider()
    st.subheader("🔤 Font ภาษาไทย")
    font_path = st.text_input("Path ไฟล์ font (.ttf):", value="Sarabun-Regular.ttf",
                              help="สำหรับ Word Cloud และกราฟ")

if uploaded_files:
    with st.spinner('กำลังอ่านและประมวลผลไฟล์...'):
        files_data, all_tokens_flat = process_corpus(
            uploaded_files, use_auto_tokenize)

    # ตรวจสอบว่ามีไฟล์ที่มี POS ไหม
    any_has_pos = any(f['has_pos'] for f in files_data)
    all_has_pos = all(f['has_pos'] for f in files_data)

    content_words = get_content_words(all_tokens_flat, custom_stopwords)
    content_word_freq = Counter(content_words)
    stats = calculate_corpus_stats(files_data, all_tokens_flat)

    # Status bar
    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    col_s1.metric("📁 ไฟล์", len(files_data))
    col_s2.metric("📝 Tokens", f"{len(all_tokens_flat):,}")
    col_s3.metric("🔤 Types", f"{len(set(all_tokens_flat)):,}")
    if any_has_pos:
        col_s4.metric("🏷️ POS", "✅ มีข้อมูล")
    else:
        col_s4.metric("🏷️ POS", "❌ ไม่มีข้อมูล")

    # File type summary
    type_summary = [f"**{f['filename']}** → `{f['file_type']}`" +
                    (" 🏷️" if f['has_pos'] else "") for f in files_data]
    with st.expander("📋 สรุปประเภทไฟล์ที่อัปโหลด"):
        for s in type_summary:
            st.markdown(s)

    st.divider()

    # --- Tabs ---
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "🔍 Concordance", "📊 Word List", "🔗 N-Grams",
        "☁️ Word Cloud", "🏷️ Tag Cloud", "📐 Collocations (MI)",
        "📈 Corpus Stats", "📄 File Content"
    ])

    # --- Tab 1: KWIC ---
    with tab1:
        st.subheader("Concordance (KWIC)")

        with st.expander("ℹ️ วิธีการใช้คำค้นหา"):
            st.markdown("""
            **รูปแบบทั่วไป:**
            | รูปแบบ | ความหมาย |
            |--------|----------|
            | `รัก` | Exact match |
            | `*รัก` | ลงท้ายด้วย "รัก" |
            | `รัก*` | ขึ้นต้นด้วย "รัก" |
            | `*รัก*` | มี "รัก" อยู่ในคำ |
            | `รัก มาก` | "รัก" ตามด้วย "มาก" |
            | `รัก <3> มาก` | ห่างกัน 3 คำ |
            | `รัก <0-3> มาก` | ห่างกัน 0–3 คำ |

            **รูปแบบที่ใช้ POS** (เฉพาะไฟล์ที่มี POS tag):
            | รูปแบบ | ความหมาย |
            |--------|----------|
            | `กิน/VACT` | คำว่า "กิน" ที่เป็นกริยาแสดงการกระทำ |
            | `/NCMN` | ทุกคำที่เป็นคำนามทั่วไป |
            | `จะ*/VAUX` | คำขึ้นต้น "จะ" ที่เป็นกริยาช่วย |
            | `กิน/VACT /NCMN` | "กิน" (กริยาแสดงการกระทำ) ตามด้วยคำนามทั่วไปใดก็ได้ |
            """)

        c1, c2, c3 = st.columns([3, 1, 1])
        search_term = c1.text_input("คำค้นหา:", "")
        window = c2.slider("Context Span:", 3, 20, 8)
        show_pos_kwic = c3.checkbox("แสดง POS", value=False, disabled=not any_has_pos,
                                    help="ใช้ได้เฉพาะไฟล์ที่มี POS tag", key="kwic_show_pos")

        if search_term:
            df = generate_kwic(files_data, search_term,
                               window, show_pos=show_pos_kwic)
            if not df.empty:
                df.index += 1
                st.write(f"พบ: **{len(df):,}** รายการ")
                st.download_button("📥 CSV", convert_df_to_csv(
                    df), f"kwic_{search_term}.csv", "text/csv")

                def render_kwic_table(df):
                    rows = ""
                    for i, (_, row) in enumerate(df.iterrows(), start=1):
                        rows += f"""
                        <tr>
                            <td style="text-align:center; color:#6c757d; font-size:0.85em;
                                    padding:6px 4px; border-bottom:1px solid #dee2e6;
                                    white-space:nowrap; width:4%;">
                                {i}
                            </td>
                            <td style="text-align:right; color:#495057; padding:6px 8px;
                                    border-bottom:1px solid #dee2e6; white-space:nowrap; width:36%;">
                                {row['Left']}
                            </td>
                            <td style="text-align:center; font-weight:bold; color:#0d6efd;
                                    padding:6px 12px; border-bottom:1px solid #dee2e6;
                                    white-space:nowrap; width:10%;">
                                {row['Node']}
                            </td>
                            <td style="text-align:left; color:#495057; padding:6px 8px;
                                    border-bottom:1px solid #dee2e6; white-space:nowrap; width:36%;">
                                {row['Right']}
                            </td>
                            <td style="text-align:left; color:#6c757d; font-size:0.8em;
                                    padding:6px 8px; border-bottom:1px solid #dee2e6;
                                    white-space:nowrap; width:14%;">
                                {row['File']}
                            </td>
                        </tr>
                        """

                    html = f"""
                    <div style="overflow-x:auto; overflow-y:auto; max-height:500px;">
                    <table style="width:100%; border-collapse:collapse; font-size:0.9em;
                                font-family:'Sarabun', sans-serif; table-layout:fixed;">
                        <thead>
                            <tr style="background:#DEE2E6; position:sticky; top:0; z-index:1;">
                                <th style="text-align:center; padding:8px; width:4%;">No.</th>
                                <th style="text-align:right; padding:8px; width:36%;">Left</th>
                                <th style="text-align:center; padding:8px; width:10%;">Node</th>
                                <th style="text-align:left; padding:8px; width:36%;">Right</th>
                                <th style="text-align:left; padding:8px; width:14%;">File</th>
                            </tr>
                        </thead>
                        <tbody>
                            {rows}
                        </tbody>
                    </table>
                    </div>
                    """
                    st.components.v1.html(html, height=520, scrolling=True)

                render_kwic_table(df)
            else:
                st.warning("ไม่พบคำที่ค้นหา")

    # --- Tab 2: Word List ---
    with tab2:
        st.subheader("Word List")
        col_a, col_b, col_c = st.columns(3)
        show_all = col_a.checkbox(
            "รวม Stop Words", value=True, key="wl_show_all")
        show_pos_wl = col_b.checkbox(
            "แสดง POS", value=False, disabled=not any_has_pos, key="wl_show_pos")
        min_freq_wl = col_c.number_input("ความถี่ขั้นต่ำ:", 1, 1000, 1)

        wc_base = Counter(all_tokens_flat) if show_all else content_word_freq
        df_wl = pd.DataFrame(wc_base.items(), columns=['Word', 'Frequency'])
        df_wl = df_wl[df_wl['Frequency'] >= min_freq_wl]
        df_wl = df_wl.sort_values(
            'Frequency', ascending=False).reset_index(drop=True)
        df_wl.index += 1
        total = df_wl['Frequency'].sum()
        df_wl['%'] = (df_wl['Frequency'] / total * 100).round(2)
        df_wl['Cumulative %'] = df_wl['%'].cumsum().round(2)

        # เพิ่มคอลัมน์ POS ถ้ามี
        if show_pos_wl and any_has_pos:
            pos_lookup = {}
            for f in files_data:
                for word, tag in f.get('pos_tokens', []):
                    if word not in pos_lookup:
                        pos_lookup[word] = []
                    if tag not in pos_lookup[word]:
                        pos_lookup[word].append(tag)
            df_wl['POS Tags'] = df_wl['Word'].map(
                lambda w: ", ".join(pos_lookup.get(w, ["-"])))

        st.write(f"พบ: **{len(df_wl):,}** รายการ")
        st.download_button("📥 CSV", convert_df_to_csv(
            df_wl), "wordlist.csv", "text/csv")
        st.dataframe(df_wl, use_container_width=True)

    # --- Tab 3: N-Grams ---
    with tab3:
        st.subheader("N-Grams / Clusters")
        c1, c2, c3 = st.columns(3)
        n_size = c1.number_input("N-gram size", 2, 5, 2)
        min_f = c2.number_input("Min Frequency", 1, 100, 2)
        use_content_only = c3.checkbox(
            "เฉพาะ Content Words", False, key="ng_content")

        if st.button("🔍 Start N-Grams"):
            source_tokens = content_words if use_content_only else all_tokens_flat
            df_ng = generate_ngrams(source_tokens, n_size, min_f)
            if not df_ng.empty:
                df_ng.index += 1
                st.write(f"พบ: **{len(df_ng):,}** รายการ")
                st.download_button("📥 CSV", convert_df_to_csv(
                    df_ng), f"{n_size}grams.csv", "text/csv")
                st.dataframe(df_ng, use_container_width=True)
            else:
                st.warning("ไม่พบข้อมูล")

    # --- Tab 4: Word Cloud ---
    with tab4:
        st.subheader("☁️ Word Cloud")
        col1, col2, col3, col4 = st.columns(4)
        wc_max = col1.slider("จำนวนคำสูงสุด:", 20, 300, 100)
        wc_cmap = col2.selectbox("Color Scheme:", [
                                 "viridis", "plasma", "inferno", "magma", "cool", "hot", "RdYlGn", "Spectral"])
        wc_bg = col3.selectbox("Background:", ["white", "black", "#1a1a2e"])
        wc_content = col4.checkbox(
            "เฉพาะ Content Words", value=True, key="wc_content")

        if st.button("☁️ Generate Word Cloud"):
            src_freq = content_word_freq if wc_content else Counter(
                all_tokens_flat)
            if src_freq:
                try:
                    fig = generate_wordcloud(
                        dict(src_freq), wc_cmap, wc_bg, wc_max, font_path)
                    if fig:
                        st.pyplot(fig)
                        buf = io.BytesIO()
                        fig.savefig(buf, format='png', dpi=150,
                                    bbox_inches='tight')
                        buf.seek(0)
                        st.download_button(
                            "📥 Download PNG", buf, "wordcloud.png", "image/png")
                    plt.close()
                except Exception as e:
                    st.error(f"เกิดข้อผิดพลาด: {e}")
                    st.info("💡 ตรวจสอบ path ของ font ใน Sidebar")

    # --- Tab 5: Tag Cloud ---
    with tab5:
        st.subheader("🏷️ Tag Cloud (Interactive)")
        col1, col2 = st.columns(2)
        tc_max = col1.slider("จำนวนคำสูงสุด:", 20, 150, 60)
        tc_content = col2.checkbox(
            "เฉพาะ Content Words", value=True, key="tc_content")
        tc_min_freq = st.slider("ความถี่ขั้นต่ำ:", 1, 50, 2)

        if st.button("🏷️ Generate Tag Cloud"):
            src_freq = content_word_freq if tc_content else Counter(
                all_tokens_flat)
            filtered = {w: f for w, f in src_freq.items() if f >= tc_min_freq}
            if filtered:
                html_cloud = generate_tag_cloud_html(filtered, tc_max)
                st.components.v1.html(html_cloud, height=350, scrolling=True)
                with st.expander("📋 ดูตาราง"):
                    df_tc = pd.DataFrame(filtered.items(), columns=[
                                         'Word', 'Frequency'])
                    df_tc = df_tc.sort_values('Frequency', ascending=False).head(
                        tc_max).reset_index(drop=True)
                    df_tc.index += 1
                    st.dataframe(df_tc, use_container_width=True)
            else:
                st.warning("ไม่มีข้อมูล")

    # --- Tab 6: Collocations ---
    with tab6:
        st.subheader("📐 Collocations (Mutual Information Score)")
        st.markdown(
            "**MI > 3** = Collocation ชัดเจน | **MI 0–3** = ปานกลาง | **MI < 0** = ไม่สัมพันธ์")
        c1, c2 = st.columns(2)
        node_word = c1.text_input(
            "คำที่ต้องการหา Collocate:", placeholder="เช่น รัก, เรียน")
        mi_min_freq = c2.number_input("ความถี่ขั้นต่ำ:", 1, 100, 2)

        if st.button("🔍 คำนวณ MI Score") and node_word:
            df_mi = calculate_mi_score(
                node_word, all_tokens_flat, mi_min_freq, files_data)
            if not df_mi.empty:
                st.write(f"พบ collocates: **{len(df_mi)}** คำ")
                st.download_button("📥 CSV", convert_df_to_csv(
                    df_mi), f"collocations_{node_word}.csv", "text/csv")

                def color_mi(val):
                    if val >= 3:
                        return 'background-color: #d4edda; color: #155724'
                    elif val >= 0:
                        return 'background-color: #fff3cd; color: #856404'
                    else:
                        return 'background-color: #f8d7da; color: #721c24'

                st.dataframe(df_mi.style.applymap(color_mi, subset=[
                             'MI Score']), use_container_width=True)
            else:
                st.warning(f"ไม่พบ collocation สำหรับ '{node_word}'")

    # --- Tab 7: Corpus Stats ---
    with tab7:
        st.subheader("📈 Corpus Statistics")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("📝 Total Tokens", f"{stats['total_tokens']:,}")
        col2.metric("🔤 Total Types", f"{stats['total_types']:,}")
        col3.metric("📊 TTR (%)", f"{stats['ttr']}%")
        col4.metric("📚 Content Words", f"{len(content_words):,}")

        st.divider()
        st.subheader("สถิติรายไฟล์")
        st.dataframe(stats['file_stats'],
                     use_container_width=True, hide_index=True)
        st.download_button("📥 Export CSV", convert_df_to_csv(
            stats['file_stats']), "corpus_stats.csv", "text/csv")

        st.divider()
        st.subheader("🏆 Top 20 Content Words")
        top20 = content_word_freq.most_common(20)
        if top20:
            df_top20 = pd.DataFrame(top20, columns=["คำ", "ความถี่"])
            df_top20.index += 1
            col_a, col_b = st.columns([1, 2])
            with col_a:
                st.dataframe(df_top20, use_container_width=True)
            with col_b:
                try:
                    fp = font_manager.FontProperties(fname=font_path)
                    fig2, ax2 = plt.subplots(figsize=(8, 5))
                    words_p = [w for w, _ in top20]
                    freqs_p = [f for _, f in top20]
                    ax2.barh(words_p[::-1], freqs_p[::-1],
                             color=plt.cm.viridis([i/20 for i in range(20)]))
                    ax2.set_xlabel("Frequency", fontproperties=fp)
                    ax2.set_title("Top 20 Content Words", fontproperties=fp)
                    for label in ax2.get_yticklabels():
                        label.set_fontproperties(fp)
                    ax2.tick_params(axis='y', labelsize=9)
                    plt.tight_layout()
                    st.pyplot(fig2)
                    plt.close()
                except Exception as e:
                    st.error(f"ไม่สามารถโหลด font ได้: {e}")

        # POS Distribution (ถ้ามี POS)
        if any_has_pos:
            st.divider()
            st.subheader("🏷️ POS Distribution")
            all_pos = []
            for f in files_data:
                all_pos.extend([tag for _, tag in f.get('pos_tokens', [])])
            if all_pos:
                pos_counts = Counter(all_pos)
                df_pos = pd.DataFrame(pos_counts.items(), columns=[
                                      'POS Tag', 'Count'])
                df_pos = df_pos.sort_values(
                    'Count', ascending=False).reset_index(drop=True)
                df_pos['%'] = (df_pos['Count'] / len(all_pos) * 100).round(2)
                df_pos.index += 1
                st.dataframe(df_pos, use_container_width=True)
                st.download_button("📥 Export POS Stats", convert_df_to_csv(
                    df_pos), "pos_stats.csv", "text/csv")

    # --- Tab 8: File Content ---
    with tab8:
        st.subheader("📄 เนื้อหาที่ผ่านการประมวลผลแล้ว")
        sel_f = st.selectbox("เลือกไฟล์:", [f['filename'] for f in files_data])
        selected = next(
            (f for f in files_data if f['filename'] == sel_f), None)
        if selected:
            st.caption(
                f"ประเภทไฟล์: `{selected['file_type']}` | มี POS: {'✅' if selected['has_pos'] else '❌'} | Tokens: {len(selected['tokens']):,}")
            st.text_area("Content:", selected['text'], height=400)
            st.caption("* เครื่องหมาย | แสดงจุดที่ตัดคำ")

else:
    st.info("👈 กรุณาเลือกไฟล์ทางด้านซ้ายเพื่อเริ่มใช้งาน")
    st.markdown("""
    ### รองรับไฟล์ 3 ประเภท:
    | ประเภท | รูปแบบ | ฟังก์ชันที่ใช้ได้ |
    |--------|--------|-----------------|
    | ไฟล์ดิบ | ข้อความธรรมดา | ทุกฟังก์ชัน (ตัดคำอัตโนมัติ) |
    | ตัดคำแล้ว | คำ\|คำ\|คำ | ทุกฟังก์ชัน |
    | มี POS | คำ/TAG\|คำ/TAG | ทุกฟังก์ชัน + POS Search + POS Distribution |
    """)
