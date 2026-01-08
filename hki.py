import streamlit as st
import pandas as pd
import re
import emoji
import torch
import joblib
import math
import html as ihtml
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import hf_hub_download
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import streamlit.components.v1 as components

# =========================
# CONFIG
# =========================
MODEL_REPO = "mesakhbesta/IndoBERT_Aspek"
DATASET_REPO = "mesakhbesta/IndoBERT_Aspek"
BATCH_SIZE = 64

st.set_page_config(page_title="AspectLens", layout="wide")

# =========================
# HEADER UI
# =========================
st.markdown(
    """
    <div style="padding:30px; border-radius:20px; background: linear-gradient(135deg, #1f4037, #99f2c8); color:white; margin-bottom:20px;">
        <h1 style="margin-bottom:5px;">🔍 AspectLens</h1>
        <h4 style="margin-top:0;">Analisis Aspek Komentar Otomatis dengan IndoBERT</h4>
        <p style="font-size:16px;">Unggah data komentar (Instagram, TikTok, dll), pilih kolom komentar, dan biarkan AI mengelompokkan komentar berdasarkan aspek secara otomatis.</p>
    </div>
    """,
    unsafe_allow_html=True
)

# =========================
# LOAD MODEL + RESOURCES
# =========================
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_REPO)
    model.eval()
    return tokenizer, model

@st.cache_resource
def load_resources():
    LABEL_PATH = hf_hub_download(repo_id=DATASET_REPO, filename="label_encoder.pkl", repo_type="dataset")
    SLANG_PATH = hf_hub_download(repo_id=DATASET_REPO, filename="slang_dict.txt", repo_type="dataset")

    lbl = joblib.load(LABEL_PATH)

    slang_dict = {}
    with open(SLANG_PATH, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) == 2:
                slang, formal = parts
                slang_dict[slang.strip()] = formal.strip()

    return lbl, slang_dict

# load once
tokenizer, model = load_model()
lbl, slang_dict = load_resources()

# =========================
# PREPROCESS
# =========================
def preprocess_text(text):
    text = str(text).lower()
    text = emoji.demojize(text, delimiters=(" ", " "))
    words = text.split()
    words = [slang_dict.get(w, w) for w in words]
    text = " ".join(words)
    text = re.sub(r"http\S+|www\S+|@\w+|#\w+", " ", text)
    text = re.sub(r"[^a-z0-9_\s]", " ", text)
    text = re.sub(r"\b\d+\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# =========================
# BATCH PREDICT + PROGRESS BAR
# =========================
def batch_predict_with_progress(text_list, progress_bar, status_text, batch_size=BATCH_SIZE):
    results_clean = []
    results_label = []

    total = len(text_list)
    done = 0

    model.eval()

    for i in range(0, total, batch_size):
        batch_texts = text_list[i:i+batch_size]
        clean_batch = [preprocess_text(t) for t in batch_texts]

        inputs = tokenizer(
            clean_batch,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )

        with torch.no_grad():
            outputs = model(**inputs)
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()

        labels = lbl.inverse_transform(preds)

        results_clean.extend(clean_batch)
        results_label.extend(labels)

        done += len(batch_texts)
        percent = int((done / total) * 100)
        progress_bar.progress(percent)
        status_text.text(f"Memproses {done:,}/{total:,} komentar ({percent}%)")

    return results_clean, results_label

# =========================
# SIDEBAR
# =========================
st.sidebar.markdown("## 📂 Upload Data")
file = st.sidebar.file_uploader("Upload file (CSV / Excel / TXT)", type=["csv", "xlsx", "txt"])
st.sidebar.markdown("---")
run_btn = st.sidebar.button("🚀 Jalankan Analisis", use_container_width=True)

# =========================
# MAIN
# =========================
if file is not None:
    if file.name.endswith(".csv"):
        df = pd.read_csv(file)
    elif file.name.endswith(".xlsx"):
        df = pd.read_excel(file)
    elif file.name.endswith(".txt"):
        df = pd.DataFrame({"text": file.read().decode("utf-8").splitlines()})
    else:
        st.error("Format file tidak didukung")
        st.stop()

    st.markdown("### 📄 Preview Data")
    st.dataframe(df.head(), use_container_width=True)

    text_col = st.selectbox("Pilih kolom yang berisi komentar", df.columns)

    if run_btn:
        progress_bar = st.progress(0)
        status_text = st.empty()

        texts = df[text_col].astype(str).tolist()
        clean_texts, labels = batch_predict_with_progress(texts, progress_bar, status_text)

        df["clean_text"] = clean_texts
        df["aspect"] = labels

        status_text.text("✅ Analisis selesai!")
        progress_bar.empty()

        # =========================
        # SUMMARY
        # =========================
        st.markdown("## 📊 Ringkasan Aspek")
        aspect_counts = Counter(labels)

        cols = st.columns(len(aspect_counts))
        for i, (asp, cnt) in enumerate(aspect_counts.items()):
            cols[i].markdown(
                f"""
                <div style=\"padding:15px;border-radius:14px;text-align:center; background:rgba(255,255,255,0.12); box-shadow:0 4px 10px rgba(0,0,0,0.15);\">
                    <h3 style=\"margin:0;\">{cnt}</h3>
                    <p style=\"margin:0;font-weight:600;\">{asp}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

        # =========================
        # PER ASPECT (ORDERED & STATE SAFE)
        # =========================
        aspect_colors = {
            "Pujian": "#2ecc71",
            "Perbandingan / Saran / Ide": "#2980b9",
            "Pertanyaan / Request": "#8e44ad",
            "Pengalaman / Hasil": "#16a085",
            "Keluhan / Kesulitan": "#e67e22",
            "Lainnya / Tidak Relevan": "#7f8c8d"
        }

        aspect_desc = {
            "Pujian": "💚 Komentar berisi apresiasi/kebanggaan.",
            "Perbandingan / Saran / Ide": "💡 Komentar memberi perbandingan, masukan, atau ide.",
            "Pertanyaan / Request": "❓ Komentar berupa pertanyaan/permintaan info.",
            "Pengalaman / Hasil": "🧠 Komentar membagikan pengalaman/hasil.",
            "Keluhan / Kesulitan": "⚠️ Komentar menyampaikan kritik/kendala.",
            "Lainnya / Tidak Relevan": "🌀 Komentar umum/tidak terkait langsung."
        }

        custom_order = [
            "Pujian",
            "Perbandingan / Saran / Ide",
            "Pertanyaan / Request",
            "Pengalaman / Hasil",
            "Keluhan / Kesulitan",
            "Lainnya / Tidak Relevan"
        ]

        aspects_order = [a for a in custom_order if a in aspect_counts]

        import html as ihtml2
        for aspect in aspects_order:
            subset = df[df["aspect"] == aspect].copy().drop_duplicates(subset=text_col).reset_index(drop=True)
            color = aspect_colors.get(aspect, "#3498db")
            desc = aspect_desc.get(aspect, "")

            st.markdown(
                f"""
                <div style=\"border-radius:12px;padding:12px 16px;margin:30px 0 15px 0;
                            background: linear-gradient(135deg, {color}, rgba(255,255,255,0.08));
                            box-shadow:0 0 12px rgba(0,0,0,0.25);backdrop-filter:blur(6px);\">
                <h5 style=\"margin:0;font-size:16px;font-weight:600;color:white;\">
                    🎯 Aspek: {aspect} — {len(subset):,} komentar
                </h5>
                <p style=\"margin-top:6px;font-size:13.5px;color:rgba(255,255,255,.85);\">{desc}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

            c1, c2 = st.columns([1, 1.3])

            # WORDCLOUD (WHITE BG)
            with c1:
                text_blob = " ".join(subset["clean_text"].tolist())
                if text_blob.strip():
                    wc = WordCloud(
                        width=500,
                        height=350,
                        background_color="white"
                    ).generate(text_blob)

                    fig, ax = plt.subplots(figsize=(5, 4))
                    ax.imshow(wc, interpolation="bilinear")
                    ax.axis("off")
                    st.pyplot(fig)
                else:
                    st.info("Tidak ada data untuk WordCloud")

            # COMMENTS + PAGINATION (STATE SAFE)
            with c2:
                total = len(subset)
                per_page = 50
                pages = max(1, math.ceil(total / per_page))

                page_key = f"page_aspect_{aspect}"
                if page_key not in st.session_state:
                    st.session_state[page_key] = 1

                page_num = st.number_input(
                    f"Halaman komentar ({aspect})",
                    1, pages, st.session_state[page_key],
                    key=page_key
                )

                start, end = (page_num - 1) * per_page, min(page_num * per_page, total)

                cards = ""
                for _, row in subset.iloc[start:end].iterrows():
                    txt = str(row[text_col]).strip()
                    if not txt:
                        continue
                    safe = ihtml2.escape(txt)
                    cards += f"""
                    <div style=\"border-left:4px solid {color};padding:12px 16px;margin-bottom:10px;
                                background:rgba(255,255,255,.08);
                                border-radius:14px;backdrop-filter:blur(8px);
                                box-shadow:0 3px 10px rgba(0,0,0,.25);\">
                        <p style=\"margin:0;font-size:14.5px;line-height:1.6;color:white;\">
                            💬 {safe}
                        </p>
                    </div>"""

                if cards.strip():
                    components.html(f"""
                    <div style=\"height:400px;padding:12px;border-radius:14px;background:rgba(255,255,255,.05);
                                box-shadow:inset 0 0 12px rgba(255,255,255,.05), 0 4px 10px rgba(0,0,0,.3);
                                backdrop-filter:blur(8px);overflow-y:auto;color:white;\">
                    {cards}
                    </div>
                    """, height=420, scrolling=False)

                    st.caption(f"Menampilkan komentar {start+1:,}–{end:,} dari {total:,}")
                else:
                    st.info("⚠️ Tidak ada komentar pada halaman ini.")

else:
    st.info("⬅️ Upload file di sidebar untuk memulai analisis")
