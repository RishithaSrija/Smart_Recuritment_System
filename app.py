import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
import re
import io
import os

from tensorflow.keras.preprocessing.sequence import pad_sequences

import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

from wordcloud import WordCloud

try:
    import PyPDF2
except:
    pass

# =====================================
# PAGE CONFIG
# =====================================

st.set_page_config(
    page_title="Smart Recruitment Intelligence Platform",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================
# LOAD CSS
# =====================================

with open("style.css") as f:
    st.markdown(
        f"<style>{f.read()}</style>",
        unsafe_allow_html=True
    )

# =====================================
# LOAD MODEL ARTIFACTS
# =====================================

@st.cache_resource
def load_artifacts():

    model = tf.keras.models.load_model(
        "resume_attention_model.keras",
        compile=False
    )

    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    with open("label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)

    return model, tokenizer, label_encoder


model, tokenizer, label_encoder = load_artifacts()

# =====================================
# CONSTANTS
# =====================================

MAX_LEN = 300

SKILLS_DB = [

    "python","java","sql",

    "html","css","javascript",

    "react","angular","nodejs",

    "machine learning",
    "deep learning",
    "nlp",

    "tensorflow",
    "pytorch",

    "aws","azure","gcp",

    "docker",
    "kubernetes",

    "tableau",
    "power bi",

    "mongodb",
    "mysql",

    "postgresql",

    "flask",
    "django",

    "git",
    "github"
]

# =====================================
# HERO SECTION
# =====================================

st.markdown(
"""
<div class='hero-box'>
<h1>🚀 Smart Recruitment Intelligence Platform</h1>
<p>
Transformer Based Resume Ranking,
Candidate Intelligence,
Explainable AI
</p>
</div>
""",
unsafe_allow_html=True
)

# =====================================
# SIDEBAR
# =====================================

st.sidebar.title("⚙ Recruitment Engine")

job_description = st.sidebar.text_area(
    "Paste Job Description",
    height=250
)

uploaded_files = st.sidebar.file_uploader(
    "Upload Resumes (PDF/TXT)",
    type=["pdf","txt"],
    accept_multiple_files=True
)

analyze_button = st.sidebar.button(
    "🚀 Analyze Candidates",
    use_container_width=True
)

# =====================================
# HELPERS
# =====================================

def read_pdf(uploaded_file):

    text = ""

    try:

        pdf = PyPDF2.PdfReader(uploaded_file)

        for page in pdf.pages:

            page_text = page.extract_text()

            if page_text:
                text += page_text

    except:
        pass

    return text


def read_txt(uploaded_file):

    return uploaded_file.read().decode(
        "utf-8",
        errors="ignore"
    )


def extract_skills(text):

    text = str(text).lower()

    found = []

    for skill in SKILLS_DB:

        if skill in text:

            found.append(skill)

    return found


def extract_experience(text):

    text = str(text).lower()

    patterns = [

        r'(\d+)\+?\s*years',

        r'(\d+)\s*yrs',

        r'(\d+)\s*year'

    ]

    years = []

    for pattern in patterns:

        years.extend(
            re.findall(pattern,text)
        )

    if years:

        return max(
            map(int,years)
        )

    return 0


def extract_projects(text):

    text = str(text).lower()

    keywords = [

        "project",
        "developed",
        "implemented",
        "built",
        "created"
    ]

    count = 0

    for k in keywords:

        count += text.count(k)

    return count


def predict_category(text):

    seq = tokenizer.texts_to_sequences(
        [text]
    )

    padded = pad_sequences(
        seq,
        maxlen=MAX_LEN,
        padding="post",
        truncating="post"
    )

    pred = model.predict(
        padded,
        verbose=0
    )

    idx = np.argmax(pred)

    category = label_encoder.inverse_transform(
        [idx]
    )[0]

    confidence = float(
        np.max(pred)
    )

    return category, confidence


# =====================================
# POSITIONAL ENCODING
# =====================================

def positional_encoding(
    position,
    d_model
):

    PE = np.zeros(
        (position,d_model)
    )

    for pos in range(position):

        for i in range(d_model):

            angle = pos / np.power(
                10000,
                (2*(i//2))/d_model
            )

            if i%2==0:

                PE[pos,i]=np.sin(angle)

            else:

                PE[pos,i]=np.cos(angle)

    return PE
# =====================================
# MAIN ANALYSIS
# =====================================

if analyze_button:

    if len(uploaded_files) == 0:

        st.warning(
            "Please upload resumes."
        )

        st.stop()

    if job_description.strip() == "":

        st.warning(
            "Please enter Job Description."
        )

        st.stop()

    jd_skills = extract_skills(
        job_description
    )

    jd_experience = extract_experience(
        job_description
    )

    jd_projects = extract_projects(
        job_description
    )

    candidates = []

    progress = st.progress(0)

    total_files = len(uploaded_files)

    for i, file in enumerate(uploaded_files):

        if file.name.endswith(".pdf"):

            resume_text = read_pdf(file)

        else:

            resume_text = read_txt(file)

        category, confidence = predict_category(
            resume_text
        )

        skills = extract_skills(
            resume_text
        )

        experience = extract_experience(
            resume_text
        )

        projects = extract_projects(
            resume_text
        )

        # =========================
        # Skill Match
        # =========================

        common_skills = set(
            skills
        ).intersection(
            set(jd_skills)
        )

        skill_match = (

            len(common_skills)

            /

            max(len(jd_skills),1)

        ) * 100

        # =========================
        # Experience Match
        # =========================

        if jd_experience == 0:

            experience_match = 100

        else:

            experience_match = min(

                experience /
                jd_experience,

                1

            ) * 100

        # =========================
        # Project Match
        # =========================

        if jd_projects == 0:

            project_match = 100

        else:

            project_match = min(

                projects /
                jd_projects,

                1

            ) * 100

        # =========================
        # Final Score
        # =========================

        final_score = (

            0.5 * skill_match

            +

            0.3 * experience_match

            +

            0.2 * project_match

        )

        candidates.append({

            "Candidate":
                file.name,

            "Category":
                category,

            "Confidence":
                round(confidence*100,2),

            "Skill Match":
                round(skill_match,2),

            "Experience Match":
                round(experience_match,2),

            "Project Match":
                round(project_match,2),

            "Final Score":
                round(final_score,2),

            "Matched Skills":
                ", ".join(common_skills),

            "Resume":
                resume_text

        })

        progress.progress(
            (i+1)/total_files
        )

    ranked_df = pd.DataFrame(
        candidates
    )

    ranked_df = ranked_df.sort_values(
        by="Final Score",
        ascending=False
    )

    # =================================
    # KPI SECTION
    # =================================

    st.markdown(
        "<h2 style='color:white'>📊 Recruitment Analytics</h2>",
        unsafe_allow_html=True
    )

    avg_score = ranked_df[
        "Final Score"
    ].mean()

    top_score = ranked_df[
        "Final Score"
    ].max()

    total_candidates = len(
        ranked_df
    )

    unique_categories = ranked_df[
        "Category"
    ].nunique()

    c1,c2,c3,c4 = st.columns(4)

    c1.metric(
        "Candidates",
        total_candidates
    )

    c2.metric(
        "Average Match",
        f"{avg_score:.1f}%"
    )

    c3.metric(
        "Top Score",
        f"{top_score:.1f}%"
    )

    c4.metric(
        "Categories",
        unique_categories
    )

    # =================================
    # TABS
    # =================================

    tab1,tab2,tab3,tab4 = st.tabs([

        "🏆 Rankings",

        "🔍 Explainability",

        "📈 Analytics",

        "⚡ Transformer"

    ])

    # =================================
    # TAB 1
    # =================================

    with tab1:

        st.subheader(
            "Top 10 Candidates"
        )

        st.dataframe(
            ranked_df.head(10),
            use_container_width=True
        )

        csv = ranked_df.to_csv(
            index=False
        )

        st.download_button(

            "⬇ Download Results",

            csv,

            "ranked_candidates.csv",

            "text/csv"
        )

        fig = px.bar(

            ranked_df.head(10),

            x="Candidate",

            y="Final Score",

            color="Final Score",

            title="Candidate Ranking"

        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # =================================
    # TAB 2
    # =================================

    with tab2:

        best = ranked_df.iloc[0]

        st.success(

            f"Top Candidate: {best['Candidate']}"

        )

        colA,colB = st.columns(2)

        with colA:

            st.write(
                "Predicted Category:",
                best["Category"]
            )

            st.write(
                "Confidence:",
                best["Confidence"]
            )

            st.write(
                "Final Score:",
                best["Final Score"]
            )

        with colB:

            st.write(
                "Skill Match:",
                best["Skill Match"]
            )

            st.write(
                "Experience Match:",
                best["Experience Match"]
            )

            st.write(
                "Project Match:",
                best["Project Match"]
            )

        st.subheader(
            "Matching Evidence"
        )

        st.info(
            best["Matched Skills"]
        )

    # =================================
    # TAB 3
    # =================================

    with tab3:

        st.subheader(
            "Category Distribution"
        )

        cat_df = ranked_df[
            "Category"
        ].value_counts()

        fig = px.pie(

            values=cat_df.values,

            names=cat_df.index,

            title="Resume Categories"

        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.subheader(
            "Skill Word Cloud"
        )

        all_skills = " ".join(

            ranked_df["Matched Skills"]

        )

        if all_skills.strip():

            wc = WordCloud(

                width=1200,

                height=600,

                background_color="white"

            ).generate(
                all_skills
            )

            fig,ax = plt.subplots(
                figsize=(10,5)
            )

            ax.imshow(wc)

            ax.axis("off")

            st.pyplot(fig)

    # =================================
    # TAB 4
    # =================================

    with tab4:

        st.subheader(
            "Positional Encoding Heatmap"
        )

        pe = positional_encoding(
            50,
            128
        )

        fig,ax = plt.subplots(
            figsize=(10,4)
        )

        im = ax.imshow(
            pe,
            aspect='auto'
        )

        plt.colorbar(im)

        st.pyplot(fig)

        st.subheader(
            "Self-Attention Architecture"
        )

        st.code(
"""
Embedding
    ↓
MultiHeadAttention
    ↓
GlobalAveragePooling
    ↓
Dense
    ↓
Softmax
"""
        )

# =====================================
# FOOTER
# =====================================

st.markdown(
"""
<div class='footer'>
Built using NLP, Transformers,
Self-Attention,
Positional Encoding,
Explainable AI
</div>
""",
unsafe_allow_html=True
)