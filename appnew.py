import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import time

# ----------------------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------------------
st.set_page_config(
    page_title="Prediction & Recommendation System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------------------------------------------------------------
# GLOBAL LIGHT THEME STYLING
# ----------------------------------------------------------------------------------
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&display=swap');

        html, body, [class*="css"] {
            font-family: 'Poppins', sans-serif;
        }

        /* ===== Animated app background ===== */
        .stApp {
            background: linear-gradient(135deg, #f5f7fa 0%, #eef2f7 50%, #f0f4f8 100%);
            background-size: 200% 200%;
            animation: gradientShift 18s ease infinite;
            color: #24292f;
        }
        @keyframes gradientShift {
            0% {background-position: 0% 50%;}
            50% {background-position: 100% 50%;}
            100% {background-position: 0% 50%;}
        }

        /* Layout container */
        .block-container {
            padding-top: 1.2rem;
            padding-right: 3rem;
            padding-left: 3rem;
            max-width: 100%;
        }

        /* ===== Headings ===== */
        h1, h2, h3, h4 {
            color: #1a1a2e !important;
        }

        /* ===== Sidebar ===== */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #ffffff 0%, #f3f6fa 100%);
            border-right: 1px solid #e1e5ea;
        }
        section[data-testid="stSidebar"] * {
            color: #24292f !important;
        }

        /* ===== Tabs ===== */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background-color: #ffffff;
            padding: 6px;
            border-radius: 14px;
            border: 1px solid #e1e5ea;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.04);
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            border-radius: 10px;
            color: #57606a;
            font-weight: 600;
            font-size: 1.05rem;
            background-color: transparent;
            transition: all 0.25s ease-in-out;
        }
        .stTabs [data-baseweb="tab"]:hover {
            background-color: rgba(47, 129, 247, 0.08);
            color: #1f6feb;
        }
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #2f81f7 0%, #1f6feb 100%);
            color: #ffffff !important;
            box-shadow: 0 4px 14px rgba(47, 129, 247, 0.30);
        }

        /* ===== Cards with fade-in + hover lift ===== */
        .glass-card {
            background-color: #ffffff;
            border: 1px solid #e1e5ea;
            border-radius: 16px;
            padding: 1.5rem 1.7rem;
            margin-bottom: 1.3rem;
            box-shadow: 0 4px 16px rgba(31, 35, 40, 0.06);
            animation: fadeInUp 0.5s ease-out;
            transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
        }
        .glass-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 10px 28px rgba(47, 129, 247, 0.12);
            border-color: #cfe0fb;
        }
        @keyframes fadeInUp {
            from {opacity: 0; transform: translateY(14px);}
            to {opacity: 1; transform: translateY(0);}
        }

        .section-title {
            font-size: 1.15rem;
            font-weight: 700;
            color: #1f6feb;
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        /* ===== Inputs ===== */
        .stSelectbox div[data-baseweb="select"] > div,
        .stNumberInput input,
        .stTextInput input {
            background-color: #f6f8fa !important;
            color: #1a1a2e !important;
            border: 1px solid #d0d7de !important;
            border-radius: 10px !important;
            transition: border-color 0.2s ease, box-shadow 0.2s ease;
        }
        .stSelectbox div[data-baseweb="select"] > div:focus-within,
        .stNumberInput input:focus,
        .stTextInput input:focus {
            border-color: #2f81f7 !important;
            box-shadow: 0 0 0 3px rgba(47, 129, 247, 0.15) !important;
        }
        label {
            color: #24292f !important;
            font-weight: 500 !important;
        }

        /* ===== Progress bars under inputs ===== */
        div[data-testid="stProgress"] > div > div {
            background: linear-gradient(90deg, #2f81f7, #2ea043) !important;
            border-radius: 6px;
        }
        div[data-testid="stProgress"] > div {
            background-color: #e9ecef !important;
            border-radius: 6px;
        }

        /* ===== Buttons ===== */
        .stButton > button {
            background: linear-gradient(135deg, #2f81f7 0%, #1f6feb 100%);
            color: #ffffff;
            border: none;
            border-radius: 12px;
            padding: 0.65rem 1.6rem;
            font-weight: 700;
            font-size: 1.02rem;
            transition: transform 0.15s ease-in-out, box-shadow 0.15s ease-in-out, filter 0.15s ease;
            box-shadow: 0 4px 14px rgba(31, 111, 235, 0.25);
        }
        .stButton > button:hover {
            transform: translateY(-2px) scale(1.01);
            box-shadow: 0 6px 22px rgba(31, 111, 235, 0.40);
            filter: brightness(1.05);
            color: #ffffff;
        }
        .stButton > button:active {
            transform: translateY(0px) scale(0.99);
        }

        /* ===== Result badges ===== */
        .result-badge {
            display: inline-block;
            padding: 0.4rem 1.2rem;
            border-radius: 999px;
            font-weight: 800;
            font-size: 1.15rem;
            letter-spacing: 0.5px;
            animation: popIn 0.4s ease-out;
        }
        @keyframes popIn {
            0% {transform: scale(0.6); opacity: 0;}
            70% {transform: scale(1.08);}
            100% {transform: scale(1); opacity: 1;}
        }
        .badge-high-risk {
            background-color: rgba(215, 58, 73, 0.10);
            color: #d73a49;
            border: 1px solid #d73a49;
            box-shadow: 0 0 14px rgba(215, 58, 73, 0.18);
        }
        .badge-no-risk {
            background-color: rgba(46, 160, 67, 0.10);
            color: #2ea043;
            border: 1px solid #2ea043;
            box-shadow: 0 0 14px rgba(46, 160, 67, 0.18);
        }

        /* ===== Metric cards ===== */
        div[data-testid="stMetric"] {
            background-color: #f6f8fa;
            border: 1px solid #e1e5ea;
            border-radius: 12px;
            padding: 0.8rem 1rem;
            transition: transform 0.2s ease, border-color 0.2s ease;
        }
        div[data-testid="stMetric"]:hover {
            transform: translateY(-2px);
            border-color: #2f81f7;
        }
        div[data-testid="stMetricValue"] {
            color: #1a1a2e !important;
        }
        div[data-testid="stMetricLabel"] {
            color: #57606a !important;
        }

        /* ===== Recommendation pills ===== */
        .course-pill {
            display: flex;
            align-items: center;
            gap: 10px;
            background-color: rgba(46, 160, 67, 0.06);
            border: 1px solid rgba(46, 160, 67, 0.30);
            color: #1a7f37;
            border-radius: 12px;
            padding: 0.7rem 1.1rem;
            margin-bottom: 0.6rem;
            font-weight: 600;
            transition: transform 0.18s ease, box-shadow 0.18s ease, background-color 0.18s ease;
            animation: fadeInUp 0.4s ease-out;
        }
        .course-pill:hover {
            transform: translateX(4px);
            background-color: rgba(46, 160, 67, 0.12);
            box-shadow: 0 4px 14px rgba(46, 160, 67, 0.15);
        }
        .pill-number {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            min-width: 26px;
            height: 26px;
            border-radius: 50%;
            background: linear-gradient(135deg, #2ea043, #1a7f37);
            color: #ffffff;
            font-weight: 800;
            font-size: 0.85rem;
        }
        .info-pill {
            display: inline-block;
            background-color: rgba(47, 129, 247, 0.08);
            border: 1px solid rgba(47, 129, 247, 0.30);
            color: #1f6feb;
            border-radius: 12px;
            padding: 0.55rem 1.2rem;
            font-weight: 700;
            transition: transform 0.18s ease, box-shadow 0.18s ease;
        }
        .info-pill:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 14px rgba(47, 129, 247, 0.18);
        }
        .field-label {
            color: #57606a;
            font-weight: 600;
            font-size: 0.95rem;
            margin-bottom: 0.4rem;
        }

        /* ===== Misc ===== */
        hr {
            border-color: #e1e5ea !important;
        }
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True
)

# ----------------------------------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center; margin-bottom: 1rem;">
            <div style="font-size: 2.6rem;">🎓</div>
            <div style="font-size: 1.2rem; font-weight: 800; color: #1a1a2e;">
                Student Insights
            </div>
            <div style="font-size: 0.85rem; color: #57606a;">
                Risk Prediction & Learning Paths
            </div>
        </div>
        <hr>
        """,
        unsafe_allow_html=True
    )

    st.markdown("#### 🧭 How to use")
    st.markdown(
        """
        <div style="font-size:0.92rem; color:#24292f; line-height:1.6;">
        <b>1.</b> Go to <b>Risk Prediction</b>, fill in student details and click
        <i>Get Prediction</i>.<br><br>
        <b>2.</b> Go to <b>Learning Path Recommendation</b>, enter a Student ID and click
        <i>Get Recommendations</i>.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("#### 🎨 Legend")
    st.markdown(
        """
        <span class="result-badge badge-no-risk" style="font-size:0.85rem;">No Risk</span>
        &nbsp;
        <span class="result-badge badge-high-risk" style="font-size:0.85rem;">High Risk</span>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<hr>", unsafe_allow_html=True)
    st.caption("Built with ❤️ using Streamlit")

# ----------------------------------------------------------------------------------
# HEADER
# ----------------------------------------------------------------------------------
st.markdown(
    """
    <h1 style="
        text-align: center;
        background: linear-gradient(135deg, #1f6feb 0%, #2f81f7 50%, #2ea043 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.7rem;
        margin-bottom: 0.2rem;
    ">
        🎓 Prediction and Recommendation System
    </h1>
    <p style="text-align:center; color:#57606a; font-size:1rem; margin-bottom:1.5rem;">
        Estimate student risk and discover a personalized learning path
    </p>
    """,
    unsafe_allow_html=True
)

# ----------------------------------------------------------------------------------
# LOAD MODELS (paths unchanged)
# ----------------------------------------------------------------------------------
logClassifier = joblib.load('output/log_classifier_model.pkl')

# ----------------------------------------------------------------------------------
# TABS
# ----------------------------------------------------------------------------------
tab1, tab2 = st.tabs(["🎯  Risk Prediction", "📚  Learning Path Recommendation"])

# ====================================================================================
# TAB 1: RISK PREDICTION
# ====================================================================================
with tab1:

    # ---- Basic Info ----
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🧍 Basic Info</div>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        Gender = st.selectbox("Gender", ['Male', 'Female'])
    with col2:
        Disability = st.selectbox("Disability", ['Yes', 'No'])
    with col3:
        Age = st.selectbox("Age", ['0-35', '35-55', '55<='])
    with col4:
        ImdBand = st.selectbox("Imd Band", [
            '0-10%', '10-20%', '20-30%', '30-40%', '40-50%',
            '50-60%', '60-70%', '70-80%', '80-90%', '90-100%'
        ])
    st.markdown('</div>', unsafe_allow_html=True)

    # ---- Clicks Info ----
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🖱️ Clicks Info</div>', unsafe_allow_html=True)
    col5, col6, col7 = st.columns(3)
    with col5:
        HomePageTotalClicks = st.number_input(
            "Home Page Total Clicks", min_value=0.0, max_value=70.0, value=0.0, step=1.0,
            help="Range: 0 to 70"
        )
        st.progress(HomePageTotalClicks / 70.0)

        ForumngTotalClicks = st.number_input(
            "Forumng Total Clicks", min_value=0.0, max_value=90.0, value=0.0, step=1.0,
            help="Range: 0 to 90"
        )
        st.progress(ForumngTotalClicks / 90.0)
    with col6:
        SubPageTotalClicks = st.number_input(
            "Sub Page Total Clicks", min_value=0.0, max_value=40.0, value=0.0, step=1.0,
            help="Range: 0 to 40"
        )
        st.progress(SubPageTotalClicks / 40.0)

        OutContentTotalClicks = st.number_input(
            "Out Content Total Clicks", min_value=0.0, max_value=150.0, value=0.0, step=1.0,
            help="Range: 0 to 150"
        )
        st.progress(OutContentTotalClicks / 150.0)
    with col7:
        ResourceTotalClicks = st.number_input(
            "Resource Total Clicks", min_value=0.0, max_value=10.0, value=0.0, step=1.0,
            help="Range: 0 to 10"
        )
        st.progress(ResourceTotalClicks / 10.0)
    st.markdown('</div>', unsafe_allow_html=True)

    # ---- Other Info ----
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📊 Other Info</div>', unsafe_allow_html=True)
    col8, col9, col10 = st.columns(3)
    with col8:
        AverageScore = st.number_input(
            "Average Score", min_value=0.0, max_value=100.0, value=0.0, step=1.0,
            help="Range: 0 to 100"
        )
        st.progress(AverageScore / 100.0)
    with col9:
        StudiedCredits = st.number_input(
            "Studied Credits", min_value=30.0, max_value=600.0, value=30.0, step=1.0,
            help="Range: 30 to 600"
        )
        st.progress((StudiedCredits - 30.0) / (600.0 - 30.0))
    with col10:
        NumofPrevAttempts = st.number_input(
            "Num of Prev Attempts", min_value=0.0, max_value=6.0, value=0.0, step=1.0,
            help="Range: 0 to 6"
        )
        st.progress(NumofPrevAttempts / 6.0)
    st.markdown('</div>', unsafe_allow_html=True)

    # ---- Button & Output ----
    col_btn = st.columns([1, 1, 1])
    with col_btn[1]:
        predict_clicked = st.button("🔍 Get Prediction", use_container_width=True)

    if predict_clicked:
        with st.spinner("🔬 Analyzing student profile..."):
            time.sleep(0.5)

            # Collect inputs
            data = {
                'gender': [Gender],
                'disability': [Disability],
                'age_band': [Age],
                'imd_band': [ImdBand],
                'homepage_totalclicks': [float(HomePageTotalClicks)],
                'subpage_totalclicks': [float(SubPageTotalClicks)],
                'forumng_totalclicks': [float(ForumngTotalClicks)],
                'outcontent_totalclicks': [float(OutContentTotalClicks)],
                'resource_totalclicks': [float(ResourceTotalClicks)],
                'average_score': [float(AverageScore)],
                'studied_credits': [float(StudiedCredits)],
                'num_of_prev_attempts': [float(NumofPrevAttempts)],
            }

            input_df = pd.DataFrame(data)

            # Apply transformations
            scaler = joblib.load('output/scaler.pkl')
            minmax_scaler = joblib.load('output/minmax_scaler.pkl')

            float_click_cols = [col for col in input_df.columns if 'totalclicks' in col]
            input_df[float_click_cols] = scaler.transform(input_df[float_click_cols])
            input_df[['average_score']] = minmax_scaler.transform(input_df[['average_score']])

            # Encode categorical
            input_df['gender'] = input_df['gender'].map({'Male': 0, 'Female': 1})
            input_df['disability'] = input_df['disability'].map({'No': 0, 'Yes': 1})
            input_df['imd_band'] = input_df['imd_band'].map({
                '0-10%':0, '10-20%':1, '20-30%':2, '30-40%':3, '40-50%':4,
                '50-60%':5, '60-70%':6, '70-80%':7, '80-90%':8, '90-100%':9
            })
            input_df['age_band'] = input_df['age_band'].map({'55<=':2, '35-55':1, '0-35':0})

            # Dummy model/prediction logic
            def predict_fn(input_df):

                model_input = input_df[[
                'average_score', 'subpage_totalclicks', 'imd_band', 'age_band', 'homepage_totalclicks',
                'forumng_totalclicks', 'outcontent_totalclicks', 'resource_totalclicks', 'studied_credits',
                'num_of_prev_attempts', 'disability', 'gender'
                ]]

                prediction = logClassifier.predict(model_input)[0]
                probability = logClassifier.predict_proba(model_input)[0][1]

                result = "No Risk" if prediction == 1 else "High Risk"

                return result, probability

            pred, prob = predict_fn(input_df)

            positive_class_prob = round(prob * 100, 2)

        # ---- Output Section ----
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📈 Prediction Result</div>', unsafe_allow_html=True)

        if pred == "High Risk":
            badge_class = "badge-high-risk"
            gauge_color = "#d73a49"
            risk_emoji = "⚠️"
        else:
            badge_class = "badge-no-risk"
            gauge_color = "#2ea043"
            risk_emoji = "✅"

        res_col1, res_col2 = st.columns([1, 1.4])

        with res_col1:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(
                f"""
                <div style="text-align:center;">
                    <p style="color:#57606a; font-weight:600; margin-bottom:0.4rem;">Prediction</p>
                    <span class="result-badge {badge_class}">{risk_emoji} {pred}</span>
                </div>
                """,
                unsafe_allow_html=True
            )
            st.markdown("<br>", unsafe_allow_html=True)

            m1, m2 = st.columns(2)
            with m1:
                st.metric("Confidence (No Risk)", f"{positive_class_prob}%")
            with m2:
                st.metric("Confidence (High Risk)", f"{round(100 - positive_class_prob, 2)}%")

        with res_col2:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=positive_class_prob,
                number={'suffix': "%", 'font': {'color': '#1a1a2e', 'size': 36}},
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [0, 100], 'tickcolor': '#57606a', 'tickfont': {'color': '#57606a'}},
                    'bar': {'color': gauge_color, 'thickness': 0.3},
                    'bgcolor': "rgba(0,0,0,0)",
                    'borderwidth': 0,
                    'steps': [
                        {'range': [0, 40], 'color': 'rgba(215, 58, 73, 0.15)'},
                        {'range': [40, 70], 'color': 'rgba(191, 135, 0, 0.15)'},
                        {'range': [70, 100], 'color': 'rgba(46, 160, 67, 0.15)'}
                    ],
                }
            ))
            fig.update_layout(
                height=240,
                margin=dict(l=20, r=20, t=20, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                font={'color': "#24292f", 'family': "Poppins"}
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

        # ---- Celebratory feedback ----
        if pred == "No Risk":
            st.balloons()
        else:
            st.toast("This student may need extra support 💡", icon="⚠️")

        # ---- Input summary (expandable) ----
        with st.expander("🔬 View model input details"):
            st.dataframe(input_df, use_container_width=True)


# ====================================================================================
# TAB 2: LEARNING PATH RECOMMENDATION
# ====================================================================================
with tab2:

    recomend_data = pd.read_csv("output/data.csv")
    gb_model = joblib.load('output/GB.pkl')

    def recommend_learning_path(student_id):
        # Concatenate the DataFrame and IDs
        student_id = int(student_id.replace(" ", ""))

        # Get the data for the specific student
        student_data = recomend_data[recomend_data['id_student'] == student_id]

        # Check if student exists in the data
        if student_data.empty:
            return "Student not found.", "", ""

        # Remove unnecessary columns for prediction
        drop_cols = [c for c in ['id_student', 'study_method_preference', 'Unnamed: 0'] if c in student_data.columns]
        student_data = student_data.drop(columns=drop_cols)

        # Extract engagement level before reordering
        engagement = student_data["engagement_classification"].iloc[0]

        # Reorder columns to exactly match training order
        model_features = gb_model.feature_names_in_.tolist()
        student_data = student_data[model_features]

        # Predict the study method preference
        predicted_label = gb_model.predict(student_data)

        # Recommendations based on study method and engagement
        recommendations = {
            0: {  # Collaborative
                0: ["Interactive AI Basics: Weekly Quizzes and Forums","Applied AI: Practical Exercises with Peer Feedback"
                   ,"Introduction to Machine Learning: Online Workshops","AI Ethics: Case Studies and Discussion Groups"],  # Moderate Engagement
                1: ["Collaborative AI Projects: Team-Based Learning","Advanced AI Techniques: Group Workshops and Peer Reviews"
                   ,"Machine Learning Bootcamp: Intensive Group Projects","AI in Practice: Team Challenges and Hackathons"],      # High Engagement
                2: ['Introduction to AI: Self-Paced Fundamentals', 'AI Basics: Introductory Video Series',
                    'Foundations of Machine Learning: Self-Study Edition','AI for Everyone: Introductory Readings and Quizzes'] # Low Engagement
            },
            1: {  # Offline Content
                0: ["AI Principles: Self-Study with Case Studies", "Machine Learning: Offline Course with Practice Problems",
                    "Applied AI: Textbook and Supplementary Materials", "Data Science: Case Studies and Analytical Exercises"], # Moderate Engagement
                1: ["Advanced AI: Comprehensive Textbook with Projects", "Deep Learning: In-Depth Study with Capstone Projects",
                   "AI and Machine Learning: Project-Based Learning", "Data Science Mastery: Offline Content with Comprehensive Projects"], # High Engagement
                2: ['AI Basics: Essential Readings and Key Concepts', 'Machine Learning Fundamentals: Self-Study Workbook',
                   "AI Concepts: Downloadable Lecture Series", "Introduction to Data Science: Offline Learning Modules"] # Low Engagement
            },
            2: {  # Interactive
                0: ["Machine Learning: Interactive Coding Exercises", "AI Applications: Interactive Case Studies",
                   "Data Science: Interactive Projects and Peer Reviews", "AI Ethics: Discussion Forums and Interactive Scenarios"], # Moderate Engagement
                1: ["Advanced AI: Interactive Group Projects and Hackathons", "Deep Learning: Interactive Labs and Collaborative Projects",
                   "Machine Learning Mastery: Interactive Workshops and Challenges","AI Research: Collaborative Research Projects and Peer Feedback"], # High Engagement
                2: ["AI Basics: Interactive Quizzes and Flashcards", "Introduction to Machine Learning: Interactive Visualizations",
                   "AI Fundamentals: Interactive Notebooks", "AI Concepts: Gamified Learning Modules"] # Low Engagement
            },
            3: {  # Informational
                0: ["Machine Learning: Structured Video Course", "AI Concepts: Comprehensive Video Series",
                   "Data Science: Interactive Reading and Video Modules", "AI in Practice: Lecture Notes and Case Studies"], # Moderate Engagement
                1: ["Advanced AI: Detailed Lecture Series and Readings", "Deep Learning: Advanced Lecture Series with Supplemental Readings",
                   "AI and Machine Learning: Research Papers and Advanced Lectures", "Data Science Masterclass: Comprehensive Reading and Video Content"], # High Engagement
                2: ["AI Overview: Short Video Lectures", "Introduction to Machine Learning: Podcast Series",
                   "AI Fundamentals: Infographics and Summaries", "Data Science: Essential Readings and Articles"] # Low Engagement
            },
            4: {  # Resource-Based
                0: ["Machine Learning: Comprehensive eBooks and Guides", "AI Applications: Case Study Compilations",
                   "Data Science: In-Depth Articles and White Papers", "AI Concepts: Research Articles and Detailed Guides"], # Moderate Engagement
                1: ["Advanced AI: Research Papers and Technical Reports", "Deep Learning: Comprehensive Textbooks and Resource Repositories",
                   "Machine Learning Mastery: Advanced Documentation and APIs", "AI Ethics: Government and Institutional Reports"], # High Engagement
                2: ["AI Basics: Curated Reading Lists", "Introduction to Machine Learning: Beginner-Friendly Blogs",
                   "Data Science Overview: Quick Reference Guides", "AI Fundamentals: Online Documentation"] # Low Engagement
            }
        }

        # Determine study method and engagement level
        study_method = predicted_label[0]
        engagement_level = engagement.iloc[0] if isinstance(engagement, pd.Series) else engagement

        # Get the recommended courses based on study method and engagement level
        recommended_courses = recommendations.get(study_method, {}).get(engagement_level, [])

        return recommended_courses, study_method, engagement_level

    def return_map_to_original_preference(x):
        if x == 0:
            return 'Collaborative'
        elif x == 1:
            return 'Offline Content'
        elif x == 2:
            return 'Interactive'
        elif x == 3:
            return 'Informational'
        elif x == 4:
            return 'Resource-Based'

    def return_map_to_original_engagement(x):
        if x == 0:
            return 'Moderate Engagement'
        elif x == 1:
            return 'High Engagement'
        elif x == 2:
            return 'Low Engagement'

    # Icons for study methods & engagement, purely cosmetic
    study_method_icons = {
        'Collaborative': '🤝',
        'Offline Content': '📦',
        'Interactive': '🎮',
        'Informational': '📰',
        'Resource-Based': '📚'
    }
    engagement_icons = {
        'High Engagement': '🔥',
        'Moderate Engagement': '⚡',
        'Low Engagement': '🌙'
    }

    # ---- Student ID input ----
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🔎 Find Student</div>', unsafe_allow_html=True)

    input_col, btn_col = st.columns([3, 1])
    with input_col:
        student_id = st.text_input(
            "Student ID",
            key="student_id_small",
            placeholder="e.g. 645019",
        )
    with btn_col:
        st.markdown("<div style='height: 1.75rem;'></div>", unsafe_allow_html=True)
        recommend_clicked = st.button("📚 Get Recommendations", use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

    if recommend_clicked and student_id:
        with st.spinner("🔍 Looking up student & generating recommendations..."):
            time.sleep(0.4)
            recommendations, study_method, engagement = recommend_learning_path(student_id)

        if recommendations == "Student not found.":
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown(
                f"<p style='color:#d73a49; font-weight:700; text-align:center; font-size:1.1rem;'>"
                f"❌ {recommendations}</p>",
                unsafe_allow_html=True
            )
            st.markdown('</div>', unsafe_allow_html=True)

        else:
            study_method_label = return_map_to_original_preference(study_method)
            engagement_label = return_map_to_original_engagement(engagement)

            sm_icon = study_method_icons.get(study_method_label, '📖')
            eng_icon = engagement_icons.get(engagement_label, '⚡')

            # ---- Profile summary card ----
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown(
                f'<div class="section-title">🧾 Profile for Student {student_id}</div>',
                unsafe_allow_html=True
            )

            prof_col1, prof_col2 = st.columns(2)
            with prof_col1:
                st.markdown('<div class="field-label">Study Method Preference</div>', unsafe_allow_html=True)
                st.markdown(f'<span class="info-pill">{sm_icon} {study_method_label}</span>', unsafe_allow_html=True)
            with prof_col2:
                st.markdown('<div class="field-label">Engagement Level</div>', unsafe_allow_html=True)
                st.markdown(f'<span class="info-pill">{eng_icon} {engagement_label}</span>', unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

            # ---- Recommended courses card ----
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">🎯 Recommended Courses</div>', unsafe_allow_html=True)

            for i, course in enumerate(recommendations, start=1):
                st.markdown(
                    f'<div class="course-pill"><span class="pill-number">{i}</span> {course}</div>',
                    unsafe_allow_html=True
                )

            st.markdown('</div>', unsafe_allow_html=True)

            st.toast(f"Recommendations ready for student {student_id} 🎉", icon="✅")
