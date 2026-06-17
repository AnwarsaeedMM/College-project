import streamlit as st
import joblib           
import pandas as pd

st.markdown(
    """
    <style>
        /* Override default Streamlit container width and alignment */
        .block-container {
            padding-top: 1rem;
            padding-right: 2rem;
            padding-left: 2rem;
            max-width: 100%;
        }

        /* Optional: set width for columns inside Streamlit if needed */
        .element-container {
            width: 100%;
        }
    </style>
    """,
    unsafe_allow_html=True
)


st.markdown(
    """
    <h1 style="
        text-align: center;
        color: #000000;
        -webkit-text-stroke: 2px #ffffff;
        font-weight: 800;
    ">
        Prediction and Recommendation System
    </h1>
    """,
    unsafe_allow_html=True
)





# Load the trained model
logClassifier = joblib.load('output/log_classifier_model.pkl')

# Section 1: Basic Info (Dropdowns)
st.subheader("Enter Basic Info :-")
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

# Section 2: Clicks Info
st.subheader("Enter Clicks Info :-")
col5, col6, col7 = st.columns(3)
with col5:
    HomePageTotalClicks = st.text_input("Home Page Total Clicks" ,placeholder="0 to 70")
    ForumngTotalClicks = st.text_input("Forumng Total Clicks", placeholder="0 to 90")
with col6:
    SubPageTotalClicks = st.text_input("Sub Page Total Clicks", placeholder="0 to 40")
    OutContentTotalClicks = st.text_input("Out Content Total Clicks", placeholder="0 to 150")
with col7:
    ResourceTotalClicks =st.text_input("Resource Total Clicks", placeholder="0 to 10")

# Section 3: Other Info
st.subheader("Enter Other Info :-")
col8, col9, col10 = st.columns(3)
with col8:
    AverageScore = st.text_input("Average Score", placeholder="0 to 100")
with col9:
    StudiedCredits = st.text_input("Studied Credits", placeholder="30 to 600")
with col10:
    NumofPrevAttempts = st.text_input("Num of Prev Attempts", placeholder="0 to 6")

# Button and Output Section
if st.button("Get Prediction"):
    # Collect inputs
    data = {
        'gender': [Gender],
        'disability': [Disability],
        'age_band': [Age],
        'imd_band': [ImdBand],
        'homepage_totalclicks': [float(HomePageTotalClicks or 0)],
       'subpage_totalclicks': [float(SubPageTotalClicks or 0)],
       'forumng_totalclicks': [float(ForumngTotalClicks or 0)],
       'outcontent_totalclicks': [float(OutContentTotalClicks or 0)],
       'resource_totalclicks': [float(ResourceTotalClicks or 0)],
       'average_score': [float(AverageScore or 0)],
       'studied_credits': [float(StudiedCredits or 0)],
       'num_of_prev_attempts': [float(NumofPrevAttempts or 0)],
    }

    input_df = pd.DataFrame(data)

    # Apply transformations
    scaler = joblib.load('output/scaler.pkl')
    minmax_scaler = joblib.load('output/minmax_scaler.pkl')

    float_click_cols = [col for col in input_df.columns if 'totalclicks' in col]
    input_df[float_click_cols] = scaler.transform(input_df[float_click_cols])
    input_df[['average_score']] = minmax_scaler.transform(input_df[['average_score']])

    # Encode categorical
   input_df['gender'] = input_df['gender'].replace({'Male': 0, 'Female': 1})
input_df['disability'] = input_df['disability'].replace({'No': 0, 'Yes': 1})
input_df['imd_band'] = input_df['imd_band'].replace({...})
input_df['age_band'] = input_df['age_band'].replace({'55<=':2, '35-55':1, '0-35':0})

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

        # print("Predicted class:", prediction)
        # print("Probability:", probability)


        return result, probability

    pred, prob = predict_fn(input_df)

    positive_class_prob = round(prob * 100, 2)

    st.subheader("Output :")

    # Conditional styling based on prediction value
    if pred == "High Risk":
        color = "red"
    else:
        color = "green"

    # Styled output using markdown with HTML
    st.markdown(
        f"Prediction: <span style='color:{color}; font-weight:bold;'>{pred}</span>,     Probability: {positive_class_prob}%",
        unsafe_allow_html=True
    )




# Section: Learning Path Recommendation
st.markdown(
    "<h2 style='text-align: center;'>Learning Path Recommendation</h2>",
    unsafe_allow_html=True
)


col1, col2 = st.columns([.2, 1.5])
with col1:
    st.markdown("**Enter Student ID:**")
with col2:
    student_id = st.text_input("Student ID", key="student_id_small", placeholder="e.g. 645019", label_visibility="collapsed")


recomend_data = pd.read_csv("output/data.csv")

# Load the trained model
gb_model = joblib.load('output/GB.pkl')

def recommend_learning_path(student_id):
    # Concatenate the DataFrame and IDs
    # recomend_data = pd.concat([df, ids], axis=1)
    student_id = int(student_id.replace(" ", ""))
    
    # Get the data for the specific student
    student_data = recomend_data[recomend_data['id_student'] == student_id]
    
    # Check if student exists in the data
    if student_data.empty:
        return "Student not found.","",""
    
    # Remove unnecessary columns for prediction
    student_data.drop(columns=['id_student', 'study_method_preference','Unnamed: 0'], inplace=True)
    
    # Predict the study method preference
    predicted_label = gb_model.predict(student_data)
    
    # Extract engagement level
    engagement = student_data["engagement_classification"].iloc[0]
    
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
    elif x==3:
        return 'Informational'
    elif x==4:
        return 'Resource-Based'
    
def return_map_to_original_engagement(x):
    if x == 0:
        return 'Moderate Engagement'
    elif x == 1:
        return 'High Engagement'
    elif x == 2:
        return 'Low Engagement'
    
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    recommend_clicked = st.button("Get Recommendations")

if recommend_clicked and student_id:
    recommendations, study_method, engagement = recommend_learning_path(student_id)
    study_method = return_map_to_original_preference(study_method)
    engagement = return_map_to_original_engagement(engagement)

    # print(f"Study method preference for student {student_id}: {study_method}")
    # print(f"Level of engagement for student {student_id}: {engegement}")
    # print(f"Recommended courses for student {student_id}: {recommendations}")

    if recommendations == "Student not found.":
        st.text(recommendations)

    else:
        if student_id:
            col1, col2 = st.columns([0.8, 2]) 
            with col1:
                st.markdown(
                    f"<div style='white-space: nowrap; font-weight: bold;'>"
                    f"Study method preference for student {student_id} :</div>",
                    unsafe_allow_html=True
                )
            with col2:
                st.markdown(
                    f"<div style='color:#1f77b4; font-weight: bold;'>{study_method}</div>",
                    unsafe_allow_html=True
                )


            col1, col2 = st.columns([0.8, 2]) 
            with col1:
                st.markdown(
                    f"<div style='white-space: nowrap; font-weight: bold;'>"
                    f"Level of engagement for student {student_id} :</div>",
                    unsafe_allow_html=True
                )
            with col2:
                st.markdown(
                    f"<div style='color:#1f77b4; font-weight: bold;'>{engagement}</div>",
                    unsafe_allow_html=True
                )

            col1, col2 = st.columns([0.8, 2]) 
            with col1:
                st.markdown(
                    f"<div style='white-space: nowrap; font-weight: bold;'>"
                    f"Recommended courses for student {student_id} :</div>",
                    unsafe_allow_html=True
                )
            with col2:
                colored_output = "<br>".join([f"<span style='color:#2ca02c; font-weight: bold;'>{course}</span>" for course in recommendations])
                st.markdown(colored_output, unsafe_allow_html=True)

#1f77b4 – blue
#d62728 – red
#2ca02c – green
#ff7f0e – orange
#444 – dark gray

