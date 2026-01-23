import streamlit as st
from ui_components import page_header
from config.database import save_resume_data, save_analysis_data
from config.courses import COURSES_BY_CATEGORY, RESUME_VIDEOS, INTERVIEW_VIDEOS, get_courses_for_role, get_category_for_role
from utils.logger import setup_logger

logger = setup_logger(__name__)

def render_empty_state(icon, message):
    """Render an empty state with icon and message"""
    return f"""
        <div class='empty-state-container'>
            <i class='{icon} empty-state-icon'></i>
            <p class='empty-state-message'>{message}</p>
        </div>
    """

def render_analyzer(old_analyzer, new_analyzer, job_roles):
    """Render the resume analyzer page"""
    
    page_header(
        "Resume Analyzer",
        "Get instant AI-powered feedback to optimize your resume"
    )
    
    categories = list(job_roles.keys())
    selected_category = st.selectbox("Job Category", categories)
    
    roles = list(job_roles[selected_category].keys())
    selected_role = st.selectbox("Specific Role", roles)
    
    role_info = job_roles[selected_category][selected_role]
    
    st.markdown(f"""
    <div class="job-role-info-card">
        <h3>{selected_role}</h3>
        <p>{role_info['description']}</p>
        <h4>Required Skills:</h4>
        <p>{', '.join(role_info['required_skills'])}</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload your resume", type=['pdf', 'docx'])
    
    if not uploaded_file:
        st.markdown(
            render_empty_state(
            "fas fa-cloud-upload-alt",
            "Upload your resume to get started with AI-powered analysis"
            ),
            unsafe_allow_html=True
        )

    if uploaded_file:
        with st.spinner("Analyzing your document with our ML models..."):
            text = ""
            try:
                if uploaded_file.type == "application/pdf":
                    text = old_analyzer.extract_text_from_pdf(uploaded_file)
                elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                    text = old_analyzer.extract_text_from_docx(uploaded_file)
                else:
                    text = uploaded_file.getvalue().decode()
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
                return

            # Use the new ML-powered analyzer
            analysis = new_analyzer.analyze_resume({'raw_text': text}, role_info)
            
            # Use the old analyzer to extract data for saving
            personal_info = old_analyzer.extract_personal_info(text)
            
            resume_data = {
                'personal_info': personal_info,
                'summary': old_analyzer.extract_summary(text),
                'target_role': selected_role,
                'target_category': selected_category,
                'education': old_analyzer.extract_education(text),
                'experience': old_analyzer.extract_experience(text),
                'projects': old_analyzer.extract_projects(text),
                'skills': analysis.get('skills', []), # Use skills from new analyzer
                'template': ''
            }
            
            try:
                resume_id = save_resume_data(resume_data)
                
                # Save analysis data from the new analyzer
                analysis_data = {
                    'resume_id': resume_id,
                    'ats_score': analysis.get('ats_score', 0),
                    'keyword_match_score': analysis.get('keyword_match', {}).get('score', 0),
                    'format_score': analysis.get('format_score', 0),
                    'section_score': analysis.get('section_score', 0),
                    'missing_skills': ','.join(analysis.get('keyword_match', {}).get('missing_skills', [])),
                    'recommendations': ', '.join([s['text'] for s in analysis.get('suggestions', [])])
                }
                save_analysis_data(resume_id, analysis_data)
                st.success("Resume data saved successfully!")
                logger.info(f"ML analysis saved for user {st.session_state.user_id}. Model: {analysis.get('model_type')}")
            except Exception as e:
                st.error(f"Error saving to database: {str(e)}")
                logger.error(f"Database error: {e}", exc_info=True)
            
            st.info(f"Analysis complete! (Powered by: **{analysis.get('model_type', 'N/A')}**)")

            col1, col2 = st.columns(2)
            
            with col1:
                # ATS Score Card
                ats_score = int(analysis.get('ats_score', 0))
                color = '#4CAF50' if ats_score >= 80 else '#FFA500' if ats_score >= 60 else '#FF4444'
                status = 'Excellent' if ats_score >= 80 else 'Good' if ats_score >= 60 else 'Needs Improvement'
                st.markdown(f"""
                <div class="feature-card">
                    <h2>ATS Score</h2>
                    <div class="ats-score-container">
                        <div class="ats-score-circle" style="background: conic-gradient({color} 0% {ats_score}%, var(--bg-dark) {ats_score}% 100%);">
                            <div class="ats-score-inner-circle" style="color: {color};">{ats_score}</div>
                        </div>
                    </div>
                    <div class="ats-score-status" style="color: {color};">{status}</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Skills Match Card
                st.markdown("""
                <div class="feature-card">
                    <h2>Skills Match</h2>
                """, unsafe_allow_html=True)
                keyword_match = analysis.get('keyword_match', {})
                st.metric("Keyword Match", f"{int(keyword_match.get('score', 0))}%")
                if keyword_match.get('missing_skills'):
                    st.markdown("#### Missing Skills:")
                    for skill in keyword_match['missing_skills']:
                        st.markdown(f"- {skill}")
                st.markdown("</div>", unsafe_allow_html=True)

            with col2:
                # Format Score Card
                st.markdown("""
                <div class="feature-card">
                    <h2>Format & Section Analysis</h2>
                """, unsafe_allow_html=True)
                st.metric("Format Score", f"{int(analysis.get('format_score', 0))}%")
                st.metric("Section Score", f"{int(analysis.get('section_score', 0))}%")
                st.markdown("</div>", unsafe_allow_html=True)

            # Suggestions Card
            st.markdown("""
            <div class="feature-card">
                <h2>📋 Resume Improvement Suggestions</h2>
            """, unsafe_allow_html=True)
            suggestions = analysis.get('suggestions', [])
            if suggestions:
                for suggestion in suggestions:
                    st.markdown(f"<div class='suggestion-item'><i class='fas {suggestion.get('icon', 'fa-check-circle')}'></i> {suggestion.get('text')}</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='suggestion-item'><i class='fas fa-star'></i> Your resume looks great! No immediate suggestions.</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            # Course Recommendations and Videos... (rest of the page is the same)
            st.markdown("""
            <div class="feature-card">
                <h2>📚 Recommended Courses</h2>
            """, unsafe_allow_html=True)
            
            courses = get_courses_for_role(selected_role)
            if not courses:
                category = get_category_for_role(selected_role)
                courses = COURSES_BY_CATEGORY.get(category, {}).get(selected_role, [])
            
            cols_courses = st.columns(2)
            for i, course in enumerate(courses[:6]):
                with cols_courses[i % 2]:
                    st.markdown(f"""
                    <div class="course-card">
                        <h4>{course[0]}</h4>
                        <a href='{course[1]}' target='_blank' class="course-link">View Course</a>
                    </div>
                    """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("""
            <div class="feature-card">
                <h2>📺 Helpful Videos</h2>
            """, unsafe_allow_html=True)
            
            tab1, tab2 = st.tabs(["Resume Tips", "Interview Tips"])
            
            with tab1:
                for category, videos in RESUME_VIDEOS.items():
                    st.subheader(category)
                    cols_videos = st.columns(2)
                    for i, video in enumerate(videos):
                        with cols_videos[i % 2]:
                            st.video(video[1])
            
            with tab2:
                for category, videos in INTERVIEW_VIDEOS.items():
                    st.subheader(category)
                    cols_videos = st.columns(2)
                    for i, video in enumerate(videos):
                        with cols_videos[i % 2]:
                            st.video(video[1])
            
            st.markdown("</div>", unsafe_allow_html=True)