import streamlit as st
from streamlit_lottie import st_lottie
import html
import re
from utils.generative_ai import generate_summary, generate_experience_description, generate_project_description
import time

def clean_page_name(page_name):
    """Helper to create consistent page keys from display names by removing emojis"""
    # Remove non-word characters (except spaces), strip, lower, and replace spaces with underscores
    text = re.sub(r'[^\w\s]', '', page_name)
    return text.strip().lower().replace(" ", "_")

def page_header(title, subtitle=None):
    """Render a consistent page header with gradient background"""
    st.markdown(
        f'''
        <div class="page-header">
            <h1 class="header-title">{title}</h1>
            {f'<p class="header-subtitle">{subtitle}</p>' if subtitle else ''}
        </div>
        ''',
        unsafe_allow_html=True
    )

def hero_section(title, subtitle=None, description=None):
    """Render a modern hero section with gradient background and animations"""
    # If description is provided but subtitle is not, use description as subtitle
    if description and not subtitle:
        subtitle = description
        description = None
    
    st.markdown(
        f'''
        <div class="page-header hero-header">
            <h1 class="header-title">{title}</h1>
            {f'<div class="header-subtitle">{subtitle}</div>' if subtitle else ''}
            {f'<p class="header-description">{description}</p>' if description else ''}
        </div>
        ''',
        unsafe_allow_html=True
    )

def feature_card(icon, title, description):
    """Render a modern feature card with hover effects"""
    st.markdown(f"""
        <div class="card feature-card">
            <div class="feature-icon icon-pulse">
                <i class="{icon}"></i>
            </div>
            <h3>{title}</h3>
            <p>{description}</p>
        </div>
    """, unsafe_allow_html=True)

def profile_section(content, image_path=None, social_links=None):
    """Render a modern about section with profile image and social links"""
    st.markdown("""
        <div class="glass-card about-section">
            <div class="profile-section">
    """, unsafe_allow_html=True)
    
    # Profile Image
    if image_path:
        st.image(image_path, use_column_width=False, width=200)
    
    # Image Upload
    uploaded_file = st.file_uploader("Upload profile picture", type=['png', 'jpg', 'jpeg'])
    if uploaded_file is not None:
        st.image(uploaded_file, use_column_width=False, width=200)
    
    # Social Links
    if social_links:
        st.markdown('<div class="social-links">', unsafe_allow_html=True)
        for platform, url in social_links.items():
            st.markdown(f'<a href="{url}" target="_blank" class="social-link"><i class="fab fa-{platform.lower()}"></i></a>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # About Content
    st.markdown(f"""
            </div>
            <div class="about-content">{content}</div>
        </div>
    """, unsafe_allow_html=True)

def metric_card(label, value, delta=None, icon=None):
    """Render a modern metric card with animations"""
    icon_html = f'<i class="{icon}"></i>' if icon else ''
    delta_html = f'<div class="metric-delta">{delta}</div>' if delta else ''
    
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-header">
                {icon_html}
                <div class="metric-label">{label}</div>
            </div>
            <div class="metric-value">{value}</div>
            {delta_html}
        </div>
    """, unsafe_allow_html=True)

def template_card(title, description, image_url=None):
    """Render a modern template card with glassmorphism effect"""
    image_html = f'<img src="{image_url}" class="template-image" />' if image_url else ''
    
    st.markdown(f"""
        <div class="glass-card template-card">
            {image_html}
            <h3>{title}</h3>
            <p>{description}</p>
            <div class="card-overlay"></div>
        </div>
    """, unsafe_allow_html=True)

def feedback_card(name, feedback, rating):
    """Render a modern feedback card with rating stars"""
    stars = "⭐" * int(rating)
    
    st.markdown(f"""
        <div class="card feedback-card">
            <div class="feedback-header">
                <div class="feedback-name">{name}</div>
                <div class="feedback-rating">{stars}</div>
            </div>
            <p class="feedback-text">{feedback}</p>
        </div>
    """, unsafe_allow_html=True)

def loading_spinner(message="Loading..."):
    """Show a modern loading spinner with message"""
    st.markdown(f"""
        <div class="loading-container">
            <div class="loading-spinner"></div>
            <p class="loading-message">{message}</p>
        </div>
    """, unsafe_allow_html=True)

def progress_bar(value, max_value, label=None):
    """Render a modern animated progress bar"""
    percentage = (value / max_value) * 100
    label_html = f'<div class="progress-label">{label}</div>' if label else ''
    
    st.markdown(f"""
        <div class="progress-container">
            {label_html}
            <div class="progress-bar">
                <div class="progress-fill" style="width: {percentage}%"></div>
            </div>
            <div class="progress-value">{percentage:.1f}%</div>
        </div>
    """, unsafe_allow_html=True)

def tooltip(content, tooltip_text):
    """Render content with a modern tooltip"""
    st.markdown(f"""
        <div class="tooltip" data-tooltip="{tooltip_text}">
            {content}
        </div>
    """, unsafe_allow_html=True)

def data_table(data, headers):
    """Render a modern data table with hover effects"""
    header_row = "".join([f"<th>{header}</th>" for header in headers])
    rows = ""
    for row in data:
        cells = "".join([f"<td>{cell}</td>" for cell in row])
        rows += f"<tr>{cells}</tr>"
    
    st.markdown(f"""
        <div class="table-container">
            <table class="modern-table">
                <thead>
                    <tr>{header_row}</tr>
                </thead>
                <tbody>
                    {rows}
                </tbody>
            </table>
        </div>
    """, unsafe_allow_html=True)

def grid_layout(*elements):
    """Create a responsive grid layout"""
    st.markdown("""
        <div class="grid">
            {}
        </div>
    """.format("".join(elements)), unsafe_allow_html=True)

def alert(message, type="info"):
    """Display a modern alert message"""
    alert_types = {
        "info": ("ℹ️", "var(--accent-color)"),
        "success": ("✅", "var(--success-color)"),
        "warning": ("⚠️", "var(--warning-color)"),
        "error": ("❌", "var(--error-color)")
    }
    icon, color = alert_types.get(type, alert_types["info"])
    
    st.markdown(f"""
        <div class="alert alert-{type}">
            <span class="alert-icon">{icon}</span>
            <span class="alert-message">{message}</span>
        </div>
    """, unsafe_allow_html=True)

def about_section(title, description, team_members=None):
    st.markdown(f"""
        <div class="about-section">
            <h2>{title}</h2>
            <p class="about-description">{description}</p>
            {generate_team_section(team_members) if team_members else ''}
        </div>
    """, unsafe_allow_html=True)

def generate_team_section(team_members):
    if not team_members:
        return ""
    
    team_html = '<div class="team-section">'
    for member in team_members:
        team_html += f"""
            <div class="team-member">
                <img src="{member['image']}" alt="{member['name']}">
                <h3>{member['name']}</h3>
                <p>{member['role']}</p>
            </div>
        """
    team_html += '</div>'
    return team_html

def render_feedback(feedback_data):
    """Render feedback with modern styling"""
    if not feedback_data:
        return
    
    feedback_html = """
    <div class="feedback-section">
        <h3 class="feedback-header">Resume Analysis Feedback</h3>
        <div class="feedback-content">
    """
    
    for category, items in feedback_data.items():
        if items:  # Only show categories with feedback
            for item in items:
                feedback_html += f"""
                <div class="feedback-item">
                    <div class="feedback-category">{html.escape(category)}</div>
                    <div class="feedback-description">{html.escape(item)}</div>
                </div>
                """
    
    feedback_html += """
        </div>
    </div>
    """
    
    st.markdown(feedback_html, unsafe_allow_html=True)

def render_feedback_form():
    st.markdown("""
        <div class="feedback-header">
            <h1>Your Voice Matters! 🗣️</h1>
            <p>Help us improve Smart Resume AI with your valuable feedback and suggestions.</p>
        </div>
        <div class="feedback-form-container">
    """, unsafe_allow_html=True)

    # Form content will be added here by the feedback manager
    st.markdown("</div>", unsafe_allow_html=True)

def render_feedback_overview():
    st.markdown("""
        <div class="feedback-section">
            <h2 class="feedback-overview-title">Feedback Overview 📊</h2>
        </div>
    """, unsafe_allow_html=True)

def render_analytics_section(resume_uploaded=False, metrics=None):
    """Render the analytics section of the dashboard"""
    if not metrics:
        metrics = {
            'views': 0,
            'downloads': 0,
            'score': 'N/A'
        }
    
    # Views Card
    st.markdown("""
        <div class="analytics-card">
            <div class="analytics-icon">
                <i class='fas fa-eye'></i>
            </div>
            <h2 class="analytics-title">Resume Views</h2>
            <p class="analytics-value">{}</p>
        </div>
    """.format(metrics['views']), unsafe_allow_html=True)
    
    # Downloads Card
    st.markdown("""
        <div class="analytics-card">
            <div class="analytics-icon">
                <i class='fas fa-download'></i>
            </div>
            <h2 class="analytics-title">Downloads</h2>
            <p class="analytics-value">{}</p>
        </div>
    """.format(metrics['downloads']), unsafe_allow_html=True)
    
    # Profile Score Card
    st.markdown("""
        <div class="analytics-card">
            <div class="analytics-icon">
                <i class='fas fa-chart-line'></i>
            </div>
            <h2 class="analytics-title">Profile Score</h2>
            <p class="analytics-value">{}</p>
        </div>
    """.format(metrics['score']), unsafe_allow_html=True)

def render_activity_section(resume_uploaded=False):
    """Render the recent activity section"""
    st.markdown("""
        <div class="activity-section">
            <h2 class="activity-title">
                <i class='fas fa-history activity-icon'></i> Recent Activity
            </h2>
    """, unsafe_allow_html=True)
    
    if resume_uploaded:
        st.markdown("""
            <div class="activity-content">
                <p class="activity-item">• Resume uploaded and analyzed</p>
                <p class="activity-item">• Generated optimization suggestions</p>
                <p class="activity-item">• Updated profile score</p>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <div class="activity-empty-state">
                <i class='fas fa-upload activity-empty-icon'></i>
                <p class="activity-empty-message">Upload your resume to see activity</p>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

def render_suggestions_section(resume_uploaded=False):
    """Render the suggestions section"""
    st.markdown("""
        <div class="suggestions-section">
            <h2 class="suggestions-title">
                <i class='fas fa-lightbulb suggestions-icon'></i> Suggestions
            </h2>
    """, unsafe_allow_html=True)
    
    if resume_uploaded:
        st.markdown("""
            <div class="suggestions-content">
                <p class="suggestions-item">• Add more quantifiable achievements</p>
                <p class="suggestions-item">• Include relevant keywords</p>
                <p class="suggestions-item">• Optimize formatting</p>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <div class="suggestions-empty-state">
                <i class='fas fa-file-alt suggestions-empty-icon'></i>
                <p class="suggestions-empty-message">Upload your resume to get suggestions</p>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

def render_personal_info_form(personal_info):
    st.subheader("Personal Information")
    col1, col2 = st.columns(2)
    with col1:
        personal_info['full_name'] = st.text_input("Full Name", value=personal_info.get('full_name', ''))
        personal_info['email'] = st.text_input("Email", value=personal_info.get('email', ''), key="email_input")
        personal_info['phone'] = st.text_input("Phone", value=personal_info.get('phone', ''))
    with col2:
        personal_info['location'] = st.text_input("Location", value=personal_info.get('location', ''))
        personal_info['linkedin'] = st.text_input("LinkedIn URL", value=personal_info.get('linkedin', ''))
        personal_info['github'] = st.text_input("GitHub Profile", value=personal_info.get('github', ''))
    return personal_info

def render_summary_form(summary):
    st.subheader("Professional Summary")
    
    # Text area for the summary
    summary_text = st.text_area("Professional Summary", value=summary, height=150, 
                                help="Write a brief summary or click 'Generate with AI' to create one.")
    
    # AI generation button
    if st.button("✨ Generate with AI", key="generate_summary"):
        with st.spinner("🧠 AI is crafting your summary..."):
            # A real app might pass keywords from skills or job descriptions
            generated_summary = generate_summary(summary_text, keywords=["Software Engineer"])
            st.session_state.form_data['summary'] = generated_summary
            st.rerun()

    return summary_text

def render_experience_form(experiences):
    st.subheader("Work Experience")
    if st.button("Add Experience"):
        experiences.append({
            'company': '',
            'position': '',
            'start_date': '',
            'end_date': '',
            'description': '',
            'responsibilities': [],
            'achievements': []
        })
        st.session_state.form_data['experiences'] = experiences # Update session state immediately
        st.rerun()

    for idx, exp in enumerate(experiences):
        with st.expander(f"Experience {idx + 1}", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                exp['company'] = st.text_input("Company Name", key=f"company_{idx}", value=exp.get('company', ''))
                exp['position'] = st.text_input("Position", key=f"position_{idx}", value=exp.get('position', ''))
            with col2:
                exp['start_date'] = st.text_input("Start Date", key=f"start_date_{idx}", value=exp.get('start_date', ''))
                exp['end_date'] = st.text_input("End Date", key=f"end_date_{idx}", value=exp.get('end_date', ''))
            
            st.markdown("##### Key Responsibilities & Achievements (as bullet points)")
            # Combine responsibilities and achievements for AI processing
            bullet_points_text = st.text_area("Enter responsibilities and achievements (one per line)", 
                                   key=f"bullets_{idx}",
                                   value='\n'.join(exp.get('responsibilities', []) + exp.get('achievements', [])),
                                   height=120,
                                   help="List your main responsibilities and achievements as bullet points.")
            
            bullet_points = [p.strip() for p in bullet_points_text.split('\n') if p.strip()]

            if st.button("✨ Improve with AI", key=f"improve_exp_{idx}"):
                with st.spinner("🤖 Rewriting your experience..."):
                    generated_desc = generate_experience_description(exp.get('company'), exp.get('position'), bullet_points)
                    st.session_state.form_data['experiences'][idx]['description'] = generated_desc
                    st.rerun()

            st.markdown("##### Generated Role Overview")
            exp['description'] = st.text_area("Role Overview", key=f"desc_{idx}", 
                                            value=exp.get('description', ''),
                                            height=150,
                                            help="This will be auto-generated by the AI or can be written manually.")
            
            if st.button("Remove Experience", key=f"remove_exp_{idx}"):
                experiences.pop(idx)
                st.session_state.form_data['experiences'] = experiences # Update session state immediately
                st.rerun()
    return experiences

def render_projects_form(projects):
    st.subheader("Projects")
    if st.button("Add Project"):
        projects.append({
            'name': '',
            'technologies': '',
            'description': '',
            'responsibilities': [],
            'achievements': [],
            'link': ''
        })
        st.session_state.form_data['projects'] = projects # Update session state immediately
        st.rerun()

    for idx, proj in enumerate(projects):
        with st.expander(f"Project {idx + 1}", expanded=True):
            proj['name'] = st.text_input("Project Name", key=f"proj_name_{idx}", value=proj.get('name', ''))
            proj['technologies'] = st.text_input("Technologies Used", key=f"proj_tech_{idx}", 
                                               value=proj.get('technologies', ''),
                                               help="List the main technologies, frameworks, and tools used")
            
            st.markdown("##### Key Responsibilities & Achievements (as bullet points)")
            bullet_points_text = st.text_area("Enter responsibilities and achievements (one per line)", 
                                        key=f"proj_bullets_{idx}",
                                        value='\n'.join(proj.get('responsibilities', []) + proj.get('achievements', [])),
                                        height=120,
                                        help="List your main responsibilities and achievements in the project.")
            
            bullet_points = [p.strip() for p in bullet_points_text.split('\n') if p.strip()]

            if st.button("✨ Improve with AI", key=f"improve_proj_{idx}"):
                with st.spinner("🤖 Rewriting your project description..."):
                    generated_desc = generate_project_description(proj.get('name'), proj.get('technologies'), bullet_points)
                    st.session_state.form_data['projects'][idx]['description'] = generated_desc
                    st.rerun()

            st.markdown("##### Generated Project Overview")
            proj['description'] = st.text_area("Project Overview", key=f"proj_desc_{idx}", 
                                             value=proj.get('description', ''),
                                             height=150,
                                             help="This will be auto-generated by the AI or can be written manually.")
            
            proj['link'] = st.text_input("Project Link (optional)", key=f"proj_link_{idx}", 
                                       value=proj.get('link', ''),
                                       help="Link to the project repository, demo, or documentation")
            
            if st.button("Remove Project", key=f"remove_proj_{idx}"):
                projects.pop(idx)
                st.session_state.form_data['projects'] = projects # Update session state immediately
                st.rerun()
    return projects

def render_education_form(education):
    st.subheader("Education")
    if st.button("Add Education"):
        education.append({
            'school': '',
            'degree': '',
            'field': '',
            'graduation_date': '',
            'gpa': '',
            'achievements': []
        })
        st.session_state.form_data['education'] = education # Update session state immediately
        st.rerun()

    for idx, edu in enumerate(education):
        with st.expander(f"Education {idx + 1}", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                edu['school'] = st.text_input("School/University", key=f"school_{idx}", value=edu.get('school', ''))
                edu['degree'] = st.text_input("Degree", key=f"degree_{idx}", value=edu.get('degree', ''))
            with col2:
                edu['field'] = st.text_input("Field of Study", key=f"field_{idx}", value=edu.get('field', ''))
                edu['graduation_date'] = st.text_input("Graduation Date", key=f"grad_date_{idx}", 
                                                     value=edu.get('graduation_date', ''))
            
            edu['gpa'] = st.text_input("GPA (optional)", key=f"gpa_{idx}", value=edu.get('gpa', ''))
            
            st.markdown("##### Achievements & Activities")
            edu_achv_text = st.text_area("Enter achievements (one per line)", 
                                       key=f"edu_achv_{idx}",
                                       value='\n'.join(edu.get('achievements', [])),
                                       height=100,
                                       help="List academic achievements, relevant coursework, or activities")
            edu['achievements'] = [a.strip() for a in edu_achv_text.split('\n') if a.strip()]
            
            if st.button("Remove Education", key=f"remove_edu_{idx}"):
                education.pop(idx)
                st.session_state.form_data['education'] = education # Update session state immediately
                st.rerun()
    return education

def render_skills_form(skills_categories):
    st.subheader("Skills")
    col1, col2 = st.columns(2)
    with col1:
        tech_skills = st.text_area("Technical Skills (one per line)", 
                                 value='\n'.join(skills_categories.get('technical', [])),
                                 height=150,
                                 help="Programming languages, frameworks, databases, etc.")
        skills_categories['technical'] = [s.strip() for s in tech_skills.split('\n') if s.strip()]
        
        soft_skills = st.text_area("Soft Skills (one per line)", 
                                 value='\n'.join(skills_categories.get('soft', [])),
                                 height=150,
                                 help="Leadership, communication, problem-solving, etc.")
        skills_categories['soft'] = [s.strip() for s in soft_skills.split('\n') if s.strip()]
    
    with col2:
        languages = st.text_area("Languages (one per line)", 
                               value='\n'.join(skills_categories.get('languages', [])),
                               height=150,
                               help="Programming or human languages with proficiency level")
        skills_categories['languages'] = [l.strip() for l in languages.split('\n') if l.strip()]
        
        tools = st.text_area("Tools & Technologies (one per line)", 
                           value='\n'.join(skills_categories.get('tools', [])),
                           height=150,
                           help="Development tools, software, platforms, etc.")
        skills_categories['tools'] = [t.strip() for t in tools.split('\n') if t.strip()]
    return skills_categories

def render_sidebar(pages, load_lottie_url, is_admin, current_admin_email, verify_admin, log_admin_action):
    with st.sidebar:
        # Custom CSS for sidebar styling
        st.markdown("""
<style>
/* Sidebar container */
section[data-testid="stSidebar"] {
    background-color: #2f343a;
}

/* Navigation buttons in the sidebar */
section[data-testid="stSidebar"] .stButton button {
    padding: 14px 18px;
    margin-bottom: 10px;
    border-radius: 10px;
    color: white;
    font-weight: 500;
    cursor: pointer;
    transition: color 0.25s ease, background-color 0.25s ease;
    
    /* Make the button background transparent to see the sidebar color */
    background-color: transparent;
    border: none;
    width: 100%;
    text-align: left;
}

/* Hover effect for navigation buttons */
section[data-testid="stSidebar"] .stButton button:hover {
    color: #1e90ff; /* blue */
    background-color: rgba(255, 255, 255, 0.06);
}

/* Make sure the text inside the button is white */
section[data-testid="stSidebar"] .stButton button p {
    color: white;
}

/* Hover effect for the text inside the button */
section[data-testid="stSidebar"] .stButton button:hover p {
    color: #1e90ff; /* blue */
}

</style>
        """, unsafe_allow_html=True)

        st_lottie(load_lottie_url("https://assets5.lottiefiles.com/packages/lf20_xyadoh9h.json"), height=180, key="sidebar_animation")
        st.markdown('<div class="sidebar-header">Smart Resume AI</div>', unsafe_allow_html=True)
        st.markdown("---")
        
        # Navigation buttons
        st.markdown("### 🧭 Menu")
        for page_name in pages.keys():
            if st.button(page_name, width='stretch', key=f"nav_btn_{page_name}"):
                st.session_state.page = clean_page_name(page_name)
                st.rerun()

        # Add some space before admin login
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("---")
        
        # Admin Login/Logout section at bottom
        if is_admin:
            st.success(f"👤 {current_admin_email}")
            if st.button("🚪 Logout", key="logout_button", type="primary"):
                try:
                    log_admin_action(current_admin_email, "logout")
                    st.session_state.is_admin = False
                    st.session_state.current_admin_email = None
                    st.success("Logged out")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        else:
            with st.expander("🔐 Admin Access"):
                admin_email_input = st.text_input("Email", key="admin_email_input")
                admin_password = st.text_input("Password", type="password", key="admin_password_input")
                if st.button("Login", key="login_button", type="primary"):
                        try:
                            if verify_admin(admin_email_input, admin_password):
                                st.session_state.is_admin = True
                                st.session_state.current_admin_email = admin_email_input
                                log_admin_action(admin_email_input, "login")
                                st.success("Welcome back!")
                                st.rerun()
                            else:
                                st.error("Invalid credentials")
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
                            
        st.markdown('<div class="sidebar-footer">© 2026 Smart Resume AI<br>v1.0.0</div>', unsafe_allow_html=True)