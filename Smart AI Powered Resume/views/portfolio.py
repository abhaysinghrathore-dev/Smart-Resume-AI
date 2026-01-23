import streamlit as st
from ui_components import page_header

def render_portfolio_page():
    """
    Renders a shareable web portfolio based on the user's project data.
    """

    # Get data from session state
    personal_info = st.session_state.form_data.get('personal_info', {})
    projects = st.session_state.form_data.get('projects', [])

    # --- Header Section ---
    page_header(personal_info.get('full_name', 'My Portfolio'), "A showcase of my projects and skills.")
    
    st.markdown("---")

    # --- Contact / Links Section ---
    col1, col2, col3 = st.columns(3)
    with col1:
        if personal_info.get('email'):
            st.markdown(f"📧 **Email:** [{personal_info['email']}](mailto:{personal_info['email']})")
    with col2:
        if personal_info.get('linkedin'):
            st.markdown(f"💼 **LinkedIn:** [{personal_info['linkedin']}]({personal_info['linkedin']})")
    with col3:
        if personal_info.get('github'):
            st.markdown(f"💻 **GitHub:** [{personal_info['github']}]({personal_info['github']})")
    
    st.markdown("---")


    # --- Projects Section ---
    if not projects:
        st.warning("You haven't added any projects yet. Go to the 'Resume Builder' to add your projects, and they will appear here.")
        return

    st.subheader("My Projects")

    # Display projects in a grid
    cols = st.columns(2)
    for i, proj in enumerate(projects):
        with cols[i % 2]:
            st.markdown(f"""
            <div class="feature-card">
                <h3>{proj.get('name', 'Unnamed Project')}</h3>
                <p><strong>Technologies:</strong> {proj.get('technologies', 'N/A')}</p>
                <p>{proj.get('description', 'No description available.')}</p>
                <a href='{proj.get('link', '#')}' target='_blank' class="course-link">View Project</a>
            </div>
            """, unsafe_allow_html=True)