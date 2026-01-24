import streamlit as st
import traceback
from config.database import save_resume_data
from ui_components import (
    render_personal_info_form, render_summary_form,
    render_experience_form, render_projects_form, render_education_form, render_skills_form
)
from utils.logger import setup_logger

logger = setup_logger(__name__)

def render_builder(builder):
    st.title("Resume Builder 📝")
    st.write("Create your professional resume")
    
    # Template selection
    template_options = ["Modern", "Professional", "Minimal", "Creative"]
    selected_template = st.selectbox("Select Resume Template", template_options)
    st.success(f"🎨 Currently using: {selected_template} Template")

    # Personal Information
    st.session_state.form_data['personal_info'] = render_personal_info_form(st.session_state.form_data['personal_info'])

    # Professional Summary
    st.session_state.form_data['summary'] = render_summary_form(st.session_state.form_data.get('summary', ''))
    
    # Experience Section
    st.session_state.form_data['experiences'] = render_experience_form(st.session_state.form_data['experiences'])
    
    # Projects Section
    st.session_state.form_data['projects'] = render_projects_form(st.session_state.form_data['projects'])
    
    # Education Section
    st.session_state.form_data['education'] = render_education_form(st.session_state.form_data['education'])
    
    # Skills Section
    st.session_state.form_data['skills_categories'] = render_skills_form(st.session_state.form_data['skills_categories'])
    
    # Update form data in session state
    st.session_state.form_data.update({
        'summary': st.session_state.form_data['summary']
    })
    # Generate Resume button
    if st.button("Generate Resume 📄", type="primary"):
        logger.info("Validating form data...")
        logger.debug(f"Session state form data: {st.session_state.form_data}")
        
        # Get the current values from form
        current_name = st.session_state.form_data['personal_info']['full_name'].strip()
        current_email = st.session_state.email_input if 'email_input' in st.session_state else ''
        
        logger.info(f"Generating resume for: {current_name}")
        
        # Validate required fields
        if not current_name:
            st.error("⚠️ Please enter your full name.")
            return
        
        if not current_email:
            st.error("⚠️ Please enter your email address.")
            return
            
        # Update email in form data one final time
        st.session_state.form_data['personal_info']['email'] = current_email
        
        try:
            logger.info("Preparing resume data...")
            # Prepare resume data with current form values
            resume_data = {
                "personal_info": st.session_state.form_data['personal_info'],
                "summary": st.session_state.form_data.get('summary', '').strip(),
                "experience": st.session_state.form_data.get('experiences', []),
                "education": st.session_state.form_data.get('education', []),
                "projects": st.session_state.form_data.get('projects', []),
                "skills": st.session_state.form_data.get('skills_categories', {
                    'technical': [],
                    'soft': [],
                    'languages': [],
                    'tools': []
                }),
                "template": selected_template
            }
            
            logger.debug(f"Resume data prepared: {resume_data}")
            
            try:
                # Generate resume
                resume_buffer = builder.generate_resume(resume_data)
                if resume_buffer:
                    try:
                        # Save resume data to database
                        save_resume_data(resume_data)
                        logger.info("Resume generated and saved successfully")
                        
                        # Offer the resume for download
                        st.success("✅ Resume generated successfully!")
                        st.download_button(
                            label="Download Resume 📥",
                            data=resume_buffer,
                            file_name=f"{current_name.replace(' ', '_')}_resume.docx",
                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                        )
                    except Exception as db_error:
                        logger.warning(f"Failed to save to database: {str(db_error)}")
                        # Still allow download even if database save fails
                        st.warning("⚠️ Resume generated but couldn't be saved to database")
                        st.download_button(
                            label="Download Resume 📥",
                            data=resume_buffer,
                            file_name=f"{current_name.replace(' ', '_')}_resume.docx",
                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                        )
                else:
                    st.error("❌ Failed to generate resume. Please try again.")
                    logger.error("Resume buffer was None")
            except Exception as gen_error:
                logger.error(f"Error during resume generation: {str(gen_error)}", exc_info=True)
                st.error(f"❌ Error generating resume: {str(gen_error)}")
                    
        except Exception as e:
            logger.error(f"Error preparing resume data: {str(e)}", exc_info=True)
            st.error(f"❌ Error preparing resume data: {str(e)}")
