"""
This module simulates a Generative AI model for creating resume and cover letter content.
In a real-world application, these functions would make API calls to an LLM (e.g., Gemini, OpenAI).
For this simulation, they return realistic, hardcoded text to allow for UI and logic development.
"""

import time

def generate_summary(existing_summary, keywords=None):
    """
    Simulates rewriting a professional summary to be more impactful.
    """
    time.sleep(2) # Simulate network latency
    
    base_text = (
        "Highly motivated and results-oriented professional with a proven track record of success. "
        "Seeking to leverage my skills and experience in a challenging new role. "
    )
    
    if "Data Scientist" in keywords:
        return (
            "Results-driven Data Scientist with 5+ years of experience in machine learning, statistical analysis, "
            "and predictive modeling. Proficient in Python, R, and SQL, with a deep understanding of data "
            "structures and algorithms. Passionate about turning large datasets into actionable insights."
        )
    if "Software Engineer" in keywords:
        return (
            "Innovative Software Engineer with a passion for developing scalable and efficient applications. "
            "Experienced in full-stack development with expertise in JavaScript (React, Node.js) and Python (Django). "
            "A strong collaborator with a commitment to writing clean, maintainable code."
        )
        
    return base_text + "Adept at problem-solving and collaborating with cross-functional teams to achieve business objectives."

def generate_experience_description(company, position, bullet_points):
    """
    Simulates rewriting experience bullet points into a professional description.
    """
    time.sleep(2) # Simulate network latency

    description = f"As a {position} at {company}, I was responsible for a variety of key initiatives. My primary contributions include:\n\n"
    
    # Simulate rewriting bullet points
    rewritten_points = []
    for point in bullet_points:
        if "led" in point.lower():
            rewritten_points.append(f"- Spearheaded the initiative to {point.replace('Led', '').strip()}, resulting in a 20% increase in efficiency.")
        elif "developed" in point.lower():
            rewritten_points.append(f"- Engineered and launched a new {point.replace('Developed', '').strip()} feature, enhancing user engagement by 15%.")
        else:
            rewritten_points.append(f"- Played a key role in {point.strip()}, which contributed to the project's overall success.")

    return description + "\n".join(rewritten_points)

def generate_project_description(project_name, technologies, bullet_points):
    """
    Simulates rewriting project bullet points into a professional description.
    """
    time.sleep(2) # Simulate network latency

    description = f"For the '{project_name}' project, I utilized a modern tech stack including {technologies} to deliver a robust solution. Key contributions to this project include:\n\n"
    
    # Simulate rewriting bullet points
    rewritten_points = []
    for point in bullet_points:
        if "built" in point.lower() or "created" in point.lower():
            rewritten_points.append(f"- Architected and developed a core component for {point.replace('Built', '').replace('Created', '').strip()}, which was critical for the project's success.")
        elif "designed" in point.lower():
            rewritten_points.append(f"- Designed a new UI/UX flow for the {point.replace('Designed', '').strip()} feature, improving user satisfaction scores by 10%.")
        else:
            rewritten_points.append(f"- Implemented the {point.strip()} functionality, ensuring high performance and scalability.")

    return description + "\n".join(rewritten_points)


def generate_cover_letter(resume_data, job_description, company_name):
    """
    Simulates generating a full, tailored cover letter.
    """
    time.sleep(3) # Simulate network latency

    user_name = resume_data.get('personal_info', {}).get('full_name', 'Your Name')
    
    return f"""
Dear Hiring Manager,

I am writing to express my enthusiastic interest in the position advertised on [Platform where you saw the ad], which I discovered through my deep respect for {company_name}'s innovative work in the industry. With my background in [Your Field, e.g., software engineering] and a proven ability to [mention a key skill from your resume], I am confident that I possess the skills and experience necessary to excel in this role.

The enclosed resume details my experience, including my recent role at [Previous Company], where I was responsible for [mention a key responsibility]. One of my proudest achievements was [mention an achievement], which demonstrates my commitment to delivering results. This experience has equipped me with a unique perspective on how to tackle the challenges outlined in your job description.

I am particularly drawn to {company_name}'s commitment to [mention a company value or project you admire]. My goal is to contribute to a team that is pushing the boundaries of what is possible, and I believe my skills are a perfect match for your needs.

Thank you for considering my application. I have attached my resume for your review and welcome the opportunity to discuss how I can be a valuable asset to your team.

Sincerely,
{user_name}
"""
