import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime
import bcrypt
import logging
import os

logger = logging.getLogger(__name__)

def get_database_connection():
    """Create and return a PostgreSQL database connection"""
    try:
        conn = psycopg2.connect(
            host=os.environ.get("DB_HOST", "localhost"),
            dbname=os.environ.get("DB_NAME", "ai_resume"),
            user=os.environ.get("DB_USER", "postgres"),
            password=os.environ.get("DB_PASSWORD", "postgres"),
            port=os.environ.get("DB_PORT", "5432")
        )
        return conn
    except psycopg2.OperationalError as e:
        logger.error(f"Could not connect to PostgreSQL database: {e}", exc_info=True)
        raise

def init_database():
    """Initialize database tables for PostgreSQL"""
    conn = get_database_connection()
    cursor = conn.cursor()
    
    # Create resume_data table (normalized)
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS resume_data (
        id SERIAL PRIMARY KEY,
        name TEXT NOT NULL,
        email TEXT NOT NULL,
        phone TEXT NOT NULL,
        linkedin TEXT,
        github TEXT,
        portfolio TEXT,
        summary TEXT,
        target_role TEXT,
        target_category TEXT,
        skills TEXT,
        template TEXT,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
    )
    ''')

    # Create experiences table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS experiences (
        id SERIAL PRIMARY KEY,
        resume_id INTEGER,
        company TEXT,
        position TEXT,
        start_date TEXT,
        end_date TEXT,
        description TEXT,
        FOREIGN KEY (resume_id) REFERENCES resume_data (id) ON DELETE CASCADE
    )
    ''')

    # Create education table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS education (
        id SERIAL PRIMARY KEY,
        resume_id INTEGER,
        school TEXT,
        degree TEXT,
        field TEXT,
        graduation_date TEXT,
        gpa TEXT,
        FOREIGN KEY (resume_id) REFERENCES resume_data (id) ON DELETE CASCADE
    )
    ''')

    # Create projects table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS projects (
        id SERIAL PRIMARY KEY,
        resume_id INTEGER,
        name TEXT,
        technologies TEXT,
        description TEXT,
        link TEXT,
        FOREIGN KEY (resume_id) REFERENCES resume_data (id) ON DELETE CASCADE
    )
    ''')
    
    # Create resume_skills table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS resume_skills (
        id SERIAL PRIMARY KEY,
        resume_id INTEGER,
        skill_name TEXT NOT NULL,
        skill_category TEXT NOT NULL,
        proficiency_score REAL,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (resume_id) REFERENCES resume_data (id) ON DELETE CASCADE
    )
    ''')
    
    # Create resume_analysis table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS resume_analysis (
        id SERIAL PRIMARY KEY,
        resume_id INTEGER,
        ats_score REAL,
        keyword_match_score REAL,
        format_score REAL,
        section_score REAL,
        missing_skills TEXT,
        recommendations TEXT,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (resume_id) REFERENCES resume_data (id) ON DELETE CASCADE
    )
    ''')
    
    # Create admin_logs table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS admin_logs (
        id SERIAL PRIMARY KEY,
        admin_email TEXT NOT NULL,
        action TEXT NOT NULL,
        timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # Create admin table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS admin (
        id SERIAL PRIMARY KEY,
        email TEXT NOT NULL UNIQUE,
        password TEXT NOT NULL,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    conn.commit()
    cursor.close()
    conn.close()

def save_resume_data(data):
    """Save resume data to the normalized PostgreSQL schema."""
    conn = get_database_connection()
    cursor = conn.cursor()
    
    try:
        personal_info = data.get('personal_info', {})
        
        # Insert into main resume_data table and get the new ID
        cursor.execute('''
        INSERT INTO resume_data (
            name, email, phone, linkedin, github, portfolio,
            summary, target_role, target_category, skills, template
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id;
        ''', (
            personal_info.get('full_name', ''),
            personal_info.get('email', ''),
            personal_info.get('phone', ''),
            personal_info.get('linkedin', ''),
            personal_info.get('github', ''),
            personal_info.get('portfolio', ''),
            data.get('summary', ''),
            data.get('target_role', ''),
            data.get('target_category', ''),
            str(data.get('skills', [])),
            data.get('template', '')
        ))
        
        resume_id = cursor.fetchone()[0]

        # Insert into experiences table
        experiences = data.get('experience', [])
        for exp in experiences:
            cursor.execute('''
            INSERT INTO experiences (resume_id, company, position, start_date, end_date, description)
            VALUES (%s, %s, %s, %s, %s, %s)
            ''', (resume_id, exp.get('company'), exp.get('position'), exp.get('start_date'), exp.get('end_date'), exp.get('description')))

        # Insert into education table
        educations = data.get('education', [])
        for edu in educations:
            cursor.execute('''
            INSERT INTO education (resume_id, school, degree, field, graduation_date, gpa)
            VALUES (%s, %s, %s, %s, %s, %s)
            ''', (resume_id, edu.get('school'), edu.get('degree'), edu.get('field'), edu.get('graduation_date'), edu.get('gpa')))

        # Insert into projects table
        projects = data.get('projects', [])
        for proj in projects:
            cursor.execute('''
            INSERT INTO projects (resume_id, name, technologies, description, link)
            VALUES (%s, %s, %s, %s, %s)
            ''', (resume_id, proj.get('name'), proj.get('technologies'), proj.get('description'), proj.get('link')))

        conn.commit()
        return resume_id
    except Exception as e:
        logger.error(f"Error saving resume data to PostgreSQL: {str(e)}", exc_info=True)
        conn.rollback()
        return None
    finally:
        cursor.close()
        conn.close()

def save_analysis_data(resume_id, analysis):
    """Save resume analysis data to PostgreSQL"""
    conn = get_database_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
        INSERT INTO resume_analysis (
            resume_id, ats_score, keyword_match_score,
            format_score, section_score, missing_skills,
            recommendations
        ) VALUES (%s, %s, %s, %s, %s, %s, %s)
        ''', (
            resume_id,
            float(analysis.get('ats_score', 0)),
            float(analysis.get('keyword_match_score', 0)),
            float(analysis.get('format_score', 0)),
            float(analysis.get('section_score', 0)),
            analysis.get('missing_skills', ''),
            analysis.get('recommendations', '')
        ))
        
        conn.commit()
    except Exception as e:
        logger.error(f"Error saving analysis data to PostgreSQL: {str(e)}", exc_info=True)
        conn.rollback()
    finally:
        cursor.close()
        conn.close()

def get_resume_stats():
    """Get statistics about resumes from PostgreSQL"""
    conn = get_database_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    try:
        cursor.execute('SELECT COUNT(*) as total_resumes FROM resume_data')
        total_resumes = cursor.fetchone()['total_resumes']
        
        cursor.execute('SELECT AVG(ats_score) as avg_score FROM resume_analysis')
        avg_ats_score = cursor.fetchone()['avg_score'] or 0
        
        cursor.execute('SELECT name, target_role, created_at FROM resume_data ORDER BY created_at DESC LIMIT 5')
        recent_activity = cursor.fetchall()
        
        return {
            'total_resumes': total_resumes,
            'avg_ats_score': round(float(avg_ats_score), 2),
            'recent_activity': recent_activity
        }
    except Exception as e:
        logger.error(f"Error getting resume stats from PostgreSQL: {str(e)}", exc_info=True)
        return None
    finally:
        cursor.close()
        conn.close()

def log_admin_action(admin_email, action):
    """Log admin login/logout actions to PostgreSQL"""
    conn = get_database_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
        INSERT INTO admin_logs (admin_email, action)
        VALUES (%s, %s)
        ''', (admin_email, action))
        conn.commit()
    except Exception as e:
        logger.error(f"Error logging admin action to PostgreSQL: {str(e)}", exc_info=True)
    finally:
        cursor.close()
        conn.close()

def get_admin_logs():
    """Get all admin login/logout logs from PostgreSQL"""
    conn = get_database_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    try:
        cursor.execute('SELECT admin_email, action, timestamp FROM admin_logs ORDER BY timestamp DESC')
        return cursor.fetchall()
    except Exception as e:
        logger.error(f"Error getting admin logs from PostgreSQL: {str(e)}", exc_info=True)
        return []
    finally:
        cursor.close()
        conn.close()

def get_all_resume_data():
    """Get all resume data for admin dashboard from PostgreSQL"""
    conn = get_database_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    try:
        cursor.execute('''
        SELECT 
            r.id, r.name, r.email, r.phone, r.linkedin, r.github, r.portfolio,
            r.target_role, r.target_category, r.created_at,
            a.ats_score, a.keyword_match_score, a.format_score, a.section_score
        FROM resume_data r
        LEFT JOIN resume_analysis a ON r.id = a.resume_id
        ORDER BY r.created_at DESC
        ''')
        return cursor.fetchall()
    except Exception as e:
        logger.error(f"Error getting resume data from PostgreSQL: {str(e)}", exc_info=True)
        return []
    finally:
        cursor.close()
        conn.close()

def verify_admin(email, password):
    """Verify admin credentials using bcrypt from PostgreSQL."""
    conn = get_database_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    try:
        cursor.execute('SELECT password FROM admin WHERE email = %s', (email,))
        result = cursor.fetchone()
        if result:
            hashed_password = result['password'].encode('utf-8')
            return bcrypt.checkpw(password.encode('utf-8'), hashed_password)
        return False
    except Exception as e:
        logger.error(f"Error verifying admin from PostgreSQL: {str(e)}", exc_info=True)
        return False
    finally:
        cursor.close()
        conn.close()

def add_admin(email, password):
    """Add a new admin with a hashed password to PostgreSQL."""
    conn = get_database_connection()
    cursor = conn.cursor()
    
    try:
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        cursor.execute('INSERT INTO admin (email, password) VALUES (%s, %s)', (email, hashed_password.decode('utf-8')))
        conn.commit()
        return True
    except psycopg2.IntegrityError: # Handle cases where the email already exists
        logger.warning(f"Attempted to add existing admin: {email}")
        conn.rollback()
        return False
    except Exception as e:
        logger.error(f"Error adding admin to PostgreSQL: {str(e)}", exc_info=True)
        conn.rollback()
        return False
    finally:
        cursor.close()
        conn.close()

