# Smart Resume AI 🚀

A next-generation, AI-powered platform designed to help you create, analyze, and optimize your resume, cover letter, and professional portfolio for the modern job market.

![Project Banner](https://placehold.co/1200x400/2f343a/ffffff?text=Smart+Resume+AI)

## 💡 What is Smart Resume AI?

In today's competitive job market, a generic resume isn't enough. Smart Resume AI tackles this problem head-on. It's not just a template filler; it's an intelligent assistant that helps you craft a compelling narrative of your skills and experiences. From analyzing your existing resume against job descriptions to generating impactful, professional content, this application is your personal career co-pilot.

This tool helps you stand out by:
- **Optimizing** your resume for Applicant Tracking Systems (ATS).
- **Generating** professional, well-written content for your resume and cover letters.
- **Showcasing** your projects in a clean, shareable portfolio.

---

## ✨ Key Features

- **🧠 ML-Powered Resume Analysis:**
  - Upload your resume to get an instant ATS score powered by a fine-tuned `BERT/TensorFlow` model.
  - Get detailed feedback on keyword matching, format quality, and section completeness.
  - Receive actionable, AI-generated suggestions for improvement.

- **✍️ Generative Content Creation:**
  - **AI-Powered Summary:** Generate a powerful, professional summary with one click.
  - **AI-Enhanced Experience/Projects:** Transform your bullet points into impactful, descriptive paragraphs that showcase your achievements.
  - **Cover Letter Generator:** Create a tailored cover letter based on your resume data and a specific job description.

- **🌐 Portfolio Viewer:**
  - Automatically assembles your project data into a clean, shareable web portfolio page.

- **📊 Analytics Dashboard:**
  - An admin-only dashboard to view application-wide statistics, such as the number of resumes analyzed, average scores, and popular skills.

- **🔒 Secure & Scalable Architecture:**
  - Built on a robust **PostgreSQL** database with a normalized schema.
  - Features secure admin authentication with password hashing (`bcrypt`).
  - Containerized with **Docker** for consistent, reproducible deployments.

---

## 🛠️ Technology Stack

- **Backend:** Python
- **Frontend:** Streamlit
- **Database:** PostgreSQL
- **AI/ML:** TensorFlow, Transformers (BERT), spaCy, Scikit-learn, Pandas
- **Containerization:** Docker
- **File Processing:** PyPDF2, python-docx

---

## 🚀 Getting Started

There are two ways to get the application running: using Docker (recommended for ease of use) or setting it up manually for local development.

### Method 1: Docker Setup (Recommended)

This is the easiest and most reliable way to run the application, as it handles all dependencies and configurations for you.

**Prerequisites:**
- [Docker](https://www.docker.com/products/docker-desktop/) installed on your machine.
- A running PostgreSQL instance. You can easily start one with Docker:
  ```bash
  docker run --name some-postgres -e POSTGRES_PASSWORD=your_password -e POSTGRES_DB=ai_resume -p 5432:5432 -d postgres
  ```

**Steps:**

1.  **Build the Docker Image:**
    Open a terminal in the project root and run:
    ```bash
    docker build -t smart-resume-ai .
    ```

2.  **Run the Docker Container:**
    Run the application, connecting it to your PostgreSQL database. Make sure to replace `your_password` with the password you used when starting the PostgreSQL container.
    ```bash
    docker run -p 8501:8501 \
      -e DB_HOST=host.docker.internal \
      -e DB_NAME=ai_resume \
      -e DB_USER=postgres \
      -e DB_PASSWORD=your_password \
      smart-resume-ai
    ```
    *Note: `host.docker.internal` is a special DNS name that lets the container connect to services running on your host machine.*

3.  **Access the Application:**
    Open your web browser and go to `http://localhost:8501`.

### Method 2: Manual Local Setup

Follow these steps if you want to set up the development environment on your machine manually.

**Prerequisites:**
- Python 3.11
- PostgreSQL installed and running.
- Microsoft C++ Build Tools (for Windows users, to install dependencies).

**Steps:**

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/ShadowAniket/AI-RESUME.git
    cd AI-RESUME
    ```

2.  **Create a Virtual Environment:**
    ```bash
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set Up PostgreSQL:**
    - Create a new database (e.g., `ai_resume`).
    - Make sure you have a user and password with access to this database.

5.  **Configure Environment Variables:**
    The application uses environment variables for database credentials. The easiest way to manage these locally is to create a `.env` file in the project root. **Note: The `.env` file is listed in `.gitignore` and should never be committed to source control.**

    Create a file named `.env` and add the following, replacing the values with your actual database credentials:
    ```
    DB_HOST=localhost
    DB_PORT=5432
    DB_NAME=ai_resume
    DB_USER=your_db_user
    DB_PASSWORD=your_db_password
    ADMIN_EMAIL=admin@example.com
    ADMIN_PASSWORD=admin123
    ```

6.  **Initialize the Database:**
    Run the setup script to create the necessary tables and a default admin user. You need to install `python-dotenv` to load the `.env` file.
    ```bash
    pip install python-dotenv
    python -c "from dotenv import load_dotenv; load_dotenv(); import setup_db"
    ```

7.  **Run the Application:**
    ```bash
    streamlit run app.py
    ```

8.  **Access the Application:**
    Open your web browser and go to `http://localhost:8501`.

---

## ⚙️ Configuration

The application is configured via environment variables:

| Variable         | Description                                        | Default              |
| ---------------- | -------------------------------------------------- | -------------------- |
| `DB_HOST`        | The hostname of the PostgreSQL server.             | `localhost`          |
| `DB_PORT`        | The port of the PostgreSQL server.                 | `5432`               |
| `DB_NAME`        | The name of the database.                          | `ai_resume`          |
| `DB_USER`        | The username for the database connection.          | `postgres`           |
| `DB_PASSWORD`    | The password for the database connection.          | `postgres`           |
| `ADMIN_EMAIL`    | The email for the default admin user on setup.     | `admin@example.com`  |
| `ADMIN_PASSWORD` | The password for the default admin user on setup.  | `admin123`           |

---

## Usage

1.  **Resume Analyzer:** Navigate to the "Resume Analyzer" page, select a job role, and upload your resume (PDF or DOCX) to receive an instant ML-powered analysis and score.
2.  **Resume Builder:** Use the form to fill in your professional details. Click the "✨" buttons to get AI-generated content for your summary and experience sections.
3.  **Cover Letter Generator:** Go to the "Cover Letter Generator", paste a job description, and receive a tailored cover letter.
4.  **Portfolio Viewer:** After adding projects in the builder, view your shareable web portfolio on the "Portfolio Viewer" page.

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:
1.  Fork the repository.
2.  Create your feature branch (`git checkout -b feature/YourAmazingFeature`).
3.  Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4.  Push to the branch (`git push origin feature/YourAmazingFeature`).
5.  Open a Pull Request.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
