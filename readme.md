# 🌿 AI Plant Doctor

AI Plant Doctor is an AI-powered web application that helps users diagnose plant health issues and receive personalized treatment recommendations. Users can upload plant images or describe symptoms, and the AI analyzes the input to identify possible diseases, nutrient deficiencies, pest infestations, or environmental problems.

The application also allows users to manage plant profiles, track diagnosis history, schedule care reminders, and monitor plant health over time.

---

## Features

* 🌱 User Registration and Login
* 📷 Plant Image Upload
* 🤖 AI-powered Plant Diagnosis
* 💬 AI Chat Assistant for Plant Care
* 🩺 Personalized Treatment Recommendations
* 📊 Plant Health History Tracking
* ⏰ Watering and Care Reminders
* 📁 Plant Profile Management
* 🔐 Secure Authentication
* 🗄 PostgreSQL Database Integration

---

## Technology Stack

### Frontend

* HTML5
* CSS3
* JavaScript
* FastAPI HTMLResponse (Server-rendered UI)

### Backend

* FastAPI
* SQLAlchemy
* Alembic
* PostgreSQL

### AI Components

* LangChain
* Chroma Vector Database (RAG)
* Large Language Model (LLM)

---

## Project Structure

```text
AI-Plant-Doctor/
│
├── frontend/
│   └── README.md
│
├── backend/
│   ├── app/
│   ├── alembic/
│   ├── chroma_store/
│   ├── uploads/
│   ├── requirements.txt
│   ├── alembic.ini
│   └── .env.example
│
├── README.md
└── .gitignore
```

> **Note:** The frontend is server-rendered using FastAPI. The HTML, CSS, and JavaScript are embedded within the FastAPI application rather than using a separate frontend framework such as React or Angular.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/AI-Plant-Doctor.git

cd AI-Plant-Doctor/backend
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it:

**Windows**

```bash
venv\Scripts\activate
```

**Linux / macOS**

```bash
source venv/bin/activate
```

---

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

### 4. Configure environment variables

Create a `.env` file based on `.env.example`.

Example:

```env
DATABASE_URL=your_database_url
OPENAI_API_KEY=your_api_key
SECRET_KEY=your_secret_key
```

---

### 5. Run database migrations

```bash
alembic upgrade head
```

---

### 6. Start the application

```bash
uvicorn app.main:app --reload
```

Open your browser:

```
http://127.0.0.1:8000
```

---

## Future Improvements

* AI image-based disease detection using computer vision
* Mobile application (Android & iOS)
* Weather-based plant care recommendations
* Multi-language support
* Push notifications for reminders
* Plant community discussion forum
* Advanced analytics dashboard

