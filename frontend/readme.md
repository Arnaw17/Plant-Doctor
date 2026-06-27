# Frontend

## Overview

The AI Plant Doctor frontend is built using **HTML, CSS, and JavaScript** and is served directly by the FastAPI backend.

Unlike applications that use frameworks such as React, Angular, or Vue, this project uses a **server-rendered frontend**. The user interface is embedded within the FastAPI application and delivered through `HTMLResponse`.

## Technologies Used

* HTML5
* CSS3
* JavaScript
* FastAPI HTMLResponse

## Features

* User Login and Registration
* Plant Dashboard
* Plant Registration
* Plant Image Upload
* AI Chat Interface
* Plant Diagnosis Display
* Diagnosis History
* Plant Care Reminders
* Responsive User Interface

## Folder Structure

```text
frontend/
└── README.md
```

> **Note:** The frontend source code (HTML, CSS, and JavaScript) is currently embedded inside `backend/app/main.py` as part of the FastAPI application. This project does not use a separate frontend framework.
