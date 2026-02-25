# Smart Agriculture Crop Recommendation System

An AI-powered system that provides crop recommendations based on real-time sensor data, historical rainfall data via Open-Meteo API, and market profit analysis. It supports both **REST API** (FastAPI) and **MQTT** (EMQX) protocols.

---

## 🚀 Getting Started

Follow these steps to set up the project on your local machine.

### 1. Clone the Repository
```bash
git clone https://github.com/jaikrishn-j/esp32-test-case-1.git
cd esp32_soil
```

### 2. Create and Activate Virtual Environment
Create the virtual environment:
```bash
python3 -m venv .
```

Activate the environment based on your operating system:
- **Windows**:
  ```cmd
  .\Scripts\activate
  ```
- **macOS / Linux**:
  ```bash
  source bin/activate
  ```

### 3. Install Requirements
```bash
pip3 install -r requirements.txt
```

### 4. Run the Application
Start typing:
```bash
uvicorn app.main:app --reload
```

### 5. Access the API Documentation
Go to your browser and navigate to:
[http://localhost:8000/docs](http://localhost:8000/docs)