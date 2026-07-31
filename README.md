# Nigerian SMS Scam Detector 🇳🇬

## Overview
This capstone project is an end-to-end machine learning application designed to classify SMS messages as either fraudulent (scam) or safe. Tailored specifically for the Nigerian context, the model detects common local phishing attempts, such as fake bank deactivation alerts and "You have won" promo scams. 

Beyond simple classification, the application provides **Real-Time Keyword Insights**, exposing the specific words in a message that triggered a fraud flag, thereby offering transparent and explainable AI to the end user.

## Features
*   **Machine Learning Pipeline:** Utilizes a `TfidfVectorizer` paired with `LogisticRegression` for efficient text classification and interpretable feature weights.
*   **Explainable AI (XAI):** Dynamically extracts and displays the exact keywords and their mathematical impact on the model's prediction.
*   **Modern REST API:** Built with **FastAPI** and **Pydantic** for robust data validation and asynchronous request handling.
*   **Lightweight UI:** A decoupled, responsive frontend built with Vanilla JavaScript, HTML, and TailwindCSS, served directly from the backend.

## Tech Stack
*   **Data Science:** `python`, `pandas`, `numpy`, `scikit-learn`
*   **Backend/Serving:** `FastAPI`, `uvicorn`, `joblib`, `pydantic`
*   **Frontend:** HTML5, Vanilla JavaScript, TailwindCSS (via CDN)

## Project Architecture & Directory Structure
```text
sms-scam-detector/
├── app.py                  # FastAPI server and inference logic
├── model_training.ipynb    # Jupyter notebook for EDA, training, and evaluation
├── sms_fraud_model.joblib  # Serialized scikit-learn model pipeline
├── README.md               # Project documentation
└── static/
    └── index.html          # Frontend user interface
```

## Getting Started
### Prerequisites
- Python 3.8+
- pip (Python package installer)

### Installation
1. Clone the repository:

```bash
git clone [https://github.com/yourusername/sms-scam-detector.git](https://github.com/yourusername/sms-scam-detector.git)
cd sms-scam-detector
```
2. Create and activate a virtual environment (Recommended):

```bash
python -m venv venv

# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

3. Install dependencies:
(_Assuming a `requirements.txt` is present. If not, install manually_: `pip install fastapi uvicorn scikit-learn pandas numpy pydantic`)

```bash
pip install -r requirements.txt
```

### Usage Workflow
1. **Train the Model**: Before running the API, you must train the classifier and generate the serialized .joblib file. Open `model_training.ipynb` in Jupyter Notebook or VS Code.
Run all cells to process the dataset, evaluate metrics (Precision, Recall, F1-Score), and export sms_fraud_model.joblib to the root directory.

2. Start the Backend Server
Run the FastAPI application using Uvicorn:

```bash
uvicorn app:app --reload
```

The `--reload` flag enables auto-reloading during development.

3. Access the Application
 - Frontend UI: Open your browser and navigate to http://localhost:8000 to interact with the web interface.

 - API Documentation: Navigate to http://localhost:8000/docs to view the automatic interactive Swagger UI for the prediction endpoint.

### Evaluation Metrics
The model is evaluated using standard classification metrics to ensure high precision (minimizing false positives, where safe messages are flagged as scams) and high recall (ensuring most scams are caught). Detailed confusion matrices and classification reports are available within the training notebook.

License
MIT License