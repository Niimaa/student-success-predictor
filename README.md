

# 🎓 Student Success Predictor

A machine learning application that predicts the likelihood of a new student successfully completing their study programme, based on demographic and educational history data.

---

## 📌 Project Objective
- Identify students at risk of not completing their program
- Use demographic and enrollment data to predict outcomes
- Provide an interpretable, accessible prediction tool for educators

---

## 🧠 How It Works
The project includes:
- Preprocessing and feature engineering in Python
- A trained XGBoost classification model
- Streamlit-based UI for real-time predictions

Predictions are based on:
- Programme Name
- Secondary School Award
- Pre-Study Activity
- Secondary School
- Ethnicity
- First Tertiary Year (Grouped)
- Age Group

---

## 🖥️ How to Run Locally

1. Clone the repo:
```bash
git clone https://github.com/your-username/student-success-predictor.git
cd student-success-predictor
```

2. Create virtual environment (optional):
```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate (Windows)
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Launch Streamlit app:
```bash
streamlit run app/ui/streamlit_app.py
```

---

## 🚀 Streamlit Cloud (Live App)
Once deployed via [Streamlit Cloud](https://streamlit.io/cloud), access the app at:
```
https://your-app-name.streamlit.app/
```

---

## 📁 Project Structure
```
student_success_predictor/
├── app/
│   ├── api/              # Optional Flask API (if used)
│   ├── model/            # Prediction logic
│   └── ui/               # Streamlit app
├── models/               # Trained model + encoders
├── requirements.txt      # Python dependencies
├── .gitignore
├── README.md             # This file
```

---

## 📬 Contact / Maintainers
Created as part of an educational analytics initiative. Contact the project owner or organization for long-term ownership, deployment, or hosting decisions.

---

✅ Ready to deploy. Predictions are explainable and grounded in historical trends.

