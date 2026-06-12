# 🔍 Anomaly Detection – Fraud Detection System

> An AI-powered fraud detection system using **Isolation Forest** to identify suspicious financial transactions.

Built as part of the **AI Internship at Skillbit Technologies**.

---

## 📌 Project Overview

This project implements an **Anomaly Detection** system applied to financial transaction data. It automatically flags unusual transactions that may indicate fraudulent activity — without needing labelled training data (unsupervised learning).

---

## 🧠 Algorithm Used

**Isolation Forest**
- An unsupervised machine learning algorithm
- Works by randomly isolating observations — anomalies are isolated faster
- Ideal for fraud detection where fraudulent samples are rare

---

## 📊 Features Used

| Feature | Description |
|---|---|
| `amount` | Transaction amount in ₹ |
| `transaction_hour` | Hour of day the transaction occurred |
| `age_of_account_days` | How old the account is (in days) |
| `num_transactions_today` | Number of transactions made today |

---

## 🗂️ Project Structure

```
anomaly-detection/
│
├── anomaly_detection.py       # Main Python script
├── anomaly_detection_results.png  # Output visualization
├── requirements.txt           # Dependencies
└── README.md                  # Project documentation
```

---

## ⚙️ How to Run

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/anomaly-detection.git
cd anomaly-detection
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the script
```bash
python anomaly_detection.py
```

---

## 📈 Output

- **Console**: Dataset overview, model evaluation report, confusion matrix
- **Chart**: 4-panel dashboard with distributions, scatter plots, and heatmap saved as `anomaly_detection_results.png`

---

## 🛠️ Tech Stack

- Python 3.x
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn

---

## 📚 What I Learned

- How unsupervised anomaly detection works
- Applying Isolation Forest for fraud detection
- Data preprocessing with StandardScaler
- Evaluating models using confusion matrix & classification report
- Visualizing ML results with matplotlib & seaborn

---

## 👨‍💻 Author

Kennice James Dabreo
AI Intern – Skillbit Technologies
[LinkedIn](https://www.linkedin.com/in/kennice-dabreo-23294a343/) | [GitHub](https://github.com/Kennicee)
