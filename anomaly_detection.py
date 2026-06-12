# ============================================================
# Anomaly Detection - Fraud Detection System
# Internship Project | Skillbit Technologies
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────
# 1. Generate Synthetic Transaction Data
# ─────────────────────────────────────────
np.random.seed(42)

n_normal = 950
n_fraud = 50

# Normal transactions
normal = pd.DataFrame({
    'amount': np.random.normal(loc=200, scale=50, size=n_normal).clip(10, 500),
    'transaction_hour': np.random.randint(8, 22, size=n_normal),
    'age_of_account_days': np.random.randint(180, 3000, size=n_normal),
    'num_transactions_today': np.random.randint(1, 6, size=n_normal),
    'label': 0  # 0 = Normal
})

# Fraudulent transactions (unusual patterns)
fraud = pd.DataFrame({
    'amount': np.random.normal(loc=1500, scale=300, size=n_fraud).clip(800, 3000),
    'transaction_hour': np.random.choice([0, 1, 2, 3, 23], size=n_fraud),
    'age_of_account_days': np.random.randint(1, 30, size=n_fraud),
    'num_transactions_today': np.random.randint(15, 40, size=n_fraud),
    'label': 1  # 1 = Fraud
})

df = pd.concat([normal, fraud], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)

print("=" * 55)
print("   ANOMALY DETECTION - FRAUD DETECTION SYSTEM")
print("=" * 55)
print(f"\n📊 Dataset Overview:")
print(f"   Total Transactions : {len(df)}")
print(f"   Normal             : {(df['label'] == 0).sum()}")
print(f"   Fraudulent         : {(df['label'] == 1).sum()}")
print(f"\n{df.describe().round(2)}\n")

# ─────────────────────────────────────────
# 2. Preprocessing
# ─────────────────────────────────────────
features = ['amount', 'transaction_hour', 'age_of_account_days', 'num_transactions_today']
X = df[features]
y = df['label']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ─────────────────────────────────────────
# 3. Train Isolation Forest Model
# ─────────────────────────────────────────
print("🤖 Training Isolation Forest Model...")
model = IsolationForest(
    n_estimators=100,
    contamination=0.05,  # ~5% expected anomalies
    random_state=42
)
model.fit(X_scaled)

# Predictions: IsolationForest returns -1 for anomaly, 1 for normal
raw_preds = model.predict(X_scaled)
df['predicted'] = (raw_preds == -1).astype(int)  # 1 = fraud, 0 = normal
df['anomaly_score'] = -model.decision_function(X_scaled)  # higher = more anomalous

# ─────────────────────────────────────────
# 4. Evaluation
# ─────────────────────────────────────────
print("\n📈 Model Evaluation:")
print("-" * 40)
print(classification_report(y, df['predicted'], target_names=['Normal', 'Fraud']))

cm = confusion_matrix(y, df['predicted'])
print("Confusion Matrix:")
print(cm)

# ─────────────────────────────────────────
# 5. Visualizations
# ─────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Anomaly Detection - Fraud Analysis Dashboard', fontsize=16, fontweight='bold')

# Plot 1: Transaction Amount Distribution
ax1 = axes[0, 0]
ax1.hist(df[df['label'] == 0]['amount'], bins=40, alpha=0.7, color='steelblue', label='Normal')
ax1.hist(df[df['label'] == 1]['amount'], bins=20, alpha=0.7, color='crimson', label='Fraud')
ax1.set_title('Transaction Amount Distribution')
ax1.set_xlabel('Amount (₹)')
ax1.set_ylabel('Frequency')
ax1.legend()

# Plot 2: Anomaly Score Distribution
ax2 = axes[0, 1]
ax2.hist(df[df['predicted'] == 0]['anomaly_score'], bins=40, alpha=0.7, color='steelblue', label='Predicted Normal')
ax2.hist(df[df['predicted'] == 1]['anomaly_score'], bins=20, alpha=0.7, color='crimson', label='Predicted Fraud')
ax2.set_title('Anomaly Score Distribution')
ax2.set_xlabel('Anomaly Score')
ax2.set_ylabel('Frequency')
ax2.legend()

# Plot 3: Transaction Hour vs Amount (scatter)
ax3 = axes[1, 0]
normal_data = df[df['predicted'] == 0]
fraud_data = df[df['predicted'] == 1]
ax3.scatter(normal_data['transaction_hour'], normal_data['amount'], alpha=0.4, color='steelblue', label='Normal', s=20)
ax3.scatter(fraud_data['transaction_hour'], fraud_data['amount'], alpha=0.8, color='crimson', label='Fraud', s=40, marker='x')
ax3.set_title('Transaction Hour vs Amount')
ax3.set_xlabel('Hour of Day')
ax3.set_ylabel('Amount (₹)')
ax3.legend()

# Plot 4: Confusion Matrix Heatmap
ax4 = axes[1, 1]
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax4,
            xticklabels=['Normal', 'Fraud'],
            yticklabels=['Normal', 'Fraud'])
ax4.set_title('Confusion Matrix')
ax4.set_ylabel('Actual')
ax4.set_xlabel('Predicted')

plt.tight_layout()
plt.savefig('anomaly_detection_results.png', dpi=150, bbox_inches='tight')
print("\n✅ Visualization saved as 'anomaly_detection_results.png'")
plt.show()

# ─────────────────────────────────────────
# 6. Sample Predictions
# ─────────────────────────────────────────
print("\n🔍 Sample Flagged Transactions (Top 10 Suspicious):")
print("-" * 55)
top_suspicious = df.nlargest(10, 'anomaly_score')[features + ['anomaly_score', 'predicted', 'label']]
print(top_suspicious.to_string(index=False))

print("\n✅ Done! Model successfully identifies fraudulent transactions.")
