# train_model_v1.py
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1. Load dataset
df = pd.read_csv("casestudy_dataset.csv")

# 2. Features and target
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# 3. Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train Logistic Regression (baseline model)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 5. Evaluate accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model V1 Accuracy: {accuracy:.4f}")

# 6. Save model
with open("model_v1.pkl", "wb") as f:
    pickle.dump(model, f)