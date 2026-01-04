# train_model_v2.py
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# 1. Load dataset
df = pd.read_csv("casestudy_dataset.csv")

# 2. Replace zeros with median in selected columns
cols_with_zeros = ["Glucose", "Insulin", "BMI"]
for col in cols_with_zeros:
    df[col] = df[col].replace(0, df[col].median())

# 3. Features and target
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 6. Train Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 7. Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model V2 Accuracy: {accuracy:.4f}")

# 8. Save model
with open("model_v2.pkl", "wb") as f:
    pickle.dump(model, f)

# 9. Save scaler (needed later for prediction app)
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)