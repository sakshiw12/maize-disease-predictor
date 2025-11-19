# train.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load dataset
df = pd.read_csv("updated_data.csv")

# Fixed crop column (optional)
df["Crop"] = "Maize"

# Features and target
features = ["crop growth stage"]   # ONLY stage
target = "Disease Name"

X = df[features]
y = df[target]

# Encode categorical values
le_stage = LabelEncoder()
le_disease = LabelEncoder()

X.loc[:, "crop growth stage"] = le_stage.fit_transform(X["crop growth stage"])
y = le_disease.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Save model + encoders
with open("model.pkl", "wb") as f:
    pickle.dump({
        "model": model,
        "le_stage": le_stage,
        "le_disease": le_disease
    }, f)

print("Model trained and saved as model.pkl!")
