import pandas as pd
from sklearn.model_selection import train_test_split
from model import DeepSurvModel
from evaluation import evaluate_model, plot_km_curve

# Load data
data = pd.read_csv("processed_pancreatitis_data.csv")
X = data.drop(columns=["time", "event"])
y_time = data["time"]
y_event = data["event"]

# Train/test split
X_train, X_test, y_time_train, y_time_test, y_event_train, y_event_test = train_test_split(
    X, y_time, y_event, test_size=0.2, random_state=42
)

# Build and train model
model = DeepSurvModel(input_dim=X.shape[1])
model.train(X_train, y_time_train, y_event_train, epochs=100)

# Predict risk scores
risk_scores = model.predict(X_test)

# Evaluate model
c_index = evaluate_model(risk_scores, y_time_test, y_event_test)
print(f"Concordance index (C-index): {c_index:.4f}")

# Plot KM curve
event_data = pd.DataFrame({
    "time": y_time_test,
    "event": y_event_test,
    "risk": risk_scores
})
plot_km_curve(event_data)