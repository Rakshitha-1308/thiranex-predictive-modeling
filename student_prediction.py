# Predictive Modeling Using Machine Learning
# Thiranex Internship Task
# Project: Student Performance Prediction using Random Forest

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc

data = pd.read_csv("student_performance.csv")

print("First 5 rows:")
print(data.head())
print("\nDataset shape:", data.shape)
print("\nMissing values:")
print(data.isnull().sum())

X = data[["Study_Hours", "Attendance", "Previous_Marks", "Assignments_Completed"]]
y = data["Result"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, list(model.classes_).index("Pass")]

accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred, labels=["Fail", "Pass"])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Fail", "Pass"])
disp.plot()
plt.title("Confusion Matrix - Student Performance Prediction")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=200)
plt.show()

y_test_binary = (y_test == "Pass").astype(int)
fpr, tpr, _ = roc_curve(y_test_binary, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"Random Forest (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.tight_layout()
plt.savefig("roc_curve.png", dpi=200)
plt.show()

importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=True)
plt.figure()
importance.plot(kind="barh")
plt.xlabel("Importance")
plt.title("Feature Importance")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=200)
plt.show()

new_student = pd.DataFrame({
    "Study_Hours": [7.0],
    "Attendance": [88.0],
    "Previous_Marks": [76.0],
    "Assignments_Completed": [8]
})

prediction = model.predict(new_student)[0]
probability = model.predict_proba(new_student).max() * 100

print("\nNew Student Prediction:")
print("Predicted Result:", prediction)
print(f"Prediction Confidence: {probability:.2f}%")
