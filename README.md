# Thiranex Internship Project
## Predictive Modeling Using Machine Learning

### Project Title
Student Performance Prediction using Random Forest

### Objective
Build a supervised machine learning model that predicts whether a student will Pass or Fail based on study hours, attendance, previous marks, and assignments completed.

### Technologies
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib

### Machine Learning Algorithm
Random Forest Classifier

### Workflow
1. Load the dataset
2. Check and prepare the data
3. Select input features and target
4. Split data into training and testing sets
5. Train Random Forest
6. Predict results
7. Evaluate accuracy
8. Generate confusion matrix and ROC curve
9. Analyze feature importance
10. Test the model with a new student

### Dataset Features
- Study_Hours: Hours spent studying
- Attendance: Attendance percentage
- Previous_Marks: Previous examination marks
- Assignments_Completed: Number of completed assignments
- Result: Pass or Fail

### How to Run
```bash
pip install pandas numpy scikit-learn matplotlib
python student_prediction.py
```

The program creates confusion_matrix.png, roc_curve.png, and feature_importance.png.

### Expected Outcome
The project demonstrates supervised learning, model training, prediction, and model evaluation using a classification problem.

### Note
The included dataset is a synthetic educational dataset created for demonstration and internship submission practice.
