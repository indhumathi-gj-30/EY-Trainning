import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Scikit-learn datasets & model selection
from sklearn.datasets import (
    make_classification, make_regression,
    load_breast_cancer, load_diabetes
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Classification models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

# Regression models
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor

# Evaluation metrics
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score,
    ConfusionMatrixDisplay
)

# Plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('husl')
np.random.seed(42)

print("All libraries loaded successfully!")
print(f"NumPy: {np.__version__} | Pandas: {pd.__version__}")



cancer_dataset = load_breast_cancer()
features_clf, labels_clf = cancer_dataset.data, cancer_dataset.target

train_X_clf, test_X_clf, train_y_clf, test_y_clf = train_test_split(
    features_clf,
    labels_clf,
    test_size=0.2,
    random_state=42
)

scaler_clf = StandardScaler()
train_X_clf = scaler_clf.fit_transform(train_X_clf)
test_X_clf = scaler_clf.transform(test_X_clf)

# Regression dataset
diabetes_dataset = load_diabetes()
features_reg, labels_reg = diabetes_dataset.data, diabetes_dataset.target

train_X_reg, test_X_reg, train_y_reg, test_y_reg = train_test_split(
    features_reg,
    labels_reg,
    test_size=0.2,
    random_state=42
)

scaler_reg = StandardScaler()
train_X_reg = scaler_reg.fit_transform(train_X_reg)
test_X_reg = scaler_reg.transform(test_X_reg)

# Dataset summary
print("CLASSIFICATION DATASET (Breast Cancer)")
print(f"Train: {train_X_clf.shape} | Test: {test_X_clf.shape}")
print(
    f"Class balance — Malignant: {(labels_clf==0).sum()} | "
    f"Benign: {(labels_clf==1).sum()}"
)

print("\nREGRESSION DATASET (Diabetes)")
print(f"Train: {train_X_reg.shape} | Test: {test_X_reg.shape}")
print(f"Target range: [{labels_reg.min():.0f}, {labels_reg.max():.0f}]")
print("Feature names:", cancer_dataset.feature_names[:5])

malignant_count = (labels_clf == 0).sum()
benign_count = (labels_clf == 1).sum()

imbalance_ratio = malignant_count / benign_count
print(f"Class imbalance ratio: {imbalance_ratio:.2f}")

# Plot distribution
class_labels = ['Malignant (0)', 'Benign (1)']
class_counts = [malignant_count, benign_count]

plt.bar(class_labels, class_counts, color=['red', 'green'])
plt.xlabel("Class")
plt.ylabel("Count")
plt.title("Class Distribution — Breast Cancer Dataset")
plt.show()

model_dict = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
}

evaluation_results = {}

for model_name, model_obj in model_dict.items():
    model_obj.fit(train_X_clf, train_y_clf)

    predictions = model_obj.predict(test_X_clf)
    probabilities = model_obj.predict_proba(test_X_clf)[:, 1]

    evaluation_results[model_name] = {
        'model': model_obj,
        'y_pred': predictions,
        'y_proba': probabilities,
        'accuracy': accuracy_score(test_y_clf, predictions),
        'cm': confusion_matrix(test_y_clf, predictions)
    }

    print(f"{model_name}: Accuracy = {evaluation_results[model_name]['accuracy']:.4f}")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for axis, (model_name, result) in zip(axes, evaluation_results.items()):
    cm_display = ConfusionMatrixDisplay(
        confusion_matrix=result['cm'],
        display_labels=cancer_dataset.target_names
    )

    cm_display.plot(ax=axis, colorbar=False, cmap='Blues')
    axis.set_title(
        f"{model_name}\nAcc: {result['accuracy']:.3f}",
        fontsize=11
    )

plt.tight_layout()
plt.suptitle(
    "Confusion Matrices — All Models",
    y=1.02,
    fontsize=13,
    fontweight='bold'
)
plt.show()


log_reg_model = LogisticRegression()
log_reg_model.fit(train_X_clf, train_y_clf)

log_predictions = log_reg_model.predict(test_X_clf)

conf_matrix = confusion_matrix(test_y_clf, log_predictions)
print("Confusion Matrix:\n", conf_matrix)

true_negative = conf_matrix[0][0]
false_positive = conf_matrix[0][1]
false_negative = conf_matrix[1][0]
true_positive = conf_matrix[1][1]

print(f"TN: {true_negative}, FP: {false_positive}, FN: {false_negative}, TP: {true_positive}")

manual_accuracy_value = (true_positive + true_negative) / (
    true_positive + true_negative + false_positive + false_negative
)

print(f"Manual Accuracy: {manual_accuracy_value:.4f}")

sklearn_accuracy_value = accuracy_score(test_y_clf, log_predictions)
print(f"Sklearn Accuracy: {sklearn_accuracy_value:.4f}")