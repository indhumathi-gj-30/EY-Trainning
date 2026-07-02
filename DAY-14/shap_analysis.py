import shap
numeric_features = ['income', 'credit_score', 'loan_amount', 'existing_loans', 'age']
loan_data = df.copy()

for column in ['gender', 'region', 'employment_type']:
    loan_data[f'{column}_code'] = LabelEncoder().fit_transform(loan_data[column])

model_features = numeric_features + [
    'gender_code',
    'region_code',
    'employment_type_code'
]

X = loan_data[model_features]
y = loan_data['rejected']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
model = LogisticRegression(max_iter=500, random_state=42)
model.fit(X_train_scaled, y_train)

accuracy = model.score(X_test_scaled, y_test)
print("Classification Accuracy:", round(accuracy, 3))
explainer = shap.LinearExplainer(
    model,
    X_train_scaled,
    feature_names=model_features
)

shap_results = explainer(X_test_scaled)
plt.figure(figsize=(9, 5))
shap.summary_plot(
    shap_results,
    X_test,
    feature_names=model_features,
    plot_type="bar",
    show=False
)

plt.title(
    "Feature Contribution Analysis for Loan Rejection Prediction",
    fontweight="bold"
)

plt.tight_layout()
plt.show()

print("\nFairness Review:")
print("Check whether gender_code or region_code appears among the most influential features.")
print("If protected attributes significantly affect predictions, the model may require bias mitigation.")