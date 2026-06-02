import mlflow, mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

mlflow.set_tracking_uri("sqlite:///quiz.db")
mlflow.set_experiment("quiz-experiment")

features, targets = load_iris(return_X_y=True)

train_features, test_features, train_targets, test_targets = train_test_split(
    features,
    targets,
    test_size=0.2,
    random_state=0
)

MAX_DEPTH = 5
N_ESTIMATORS = 100

with mlflow.start_run(run_name="quiz-run"):

    # Log parameters
    mlflow.log_param("max_depth", MAX_DEPTH)
    mlflow.log_param("n_estimators", N_ESTIMATORS)

    # Train model
    random_forest_model = RandomForestClassifier(
        max_depth=MAX_DEPTH,
        n_estimators=N_ESTIMATORS,
        random_state=42
    )

    random_forest_model.fit(train_features, train_targets)

    predicted_labels = random_forest_model.predict(test_features)

    # Log metrics
    mlflow.log_metric(
        "accuracy",
        accuracy_score(test_targets, predicted_labels)
    )

    mlflow.log_metric(
        "f1",
        f1_score(test_targets, predicted_labels, average="macro")
    )

    # Log and register model
    mlflow.sklearn.log_model(
        random_forest_model,
        "random-forest-model",
        registered_model_name="iris-rf-classifier"
    )

    # Set tag
    mlflow.set_tag("team", "data-science")