import mlflow
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from mlflow.models import infer_signature
import mlflow.sklearn
import mlflow.exceptions
from pathlib import Path
import joblib

# Load the Iris dataset
X_train = pd.read_csv("data/processed/titanic_train.csv")
y_train = pd.read_csv("data/processed/titanic_train_labels.csv")
X_test = pd.read_csv("data/processed/titanic_test.csv")
y_test = pd.read_csv("data/processed/titanic_test_labels.csv")

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

lr = LogisticRegression()
lr.fit(X_train, y_train)


y_pred = lr.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="macro")
recall = recall_score(y_test, y_pred, average="macro")
f1 = f1_score(y_test, y_pred, average="macro")
print(accuracy, precision, recall, f1)


experiment_name = "MLflow Titanic"
run_name = "run 01"
try:
    experiment_id = mlflow.create_experiment(name=experiment_name)
except mlflow.exceptions.MlflowException as e:
    experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

print(experiment_id)

with mlflow.start_run(run_name=run_name, experiment_id=experiment_id) as run:

    mlflow.log_params(params={})

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1", f1)
    mlflow.log_metrics({
        "accuracy": accuracy,
        "f1": f1
    })

    mlflow.set_tag("Training Info", "Basic LR model for titanic data")

    signature = infer_signature(X_test, y_test)


    model_info = mlflow.sklearn.log_model(
        sk_model=lr,
        artifact_path="titanic_model",
        signature=signature,
        input_example=X_test,
        registered_model_name="titanic_model",
        pyfunc_predict_fn = "predict_proba"
    )

    sk_pyfunc = mlflow.sklearn.load_model(model_uri=model_info.model_uri)

    predictions = sk_pyfunc.predict(X_test)
    print(predictions)

    

    eval_data = pd.DataFrame(y_test)
    eval_data.columns = ["label"]
    eval_data["predictions"] = predictions
    
    results = mlflow.evaluate(
        data=eval_data,
        model_type="classifier",
        targets= "label",
        predictions="predictions",
        evaluators = ["default"]
    )

    print(f"metrics:\n{results.metrics}")
    print(f"artifacts:\n{results.artifacts}")

# Persist the trained scaler and model to the repository root models/trained_model
repo_root = Path(__file__).resolve().parents[2]
models_dir = repo_root / "models"
models_dir.mkdir(parents=True, exist_ok=True)
output_path = models_dir / "trained_model"
joblib.dump({"scaler": scaler, "model": lr}, output_path)
print(f"Saved trained artifacts to: {output_path}")