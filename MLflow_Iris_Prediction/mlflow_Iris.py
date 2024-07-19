import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
import tempfile
# Load the Iris dataset
iris = datasets.load_iris()
X = pd.DataFrame(data=iris.data, columns=iris.feature_names)
y = pd.Series(iris.target)

# Check for NaN values and drop rows if any (though Iris dataset shouldn't have any NaNs)
X = X.dropna()

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Check if the train and test sets have samples
if X_train.shape[0] == 0 or X_test.shape[0] == 0:
    raise ValueError("Train or test set is empty after splitting. Please check the dataset or splitting parameters.")



# Function to plot and save confusion matrix
def plot_confusion_matrix_and_log(model, X_test, y_test, model_name):
    plt.figure(figsize=(8, 6))
    predictions = model.predict(X_test)
    cm = confusion_matrix(y_test, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix: {model_name}")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
        plt.savefig(tmpfile.name)
        mlflow.log_artifact(tmpfile.name, artifact_path="plots")
    plt.close()

# Function to plot and save feature importance
def plot_feature_importance_and_log(model, feature_names, model_name):
    if hasattr(model, 'feature_importances_'):
        plt.figure(figsize=(10, 6))
        importance = model.feature_importances_
        sns.barplot(x=importance, y=feature_names)
        plt.title(f"Feature Importance: {model_name}")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
            plt.savefig(tmpfile.name)
            mlflow.log_artifact(tmpfile.name, artifact_path="plots")
        plt.close()

# List of models to try
models = [
    {"name": "Logistic Regression", "model": LogisticRegression(max_iter=200), "params": {}},
    {"name": "Decision Tree", "model": DecisionTreeClassifier(), "params": {"max_depth": 5}},
    {"name": "Random Forest", "model": RandomForestClassifier(), "params": {"n_estimators": 100, "max_depth": 5}}
]

# Iterate through models
for model_info in models:
    model_name = model_info["name"]
    model = model_info["model"]
    params = model_info["params"]
    
    # Start an MLflow run
    with mlflow.start_run(run_name=model_name):
        # Set model parameters
        model.set_params(**params)
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        predictions = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, predictions)
        
        # Log parameters and metrics
        mlflow.log_params(params)
        mlflow.log_metric("accuracy", accuracy)
        
        # Log the model
        mlflow.sklearn.log_model(model, "model")
        
        # Plot and log confusion matrix
        plot_confusion_matrix_and_log(model, X_test, y_test, model_name)
        
        # Plot and log feature importance (for tree-based models)
        plot_feature_importance_and_log(model, X.columns, model_name)
        
        print(f"{model_name} - Accuracy: {accuracy}")

        

print("Model training, logging, and plotting completed!")
