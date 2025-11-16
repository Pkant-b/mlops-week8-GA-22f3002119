import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
import mlflow
import mlflow.sklearn
import os


mlflow.set_tracking_uri("http://35.184.50.158:5000")

data = pd.read_csv("./iris.csv")
X = data.drop("species", axis = 1)
y = LabelEncoder().fit_transform(data["species"])

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=21, stratify = y)

def poison_data(y_true, poison_percentage):
    y_poisoned = np.copy(y_true)
    n_samples = len(y_poisoned)
    n_to_poison = int(n_samples * poison_percentage)
    if n_to_poison == 0:
        return y_poisoned
    
    poison_indices = np.random.choice(n_samples, size = n_to_poison, replace = False)
    all_classes = [0,1,2]
    
    for index in poison_indices:
        current_level = y_poisoned[index]
        possible_new_labels = np.setdiff1d(all_classes, [current_level])
        new_label = np.random.choice(possible_new_labels)
        y_poisoned[index] = new_label
        
    return y_poisoned


mlflow.set_experiment("IRIS_poisoning_test")
max_depth = 3
random_state = 1
poison_levels = [0.0, 0.05, 0.10, 0.50]

for i in poison_levels:
    if i == 0.0:
        run_name = "Baseline_0_percent_poison"
    else:
        run_name = f"Poisoned_{int(i*100)}_percent"
        
    with mlflow.start_run(run_name = run_name):
        mlflow.log_param("model_type", "DecisionTreeClassifier")
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("poison_percent", i)
        
        y_train_current = poison_data(y_train, i)
        mod_dt = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
        mod_dt.fit(X_train, y_train_current)

        prediction = mod_dt.predict(X_val)
        accuracy = metrics.accuracy_score(prediction, y_val)
        mlflow.log_metric("accuracy", accuracy)

        input_example = X_train.head(5)
        mlflow.sklearn.log_model(
                            mod_dt, 
                            "model",
                            input_example=input_example
                        )