import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import os

#train random forest on extracted embeddings
#provides ensemble approach on top of deep features
def train_random_forest(train_embeddings: np.ndarray, 
                       train_labels: np.ndarray,
                       config: dict) -> RandomForestClassifier:
    
    rf = RandomForestClassifier(
        n_estimators=config['rf_estimators'],
        random_state=config['random_state'],
        n_jobs=-1
    )
    
    rf.fit(train_embeddings, train_labels)
    
    return rf

#evaluate random forest on test embeddings
def evaluate_random_forest(rf: RandomForestClassifier,
                          test_embeddings: np.ndarray,
                          test_labels: np.ndarray) -> dict:
    
    predictions = rf.predict(test_embeddings)
    accuracy = accuracy_score(test_labels, predictions)
    
    return {
        'accuracy': accuracy,
        'predictions': predictions
    }

#get detailed classification metrics
def get_classification_metrics(true_labels: np.ndarray, 
                               predictions: np.ndarray) -> dict:
    
    cm = confusion_matrix(true_labels, predictions)
    report = classification_report(true_labels, predictions, 
                                   target_names=['benign', 'malignant'],
                                   output_dict=True)
    
    return {
        'confusion_matrix': cm,
        'classification_report': report
    }

#save random forest model to disk
def save_random_forest(rf: RandomForestClassifier, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(rf, f)
    print(f"random forest saved to {path}")

#load random forest model from disk
def load_random_forest(path: str) -> RandomForestClassifier:
    with open(path, 'rb') as f:
        return pickle.load(f)
