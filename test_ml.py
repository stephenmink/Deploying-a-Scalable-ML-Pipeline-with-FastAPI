import pytest
# TODO: add necessary import
import numpy as np  
from sklearn.ensemble import RandomForestClassifier  
from ml.data import process_data  
from ml.model import train_model, compute_model_metrics, inference  
import pandas as pd

# TODO: implement the first test. Change the function name and input as needed
def test_process_data_output():
    """
    Test if `process_data` returns the expected types of results.
    """
    # Create a small synthetic dataset
    data = {
        "workclass": ["Private", "Self-emp-not-inc"],
        "education": ["Bachelors", "HS-grad"],
        "marital-status": ["Never-married", "Married-civ-spouse"],
        "occupation": ["Tech-support", "Craft-repair"],
        "relationship": ["Not-in-family", "Husband"],
        "race": ["White", "Black"],
        "sex": ["Male", "Female"],
        "native-country": ["United-States", "India"],
        "salary": [">50K", "<=50K"],
    }
    
    import pandas as pd
    df = pd.DataFrame(data)

    cat_features = ["workclass", "education", "marital-status", "occupation", "relationship", "race", "sex", "native-country"]

    # Process the data
    X, y, encoder, lb = process_data(df, categorical_features=cat_features, label="salary", training=True)

    # assert the types of outputs
    assert isinstance(X, np.ndarray), "X should be a NumPy array"  
    assert isinstance(y, np.ndarray), "y should be a NumPy array"  
    assert encoder is not None, "Encoder should not be None"  
    assert lb is not None, "Label Binarizer should not be None"  



# TODO: implement the second test. Change the function name and input as needed
def test_train_model_algorithm():
    """
    Test if the `train_model` function returns a model of the expected type.
    """
    # Create synthetic data
    X_train = np.random.rand(100, 5)  # 100 samples, 5 features
    y_train = np.random.randint(0, 2, 100)  # Binary labels (0 or 1)

    # Train the model
    model = train_model(X_train, y_train)

    # Assert the model type
    assert isinstance(model, RandomForestClassifier), "Model should be a RandomForestClassifier"  


# TODO: implement the third test. Change the function name and input as needed
def test_compute_model_metrics():
    """
    Test if `compute_model_metrics` returns the expected values for a known case.
    """
    # Highlighted addition: Define known inputs and outputs
    y_true = np.array([1, 0, 1, 0])
    y_preds = np.array([1, 0, 1, 1])  # One false positive

    # Compute metrics
    precision, recall, fbeta = compute_model_metrics(y_true, y_preds)

    # Assert the computed metrics
    assert precision == pytest.approx(2 / 3, rel=1e-2), "Precision should be approximately 0.6667"  
    assert recall == pytest.approx(1.0, rel=1e-2), "Recall should be approximately 1.0"  
    assert fbeta == pytest.approx(0.8, rel=1e-2), "F1 score should be approximately 0.8" 
