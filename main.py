import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

def prepare_dataset():
    data = pd.read_csv('creditcard.csv')  # Replace 'your_csv_file.csv' with the actual file name
    X = data[['Time', 'Amount', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28']]
    y = data['Class']  # Assuming 'Class' is the column indicating fraud (binary 1 for fraud, 0 for non-fraud)
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, y_train

def train_logistic_regression_model(X_train, y_train, python_flavor):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = LogisticRegression(max_iter=1000)

    start_time = time.time()

    model.fit(X_train_scaled, y_train)

    end_time = time.time()

    elapsed_time = end_time - start_time
    
    print(f"{python_flavor} Training Time:", elapsed_time, "seconds")
    return model

if __name__ == "__main__":
    X_train, y_train = prepare_dataset()
    model_cpython = train_logistic_regression_model(X_train, y_train, python_flavor="CPython")
    # To test PyPy, make sure PyPy is properly installed and available in your system PATH.
    model_pypy = train_logistic_regression_model(X_train, y_train, python_flavor="PyPy")
    # You can also run the same code with IronPython if you have it installed and set up.
    model_ironpython = train_logistic_regression_model(X_train, y_train, python_flavor="IronPython")

