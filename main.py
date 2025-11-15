import numpy as np
import pandas as pd
import time
import sys

# Import our models from algorithms.py
try:
    from algorithms import (
        StandardScaler,
        LogisticRegression,
        OneVsRestClassifier,
        calculate_f1_score
    )
except ImportError:
    print("Error: 'algorithms.py' not found.")
    print("Please make sure it's in the same folder as main.py.")
    sys.exit(1)



def read_data(trainfile, validationfile):
    dftrain = pd.read_csv(trainfile)
    dfval = pd.read_csv(validationfile)


    target_col = 'label'
    
    all_cols = list(dftrain.columns)
    
    # Get all pixel columns by removing label and even
    non_feature_cols = ['label', 'even']
    feature_cols = [col for col in all_cols if col not in non_feature_cols]

    # Normalize pixels from 0-255 to 0-1
    Xtrain = np.array(dftrain[feature_cols]).astype(np.float64) / 255.0
    ytrain = np.array(dftrain[target_col]).astype(np.int64)
    
    Xval = np.array(dfval[feature_cols]).astype(np.float64) / 255.0
    yval = np.array(dfval[target_col]).astype(np.int64)

    return (Xtrain, ytrain, Xval, yval)


if __name__ == "__main__":
    
    print("Running Logistic Regression (One-vs-Rest)")
    
    # 1. Load Data
    print("Loading data")
    Xtrain, ytrain, Xval, yval = read_data('MNIST_train.csv', 'MNIST_validation.csv')

    print(f"Training data:   {Xtrain.shape}")
    print(f"Validation data: {Xval.shape}\n")
    
    # 2. Preprocess Data
    print("Scaling data...")
    scaler = StandardScaler()
    Xtrain_scaled = scaler.fit_transform(Xtrain)
    Xval_scaled = scaler.transform(Xval)

    # 3. Initialize Model
    log_reg_base = LogisticRegression(
        learning_rate=0.1,
        n_iterations=500
    )
    
    # Use the OvR wrapper to handle 10 classes
    model = OneVsRestClassifier(base_classifier=log_reg_base)

    # 4. Train Model
    print("Training model (this may take a moment)")
    start_time = time.time()
    
    model.fit(Xtrain_scaled, ytrain)
    
    training_time = time.time() - start_time
    print(f"Training complete in {training_time:.2f} seconds.\n")

    # 5. Evaluate
    print("Evaluating on validation data")
    ypred = model.predict(Xval_scaled)

    # 6. Results
    accuracy = np.mean(yval == ypred)
    f1_macro = calculate_f1_score(yval, ypred, average='macro')

    print(" Validation Results")
    print(f"Training Time: {training_time:.2f}s")
    print(f"Accuracy:      {accuracy * 100:.2f}%")
    print(f"F1-Score (Macro): {f1_macro:.4f}")