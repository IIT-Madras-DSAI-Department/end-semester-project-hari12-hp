import numpy as np
import pandas as pd
import time
import sys

try:
    from algorithms import (
        StandardScaler,
        LogisticRegression,
        OneVsRestClassifier,
        calculate_f1_score
    )
except ImportError:
    print("Error: 'algorithms.py' not found.")
    sys.exit(1)


def read_data(trainfile, validationfile):
    try:
        dftrain = pd.read_csv(trainfile)
        dfval = pd.read_csv(validationfile)
    except FileNotFoundError as e:
        print(f"Error: Could not find data file {e.filename}")
        sys.exit(1)

    target_col = 'label'
    all_cols = list(dftrain.columns)
    
    # Get pixel columns by removing label and even
    non_feature_cols = ['label', 'even']
    feature_cols = [col for col in all_cols if col not in non_feature_cols]

    # Normalize data
    Xtrain = np.array(dftrain[feature_cols]).astype(np.float64) / 255.0
    ytrain = np.array(dftrain[target_col]).astype(np.int64)
    Xval = np.array(dfval[feature_cols]).astype(np.float64) / 255.0
    yval = np.array(dfval[target_col]).astype(np.int64)

    return (Xtrain, ytrain, Xval, yval)


if __name__ == "__main__":
    
    print("Running Logistic Regression (One-vs-Rest)")
    
    # Load Data
    print("Loading data")
    Xtrain, ytrain, Xval, yval = read_data('MNIST_train.csv', 'MNIST_validation.csv')

    print(f"Training data:   {Xtrain.shape}")
    print(f"Validation data: {Xval.shape}")
    
    # Preprocess Data
    print("Scaling data...")
    scaler = StandardScaler()
    Xtrain_scaled = scaler.fit_transform(Xtrain)
    Xval_scaled = scaler.transform(Xval)

    # Initialize Model
    log_reg_base = LogisticRegression(
        learning_rate=0.1,
        n_iterations=500
    )
    
    model = OneVsRestClassifier(base_classifier=log_reg_base)

    # Train Model
    print("Training model (this may take a moment)")
    start_time = time.time()
    
    model.fit(Xtrain_scaled, ytrain)
    
    training_time = time.time() - start_time
    print(f"Training complete in {training_time:.2f} seconds.")

    # Evaluate
    print("Evaluating on validation data")
    ypred = model.predict(Xval_scaled)

    # Results
    accuracy = np.mean(yval == ypred)
    f1_macro = calculate_f1_score(yval, ypred, average='macro')

    print("\nValidation Results")
    print(f"Training Time:    {training_time:.2f}s")
    print(f"Accuracy:         {accuracy * 100:.2f}%")
    print(f"F1-Score (Macro): {f1_macro:.4f}")
