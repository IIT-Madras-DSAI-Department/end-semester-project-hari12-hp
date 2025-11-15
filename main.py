import numpy as np
import pandas as pd
import time
import sys

# Import our models from algorithms.py
try:
    from algorithms import (
        StandardScaler,
        RandomForest,
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
    
    print("Running Random Forest Classifier (Bagging)")
    
    # Load Data
    print("Loading data")
    Xtrain, ytrain, Xval, yval = read_data('MNIST_train.csv', 'MNIST_validation.csv')

    print(f"Training data:   {Xtrain.shape}")
    print(f"Validation data: {Xval.shape}")
    
    # Preprocess Data
    # Scaling is less critical for tree-based models, but good practice
    print("Scaling data...")
    scaler = StandardScaler()
    Xtrain_scaled = scaler.fit_transform(Xtrain)
    Xval_scaled = scaler.transform(Xval)

    # Initialize Model Parameters
    # Tune these parameters for the report!
    rf_params = {
        'n_trees': 50,              # Number of trees
        'max_depth': 12,            # Tree depth
        'n_features': 'sqrt',       # Max features to consider at each split
        'random_state': 42
    }
    
    # Initialize the Random Forest model
    model = RandomForest(**rf_params)

    # Train Model
    print("Training model (this will take time to build 50 trees)")
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
