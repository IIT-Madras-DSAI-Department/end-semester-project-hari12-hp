import numpy as np
import pandas as pd
import time
import sys
import os

# Import all necessary models and utilities
try:
    from algorithms import (
        StandardScaler, PCA, KNearestNeighbors, 
        LogisticRegression, OneVsRestClassifier, 
        RandomForest, OneVsRestXGBoost, 
        calculate_f1_score
    )
except ImportError:
    print("Error: 'algorithms.py' not found or is missing a model (like optimized KNN).")
    sys.exit(1)


def read_data(trainfile, validationfile):
    """ Reads the MNIST CSV files. """
    try:
        dftrain = pd.read_csv(trainfile)
        dfval = pd.read_csv(validationfile)
    except FileNotFoundError as e:
        print(f"Error: Could not find data file {e.filename}")
        sys.exit(1)

    target_col = 'label'
    all_cols = list(dftrain.columns)
    
    non_feature_cols = ['label', 'even']
    feature_cols = [col for col in all_cols if col not in non_feature_cols]

    Xtrain = np.array(dftrain[feature_cols]).astype(np.float64) / 255.0
    ytrain = np.array(dftrain[target_col]).astype(np.int64)
    Xval = np.array(dfval[feature_cols]).astype(np.float64) / 255.0
    yval = np.array(dfval[target_col]).astype(np.int64)

    return (Xtrain, ytrain, Xval, yval)


if __name__ == "__main__":
    
    print("--- Running Final Stacking Ensemble (RF + XGB + LogReg + KNN) [TIME OPTIMIZED] ---")
    
    # 1. Load Data
    print("Loading data")
    Xtrain, ytrain, Xval, yval = read_data('MNIST_train.csv', 'MNIST_validation.csv')

    # 2. Preprocessing
    # Path 1: Scaled Data (for RF, XGB, LogReg)
    print("Scaling data...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(Xtrain)
    X_val_scaled = scaler.transform(Xval)
    print("Scaling complete.")

    # Path 2: PCA Data (for KNN model ONLY)
    N_COMPONENTS = 50  # <-- REDUCED FROM 100 TO SPEED UP KNN
    print(f"Running PCA to {N_COMPONENTS} components (for KNN)...")
    pca = PCA(n_components=N_COMPONENTS)
    X_train_pca = pca.fit_transform(X_train_scaled) # Use scaled data for PCA
    X_val_pca = pca.transform(X_val_scaled)
    print("PCA complete.")


    # 3. Initialize Base Models (Level 0)
    
    # Model 1: Random Forest
    rf_model = RandomForest(
        n_trees=15,       # <-- REDUCED FROM 25 TO SPEED UP RF
        max_depth=10,     
        n_features='sqrt', 
        random_state=42
    )
    
    # Model 2: XGBoost
    xgb_params = {
        'n_estimators': 20, 'learning_rate': 0.3, 'max_depth': 3, 
        'subsample': 0.8, 'colsample_bytree': 0.2, 
        'random_state': 42, 'n_bins': 32
    }
    xgb_model = OneVsRestXGBoost(base_xgb_params=xgb_params)

    # Model 3: Logistic Regression
    log_reg_base = LogisticRegression(
        learning_rate=0.1, 
        n_iterations=150,
        random_state=42
    )
    log_reg_model = OneVsRestClassifier(base_classifier=log_reg_base)

    # Model 4: K-Nearest Neighbors (using k=3 is slightly faster)
    knn_model = KNearestNeighbors(k=3) # <-- k=3 is faster than k=5

    # Level 1 (Meta-Model)
    meta_lr_base = LogisticRegression(learning_rate=0.1, n_iterations=150)
    meta_model = OneVsRestClassifier(base_classifier=meta_lr_base)

    base_models = {
        'RandomForest': (rf_model, X_train_scaled, X_val_scaled),
        'XGBoost (OvR)': (xgb_model, X_train_scaled, X_val_scaled),
        'LogReg (OvR)': (log_reg_model, X_train_scaled, X_val_scaled),
        'KNN (PCA-50)': (knn_model, X_train_pca, X_val_pca) 
    }

    # 4. Stacking: Train Level 0 and Generate Meta-Features
    meta_features_train_list = []
    meta_features_val_list = []
    
    start_time = time.time()
    print("\nTraining Level 0 Models...")
    
    for name, (model, X_tr, X_val_base) in base_models.items():
        print(f"-> Training Level 0 Model: {name}")
        model.fit(X_tr, ytrain)
        
        print(f"  -> Generating meta-features with {name}...")
        train_preds_proba = model.predict_proba(X_tr)
        val_preds_proba = model.predict_proba(X_val_base)

        meta_features_train_list.append(train_preds_proba)
        meta_features_val_list.append(val_preds_proba)

    # Concatenate all meta-features (10002, 40)
    X_train_meta = np.hstack(meta_features_train_list)
    X_val_meta = np.hstack(meta_features_val_list)

    print(f"\nMeta-features created. Train shape: {X_train_meta.shape}, Val shape: {X_val_meta.shape}")

    # 5. Train Level 1 Meta-Model
    print("-> Training Level 1 Meta-Model (Logistic Regression)...")
    
    meta_scaler = StandardScaler()
    X_train_meta_scaled = meta_scaler.fit_transform(X_train_meta)
    X_val_meta_scaled = meta_scaler.transform(X_val_meta)
    
    meta_model.fit(X_train_meta_scaled, ytrain)
    
    training_time = time.time() - start_time
    print(f"Total Stacking Training Time: {training_time:.2f} seconds.")

    # 6. Final Evaluation
    print("Evaluating final Stacking Ensemble...")
    y_pred_final = meta_model.predict(X_val_meta_scaled)

    accuracy = np.mean(yval == y_pred_final)
    f1_macro = calculate_f1_score(yval, y_pred_final, average='macro')

    print("\nFINAL VALIDATION RESULTS (Stacking Ensemble)")
    print(f"Total Time:       {training_time:.2f}s")
    print(f"Accuracy:         {accuracy * 100:.2f}%")
    print(f"F1-Score (Macro): {f1_macro:.4f}")
