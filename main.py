assifier, 
        LinearSVM,  # <-- We need this for feature selection
        RandomForest, OneVsRestXGBoost, 
        calculate_f1_score
    )
except ImportError:
    print("Error: 'algorithms.py' not found or is missing a model (like optimized KNN).")
    print("Error: 'algorithms.py' not found or is missing a model.")
    sys.exit(1)


@@ -42,70 +43,97 @@ def read_data(trainfile, validationfile):
