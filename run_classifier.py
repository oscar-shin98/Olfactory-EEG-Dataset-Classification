import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from olfactory_eeg_classification import OlfactoryEEGClassifier
from sklearn.svm import SVC

import pywt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from scipy.linalg import eig
import matplotlib.pyplot as plt
import seaborn as sns
import time

def run_cross_validation(data_dir, n_folds=5):
    """
    Run k-fold cross-validation on the dataset
    
    Parameters:
    -----------
    data_dir : str
        Path to the dataset directory
    n_folds : int
        Number of folds for cross-validation
    """
    # Initialize the classifier
    classifier = OlfactoryEEGClassifier(data_dir)
    
    # Load all data
    print("Loading all data...")
    X_all, y_all = classifier.load_data()
    print(f"Loaded {len(X_all)} samples with {len(np.unique(y_all))} classes")
    
    # Initialize k-fold cross-validation
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # Track performances
    accuracies = []
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(X_all)):
        print(f"\n--- Fold {fold+1}/{n_folds} ---")
        
        # Split data
        X_train, X_test = X_all[train_idx], X_all[test_idx]
        y_train, y_test = y_all[train_idx], y_all[test_idx]
        
        # Extract features for training
        print("Extracting features for training data...")
        X_train_features, y_train = classifier.extract_features(X_train, y_train, is_training=True)
        
        # Train SVM
        print("Training SVM classifier...")
        classifier.clf = SVC(kernel='linear', C=1.0, random_state=42)
        classifier.clf.fit(X_train_features, y_train)
        
        # Extract features for testing
        print("Extracting features for testing data...")
        X_test_features, y_test = classifier.extract_features(X_test, y_test, 
                                                             classifier.spatial_filter, 
                                                             is_training=False)
        
        # Evaluate
        y_pred = classifier.clf.predict(X_test_features)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Fold {fold+1} accuracy: {accuracy:.4f}")
        
        # Save results
        accuracies.append(accuracy)
    
    # Print overall results
    print("\n--- Cross-Validation Results ---")
    print(f"Mean accuracy: {np.mean(accuracies):.4f}")
    print(f"Standard deviation: {np.std(accuracies):.4f}")
    
    return accuracies

def subject_independent_evaluation(data_dir):
    """
    Evaluate using leave-one-subject-out cross-validation
    
    Parameters:
    -----------
    data_dir : str
        Path to the dataset directory
    """
    # Get list of all subjects
    subjects = [f"Sub.{i}" for i in range(1, 12)]
    
    # Track performance
    subject_accuracies = {}
    
    for test_subject in subjects:
        print(f"\n--- Testing on subject: {test_subject} ---")
        
        # Initialize data containers
        X_train = []
        y_train = []
        X_test = []
        y_test = []
        
        # Classes (A through M)
        classes = [chr(i) for i in range(ord('A'), ord('N'))]
        
        # Load training data (all subjects except test_subject)
        for subject in subjects:
            if subject == test_subject:
                continue
                
            for class_label in classes:
                class_dir = os.path.join(data_dir, subject, class_label)
                if not os.path.exists(class_dir):
                    continue
                    
                for file in os.listdir(class_dir):
                    if file.endswith('.csv'):
                        file_path = os.path.join(class_dir, file)
                        eeg_data = pd.read_csv(file_path).values
                        X_train.append(eeg_data)
                        y_train.append(ord(class_label) - ord('A'))
        
        # Load test data (only test_subject)
        for class_label in classes:
            class_dir = os.path.join(data_dir, test_subject, class_label)
            if not os.path.exists(class_dir):
                continue
                
            for file in os.listdir(class_dir):
                if file.endswith('.csv'):
                    file_path = os.path.join(class_dir, file)
                    eeg_data = pd.read_csv(file_path).values
                    X_test.append(eeg_data)
                    y_test.append(ord(class_label) - ord('A'))
        
        # Convert to numpy arrays
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_test = np.array(X_test)
        y_test = np.array(y_test)
        
        print(f"Training samples: {len(X_train)}")
        print(f"Testing samples: {len(X_test)}")
        
        # Initialize classifier
        classifier = OlfactoryEEGClassifier(data_dir)
        
        # Extract features for training
        X_train_features, y_train = classifier.extract_features(X_train, y_train, is_training=True)
        
        # Train SVM
        classifier.clf = SVC(kernel='linear', C=2.0, random_state=42) # linear kernel, penalty factor = 2
        classifier.clf.fit(X_train_features, y_train)
        
        # Extract features for testing
        X_test_features, y_test = classifier.extract_features(X_test, y_test, 
                                                             classifier.spatial_filter, 
                                                             is_training=False)
        
        # Evaluate
        y_pred = classifier.clf.predict(X_test_features)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Accuracy for subject {test_subject}: {accuracy:.4f}")
        subject_accuracies[test_subject] = accuracy
    
    # Print overall results
    print("\n--- Subject-Independent Evaluation Results ---")
    for subject, acc in subject_accuracies.items():
        print(f"{subject}: {acc:.4f}")
    
    mean_acc = np.mean(list(subject_accuracies.values()))
    std_acc = np.std(list(subject_accuracies.values()))
    print(f"Mean accuracy: {mean_acc:.4f}")
    print(f"Standard deviation: {std_acc:.4f}")
    
    return subject_accuracies

if __name__ == "__main__":
    data_dir = "processed_dataset"  # Update with your dataset path
    
    print("\n=== Running K-Fold Cross-Validation ===")
    cv_accuracies = run_cross_validation(data_dir, n_folds=5)
    
    print("\n=== Running Subject-Independent Evaluation ===")
    #subject_accuracies = subject_independent_evaluation(data_dir)
    
    print("\nDone!")
