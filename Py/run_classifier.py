import os
import numpy as np
from sklearn.model_selection import KFold
from olfactory_eeg_classification import OlfactoryEEGClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


def run_cross_validation(data_dir, n_folds=5):
    """
    Run k-fold cross-validation on the dataset based on all .set files.
    """
    # Initialize classifier instance
    classifier = OlfactoryEEGClassifier(data_dir)

    print("Loading all data...")
    X_all, y_all, _ = classifier.load_data()
    print(f"Loaded {len(X_all)} samples with {len(np.unique(y_all))} classes")

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    accuracies = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X_all)):
        print(f"\n--- Fold {fold + 1}/{n_folds} ---")
        X_train, X_test = X_all[train_idx], X_all[test_idx]
        y_train, y_test = y_all[train_idx], y_all[test_idx]

        print("Extracting features for training data...")
        X_train_features, y_train = classifier.extract_features(X_train, y_train, is_training=True)

        print("Training SVM classifier...")
        classifier.clf = SVC(kernel='linear', C=1.0, random_state=42)
        classifier.clf.fit(X_train_features, y_train)

        print("Extracting features for testing data...")
        X_test_features, y_test = classifier.extract_features(X_test, y_test, classifier.spatial_filter,
                                                              is_training=False)

        y_pred = classifier.clf.predict(X_test_features)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Fold {fold + 1} accuracy: {accuracy:.4f}")
        accuracies.append(accuracy)

    print("\n--- Cross-Validation Results ---")
    print(f"Mean accuracy: {np.mean(accuracies):.4f}")
    print(f"Standard deviation: {np.std(accuracies):.4f}")

    return accuracies

if __name__ == "__main__":
    data_dir = "../DataSets/convert"  # Update this path to your processed .set file folder

    print("\n=== Running K-Fold Cross-Validation ===")
    cv_accuracies = run_cross_validation(data_dir, n_folds=5)

    print("\nDone!")
