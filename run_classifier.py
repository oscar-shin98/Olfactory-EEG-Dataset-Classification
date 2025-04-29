import os
import numpy as np
from sklearn.model_selection import KFold
from olfactory_eeg_classification import OlfactoryEEGClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pandas as pd


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


def subject_independent_evaluation(data_dir):
    """
    Evaluate using leave-one-subject-out cross-validation based on subject information
    extracted from the .set file names.
    """
    # List all .set files in the directory.
    all_files = [f for f in os.listdir(data_dir) if f.endswith('.set')]
    # Dictionaries to store sample data per subject.
    subject_data = {}
    subject_labels = {}

    for file in all_files:
        parts = file.split('_')
        if len(parts) < 3:
            continue
        subject_label = parts[0].strip()  # e.g., "Sub. 1"
        class_label = parts[1].strip()
        try:
            file_path = os.path.join(data_dir, file)
            # Read with MNE (EEGLAB file)
            import mne
            raw = mne.io.read_raw_eeglab(file_path, preload=True, verbose=False)
            eeg_data = raw.get_data()  # shape: (n_channels, n_times)
            if eeg_data.size == 0:
                continue
            if subject_label not in subject_data:
                subject_data[subject_label] = []
                subject_labels[subject_label] = []
            subject_data[subject_label].append(eeg_data)
            subject_labels[subject_label].append(ord(class_label.upper()) - ord('A'))
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    all_subjects = list(subject_data.keys())
    print(f"Subjects found: {all_subjects}")

    subject_accuracies = {}

    for test_subject in all_subjects:
        print(f"\n--- Testing on subject: {test_subject} ---")
        X_train = []
        y_train = []
        X_test = []
        y_test = []
        # All subjects except test_subject as training data.
        for subj in all_subjects:
            if subj == test_subject:
                continue
            X_train.extend(subject_data[subj])
            y_train.extend(subject_labels[subj])
        # Use test_subject data for testing.
        X_test.extend(subject_data[test_subject])
        y_test.extend(subject_labels[test_subject])

        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_test = np.array(X_test)
        y_test = np.array(y_test)

        print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

        classifier = OlfactoryEEGClassifier(data_dir)
        X_train_features, y_train = classifier.extract_features(X_train, y_train, is_training=True)

        classifier.clf = SVC(kernel='linear', C=2.0, random_state=42)
        classifier.clf.fit(X_train_features, y_train)

        X_test_features, y_test = classifier.extract_features(X_test, y_test, classifier.spatial_filter,
                                                              is_training=False)
        y_pred = classifier.clf.predict(X_test_features)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"Accuracy for subject {test_subject}: {accuracy:.4f}")
        subject_accuracies[test_subject] = accuracy

    print("\n--- Subject-Independent Evaluation Results ---")
    for subj, acc in subject_accuracies.items():
        print(f"{subj}: {acc:.4f}")
    mean_acc = np.mean(list(subject_accuracies.values()))
    std_acc = np.std(list(subject_accuracies.values()))
    print(f"Mean accuracy: {mean_acc:.4f}")
    print(f"Standard deviation: {std_acc:.4f}")

    return subject_accuracies


if __name__ == "__main__":
    data_dir = "preprocessed_5th"  # Update this path to your processed .set file folder

    print("\n=== Running K-Fold Cross-Validation ===")
    cv_accuracies = run_cross_validation(data_dir, n_folds=5)

    print("\n=== Running Subject-Independent Evaluation ===")
    subject_accuracies = subject_independent_evaluation(data_dir)

    print("\nDone!")
