import os
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import mne  # To read raw file for channel names if needed
import time
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# Import custom classifier and selection
from correlation_classification import OlfactoryEEGClassifier, select_channels_by_correlation

def run_cross_validation(data_dir,
                         n_folds=5,
                         correlation_threshold=0.95,
                         random_state_cv=42):
    """
    Run k-fold cross-validation with multiple classifiers.
    Channel selection based on correlation is performed once before cross-validation.
    Returns a dict of accuracies and fit-time summaries per classifier.
    """
    # 1) Initial load for channel selection
    print("Step 1: Initializing for channel selection...")
    init_clf = OlfactoryEEGClassifier(data_dir=data_dir, random_state=random_state_cv)
    try:
        X_raw, _, _ = init_clf.load_data()
    except Exception as e:
        print(f"Failed to load data: {e}")
        return {}

    channels = init_clf.all_channel_names_loaded_
    if not channels:
        print("Fallback: Reading channel names via MNE...")
        first = next((os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.set')), None)
        raw_info = mne.io.read_raw_eeglab(first, preload=False, verbose=False)
        channels = raw_info.ch_names

    # 2) Channel selection
    print("Step 2: Selecting channels by correlation...")
    sel_idx, sel_names = select_channels_by_correlation(
        X_raw,
        channels,
        threshold=correlation_threshold,
        verbose=False
    )
    print(f"Selected {len(sel_idx)} channels: {sel_names}")

    # 3) Load data with selected channels
    clf = OlfactoryEEGClassifier(data_dir,
                                 selected_channel_indices=sel_idx,
                                 random_state=random_state_cv)
    X, y, _ = clf.load_data()
    print(f"Step 3: Loaded data shape: {X.shape}, classes: {np.unique(y)}")

    # Prepare classifiers and timing lists
    classifiers = {
        'SVM': SVC(kernel='linear', C=0.5, random_state=random_state_cv, probability=True),
        'KNN': KNeighborsClassifier(n_neighbors=1),
        'NB': GaussianNB()
    }
    accuracies = {name: [] for name in classifiers}
    fit_times = {name: [] for name in classifiers}

    # timing lists for train/test feature extraction
    train_wavelet_decomposition_time_list = []
    train_ovr_csp_learning_times_list = []
    train_feature_extraction_from_CSP_list = []
    test_wavelet_decomposition_time_list = []
    test_feature_extraction_from_CSP_list = []

    # K-Fold cross-validation
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state_cv)
    for fold, (train_idx, test_idx) in enumerate(kf.split(X, y), 1):
        print(f"\n--- Fold {fold}/{n_folds} ---")
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Feature extraction with timings
        print("Extracting features for training data...")
        X_train_feats, y_train_proc, train_times = clf.extract_features(X_train, y_train, is_training=True)
        print("Extracting features for testing data...")
        X_test_feats, y_test_proc, test_times = clf.extract_features(
            X_test, y_test, clf.spatial_filter, is_training=False)

        # record train/test times
        train_wavelet_decomposition_time_list.append(train_times.get("wavelet_decomposition_time", 0))
        train_ovr_csp_learning_times_list.append(train_times.get("ovr_csp_learning_time", 0))
        train_feature_extraction_from_CSP_list.append(train_times.get("feature_extraction_from_CSP", 0))
        test_wavelet_decomposition_time_list.append(test_times.get("wavelet_decomposition_time", 0))
        test_feature_extraction_from_CSP_list.append(test_times.get("feature_extraction_from_CSP", 0))

        # Train and evaluate each classifier
        for name, model in classifiers.items():
            print(f"Training {name} classifier...")
            t0 = time.time()
            model.fit(X_train_feats, y_train_proc)
            t_fit = time.time() - t0
            fit_times[name].append(t_fit)
            print(f"{name} fit time: {t_fit:.2f}s")

            y_pred = model.predict(X_test_feats)
            acc = accuracy_score(y_test_proc, y_pred)
            accuracies[name].append(acc)
            print(f"{name} accuracy: {acc:.4f}")

    # Summary
    print("\n--- Cross-Validation Summary ---")
    for name in classifiers:
        if accuracies[name]:
            mean_acc = np.mean(accuracies[name])
            std_acc = np.std(accuracies[name])
            mean_time = np.mean(fit_times[name])
            print(f"{name}: Accuracy mean={mean_acc:.4f}, std={std_acc:.4f}; "
                  f"Average fit time={mean_time:.3f}s over {len(fit_times[name])} folds")
        else:
            print(f"{name}: No results.")

    # 평균 시간 출력
    print("\nTrain Phase Mean Times:")
    if train_wavelet_decomposition_time_list:
        print(
            f"  Mean Wavelet decomposition completed time: {np.mean(train_wavelet_decomposition_time_list):.3f}s")
    if train_ovr_csp_learning_times_list:
        print(
            f"  Mean OVR-CSP completed (filter learning) time: {np.mean(train_ovr_csp_learning_times_list):.3f}s")
    if train_feature_extraction_from_CSP_list:
        print(
            f"  Mean Feature extraction from CSP (apply filters & log-var) completed time: {np.mean(train_feature_extraction_from_CSP_list):.3f}s")

    print("\nTest Phase Mean Times:")
    if test_wavelet_decomposition_time_list:
        print(
            f"  Mean Wavelet decomposition completed time: {np.mean(test_wavelet_decomposition_time_list):.3f}s")
    if test_feature_extraction_from_CSP_list:
        print(
            f"  Mean Feature extraction from CSP (apply filters & log-var) completed time: {np.mean(test_feature_extraction_from_CSP_list):.3f}s")

    return {'accuracy': accuracies, 'fit_time': fit_times,
            'train_times': {
                'wavelet': train_wavelet_decomposition_time_list,
                'ovr_csp': train_ovr_csp_learning_times_list,
                'csp_feat': train_feature_extraction_from_CSP_list
            },
            'test_times': {
                'wavelet': test_wavelet_decomposition_time_list,
                'csp_feat': test_feature_extraction_from_CSP_list
            }}

if __name__ == "__main__":
    base = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base, "preprocessed_5th")
    if not os.path.isdir(data_dir):
        print(f"Data directory not found: {data_dir}")
    else:
        run_cross_validation(data_dir,
                             n_folds=5,
                             correlation_threshold=0.85,
                             random_state_cv=42)