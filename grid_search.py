import os
import numpy as np
import time
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import mne  # For channel names fallback

# Import custom classifier and selection
from correlation_classification import OlfactoryEEGClassifier, select_channels_by_correlation

def run_cross_validation(data_dir,
                         n_folds=5,
                         correlation_threshold=0.95,
                         random_state_cv=42):
    """
    Run k-fold cross-validation and compute:
      1) Average train-phase times: wavelet decomposition, OVR-CSP training, CSP feature extraction
      2) SVM accuracies: for C in 2^{-10}..2^{10}, 5-fold average
      3) KNN accuracies: for k in {1,3,5}, 5-fold average
    """
    # 1) Initial load for channel selection
    init_clf = OlfactoryEEGClassifier(data_dir=data_dir, random_state=random_state_cv)
    X_raw, _, _ = init_clf.load_data()
    channels = init_clf.all_channel_names_loaded_
    if not channels:
        first = next((os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.set')), None)
        raw_info = mne.io.read_raw_eeglab(first, preload=False, verbose=False)
        channels = raw_info.ch_names

    sel_idx, sel_names = select_channels_by_correlation(
        X_raw, channels, threshold=correlation_threshold, verbose=False)

    # Load data with selected channels
    clf = OlfactoryEEGClassifier(data_dir,
                                 selected_channel_indices=sel_idx,
                                 random_state=random_state_cv)
    X, y, _ = clf.load_data()

    # Prepare hyperparameter lists
    C_list = [2**i for i in range(-10, 11)]  # SVM Cs
    k_list = [1, 3, 5]                      # KNN neighbors

    # Initialize accumulators
    svm_acc = {C: [] for C in C_list}
    knn_acc = {k: [] for k in k_list}
    wavelet_times = []
    ovr_csp_times = []
    csp_feat_times = []

    # K-Fold CV
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state_cv)
    for train_idx, test_idx in kf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Feature extraction with timing
        X_train_feats, y_train_proc, train_times = clf.extract_features(
            X_train, y_train, is_training=True)
        X_test_feats, y_test_proc, _ = clf.extract_features(
            X_test, y_test, clf.spatial_filter, is_training=False)

        wavelet_times.append(train_times.get("wavelet_decomposition_time", 0))
        ovr_csp_times.append(train_times.get("ovr_csp_learning_time", 0))
        csp_feat_times.append(train_times.get("feature_extraction_from_CSP", 0))

        # SVM for each C
        for C in C_list:
            svm = SVC(kernel='linear', C=C, random_state=random_state_cv)
            svm.fit(X_train_feats, y_train_proc)
            y_pred = svm.predict(X_test_feats)
            svm_acc[C].append(accuracy_score(y_test_proc, y_pred))

        # KNN for each k
        for k in k_list:
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(X_train_feats, y_train_proc)
            y_pred = knn.predict(X_test_feats)
            knn_acc[k].append(accuracy_score(y_test_proc, y_pred))

    # Compute averages
    avg_wavelet = np.mean(wavelet_times)
    avg_ovr_csp = np.mean(ovr_csp_times)
    avg_csp_feat = np.mean(csp_feat_times)

    avg_svm_acc = {C: np.mean(accs) for C, accs in svm_acc.items()}
    avg_knn_acc = {k: np.mean(accs) for k, accs in knn_acc.items()}

    # Print summary
    print("Train-phase average times:")
    print(f"  Wavelet decomposition: {avg_wavelet:.3f}s")
    print(f"  OVR-CSP training: {avg_ovr_csp:.3f}s")
    print(f"  CSP feature extraction: {avg_csp_feat:.3f}s")

    print("\nSVM 5-fold average accuracies by C:")
    for C, acc in avg_svm_acc.items():
        print(f"  C={C:>7}: {acc:.4f}")

    print("\nKNN 5-fold average accuracies by k:")
    for k, acc in avg_knn_acc.items():
        print(f"  k={k}: {acc:.4f}")

    return {
        'avg_times': {
            'wavelet': avg_wavelet,
            'ovr_csp': avg_ovr_csp,
            'csp_feat': avg_csp_feat
        },
        'svm_accuracy': avg_svm_acc,
        'knn_accuracy': avg_knn_acc
    }

if __name__ == "__main__":
    base = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base, "preprocessed_5th")
    results = run_cross_validation(data_dir,
                                   n_folds=5,
                                   correlation_threshold=1.0,
                                   random_state_cv=42)
    print(results)
