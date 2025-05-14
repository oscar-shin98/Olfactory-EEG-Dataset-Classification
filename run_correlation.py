# main_olfactory_cv.py
import os
import numpy as np
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
# import pandas as pd # 이 파일에서 직접 사용 안 함
import mne  # To read raw file for channel names if needed

# correlation_classification.py 파일에서 클래스와 함수 가져오기
from correlation_classification import OlfactoryEEGClassifier, select_channels_by_correlation

def run_cross_validation(data_dir, n_folds=5, correlation_threshold=0.85, random_state_cv=42):
    """
    Run k-fold cross-validation on the dataset.
    Channel selection based on correlation is performed once before cross-validation.
    """
    # 1. 초기 분류기 인스턴스 생성 (채널 선택 전, 모든 채널 로드 목적)
    print("Step 1: Initializing classifier for preliminary data load (all channels)...")
    initial_classifier = OlfactoryEEGClassifier(data_dir=data_dir, random_state=random_state_cv)

    try:
        X_for_corr_selection, _, _ = initial_classifier.load_data()
    except Exception as e:
        print(f"Error during initial data load for channel selection: {e}")
        return []

    all_channel_names_from_loader = initial_classifier.all_channel_names_loaded_
    if not all_channel_names_from_loader:
        print("Warning: Channel names not retrieved from OlfactoryEEGClassifier. Attempting direct read...")
        try:
            first_file_found = None
            for f_name in os.listdir(data_dir):
                if f_name.endswith('.set'):
                    first_file_found = os.path.join(data_dir, f_name)
                    break
            if not first_file_found:
                raise FileNotFoundError("No .set files found to read channel names.")
            raw_temp = mne.io.read_raw_eeglab(first_file_found, preload=False, verbose=False)
            all_channel_names_from_loader = raw_temp.ch_names
            print(f"Successfully read channel names directly: {all_channel_names_from_loader}")
        except Exception as e:
            print(f"Critical Error: Could not determine channel names for correlation analysis: {e}")
            return []

    # 2. 상관 관계 기반 채널 선택 수행
    print("\nStep 2: Selecting channels based on correlation...")
    selected_channel_indices, selected_channel_names = select_channels_by_correlation(
        X_for_corr_selection,
        all_channel_names_from_loader,
        threshold=correlation_threshold,
        verbose=True  # 상관관계 히트맵 및 선택 과정 출력
    )
    if not selected_channel_indices:
        print("Error: No channels were selected by correlation analysis. Aborting cross-validation.")
        return []
    print(f"Selected {len(selected_channel_indices)} channels: {selected_channel_names}")

    # 3. 선택된 채널을 사용하여 최종 분류기 초기화
    print("\nStep 3: Initializing classifier with selected channels...")
    classifier = OlfactoryEEGClassifier(data_dir,
                                        selected_channel_indices=selected_channel_indices,
                                        random_state=random_state_cv)

    print("Loading all data with selected channels for cross-validation...")
    try:
        X_all, y_all, _ = classifier.load_data()
    except Exception as e:
        print(f"Error during final data load with selected channels: {e}")
        return []

    if X_all.shape[0] == 0:
        print("No data loaded after channel selection. Check paths or selection criteria.")
        return []
    print(
        f"Loaded {X_all.shape[0]} samples, each with {X_all.shape[1]} selected channels, for {len(np.unique(y_all))} classes.")

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state_cv)
    accuracies = []
    fold_count = 0

    for train_idx, test_idx in kf.split(X_all, y_all):
        fold_count += 1
        print(f"\n--- Fold {fold_count}/{n_folds} ---")
        X_train, X_test = X_all[train_idx], X_all[test_idx]
        y_train, y_test = y_all[train_idx], y_all[test_idx]

        print(f"Training data shape: {X_train.shape}, Test data shape: {X_test.shape}")

        # 학습 데이터 특징 추출 및 SVM 학습
        print("Extracting features for training data...")
        X_train_features, y_train_proc = classifier.extract_features(X_train, y_train, is_training=True)
        if X_train_features.shape[0] == 0:
            print(f"Fold {fold_count} failed: No training features extracted. Skipping fold.")
            continue

        print("Training SVM classifier...")
        # SVC의 random_state도 고정하여 재현성 확보
        svm_model = SVC(kernel='linear', C=1.0, random_state=random_state_cv, probability=True)
        svm_model.fit(X_train_features, y_train_proc)
        classifier.clf = svm_model  # 학습된 모델을 classifier 객체에 저장 (필수는 아님)

        # 테스트 데이터 특징 추출 및 예측 (학습 시 생성된 spatial_filter 사용)
        print("Extracting features for testing data...")
        X_test_features, y_test_proc = classifier.extract_features(X_test, y_test, classifier.spatial_filter,
                                                                   is_training=False)
        if X_test_features.shape[0] == 0:
            print(f"Fold {fold_count} failed: No testing features extracted. Skipping fold.")
            continue

        if X_train_features.shape[1] != X_test_features.shape[1]:
            print(
                f"Fold {fold_count} Warning: Mismatch in feature dimensions between train ({X_train_features.shape[1]}) and test ({X_test_features.shape[1]}). Skipping prediction.")
            continue

        y_pred = svm_model.predict(X_test_features)
        accuracy = accuracy_score(y_test_proc, y_pred)
        print(f"Fold {fold_count} accuracy: {accuracy:.4f}")
        accuracies.append(accuracy)

    if accuracies:
        print("\n--- Cross-Validation Results ---")
        print(f"Accuracies for each fold: {accuracies}")
        print(f"Mean accuracy: {np.mean(accuracies):.4f}")
        print(f"Standard deviation of accuracy: {np.std(accuracies):.4f}")
    else:
        print("\n--- Cross-Validation Failed or Incomplete ---")
        if fold_count > 0:
            print("No accuracy results to report, possibly due to feature extraction issues in all folds.")
        else:
            print("Cross-validation did not run any folds.")
    return accuracies


if __name__ == "__main__":
    # 데이터 디렉토리 설정 (실제 경로로 수정 필요)
    # 예시: data_dir = "D:/EEG_data/preprocessed_5th"
    #       data_dir = "/mnt/d/EEG_data/preprocessed_5th"
    #       data_dir = "./preprocessed_5th" # 스크립트와 같은 위치에 있는 경우

    current_script_path = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_script_path, "preprocessed_5th")  # 스크립트 폴더 내의 preprocessed_5th 폴더로 가정

    # 데이터 디렉토리가 실제로 존재하는지 확인
    if not os.path.isdir(data_dir):
        print(f"Error: Data directory not found at '{data_dir}'")
        print(
            "Please ensure the 'preprocessed_5th' folder exists in the same directory as the script, or update 'data_dir' path.")
    else:
        print(f"Using data directory: {data_dir}")

        # 상관 관계 임계값 설정 (0.0 ~ 1.0 사이, 높을수록 적은 채널 제거)
        # 실험을 통해 최적값 탐색 (예: 0.8, 0.85, 0.9, 0.95)
        correlation_thresh = 0.90

        # 재현성을 위한 random_state 값
        global_random_state = 42

        print(
            f"\n=== Running K-Fold Cross-Validation with Correlation-Based Channel Selection (Threshold: {correlation_thresh}) ===")
        cv_accuracies = run_cross_validation(data_dir,
                                             n_folds=5,  # K-Fold의 K 값
                                             correlation_threshold=correlation_thresh,
                                             random_state_cv=global_random_state)

    print("\nDone!")