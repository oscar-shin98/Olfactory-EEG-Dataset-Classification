# OPPD.py
import os
import numpy as np
import pandas as pd
import pywt
import time
import mne
import scipy.io  # .mat 파일을 읽기 위해 추가
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import eig

def select_channels_by_correlation(X_all_data, all_channel_names, threshold, verbose=True):
    """상관 관계 기반 채널 선택 함수"""
    if X_all_data.ndim != 3 or X_all_data.shape[1] < 2:
        print("Warning: 상관관계 기반 채널 선택을 수행하기에 데이터/채널이 부족하여 모든 채널을 사용합니다.")
        return list(range(X_all_data.shape[1])), all_channel_names if all_channel_names else [f"Ch{i + 1}" for i in
                                                                                              range(
                                                                                                  X_all_data.shape[1])]

    n_samples, n_channels, n_times = X_all_data.shape
    if not all_channel_names or len(all_channel_names) != n_channels:
        all_channel_names = [f"Ch{i + 1}" for i in range(n_channels)]

    data_for_corr = X_all_data.transpose(1, 0, 2).reshape(n_channels, -1)
    corr_matrix = np.corrcoef(data_for_corr)

    if verbose:
        print(f"\n상관계수 절댓값이 {threshold}를 초과하는 채널 쌍:")
        for i in range(n_channels):
            for j in range(i + 1, n_channels):
                if abs(corr_matrix[i, j]) > threshold:
                    print(f"  {all_channel_names[i]}-{all_channel_names[j]}: {corr_matrix[i, j]:.3f}")

    channels_to_keep_indices = list(range(n_channels))
    channels_to_remove_indices = set()

    for i in range(n_channels):
        if i in channels_to_remove_indices: continue
        for j in range(i + 1, n_channels):
            if j in channels_to_remove_indices: continue
            if abs(corr_matrix[i, j]) > threshold:
                channels_to_remove_indices.add(j)
                if verbose:
                    print(
                        f"채널 '{all_channel_names[j]}' 제거 예정 (높은 상관관계: '{all_channel_names[i]}', 값: {corr_matrix[i, j]:.2f})")

    final_selected_indices = [idx for idx in channels_to_keep_indices if idx not in channels_to_remove_indices]
    final_selected_names = [all_channel_names[i] for i in final_selected_indices]

    if verbose:
        print(f"\n원본 채널 수: {n_channels}")
        print(f"제거 대상 채널 수: {len(channels_to_remove_indices)}")
        print(f"최종 선택된 채널 수: {len(final_selected_indices)}")

    if not final_selected_indices:
        print("Warning: 모든 채널이 제거 대상으로 표시되어, 오류 방지를 위해 첫 번째 채널을 기본으로 유지합니다.")
        return [0], [all_channel_names[0]]

    return final_selected_indices, final_selected_names

class OlfactoryEEGClassifier:
    def __init__(self, data_dir, oppd_condition='eyes_open',
                 wavelet='db4', wavelet_level=5, selected_channel_indices=None, random_state=42):
        """OPPD 데이터셋 전용 분류기 초기화"""
        self.data_dir = data_dir
        self.oppd_condition = oppd_condition
        self.wavelet = wavelet
        self.wavelet_level = wavelet_level
        self.selected_channel_indices = selected_channel_indices
        self.random_state = random_state
        self.all_channel_names_loaded_ = None
        self.spatial_filter = None
        self.classes = [1, 2, 3, 4]  # 4 odors

    def load_data(self):
        """OPPD 데이터셋(.mat) 로딩 로직"""
        X_all, y_all, subjects = [], [], []
        # 'subject'로 시작하는 디렉토리만 대상으로 하도록 조건 추가
        subject_dirs = sorted([
            d for d in os.listdir(self.data_dir)
            if os.path.isdir(os.path.join(self.data_dir, d)) and d.lower().startswith('subject')
        ])

        print(f"'OPPD' 데이터셋 로딩 중... (피험자: {len(subject_dirs)}명, 조건: {self.oppd_condition})")

        for subj_dir in subject_dirs:
            condition_path = os.path.join(self.data_dir, subj_dir, self.oppd_condition)
            if not os.path.exists(condition_path):
                print(f"경고: '{condition_path}' 경로를 찾을 수 없어 건너뜁니다.")
                continue

            mat_files = sorted([f for f in os.listdir(condition_path) if f.endswith('.mat')])
            for mat_file in mat_files:
                try:
                    class_label = int(''.join(filter(str.isdigit, mat_file))) - 1  # 0-indexed
                    mat_path = os.path.join(condition_path, mat_file)
                    mat_contents = scipy.io.loadmat(mat_path)

                    eeg_data_original = mat_contents['X_event'].astype(np.float64)
                    eeg_data_trials = np.transpose(eeg_data_original, (2, 1, 0))  # (시간,채널,시행)->(시행,채널,시간)

                    if not self.all_channel_names_loaded_:
                        n_channels = eeg_data_trials.shape[1]
                        self.all_channel_names_loaded_ = [f"Ch{i + 1}" for i in range(n_channels)]
                        print(f"로드된 원본 채널 수: {n_channels}")

                    eeg_data_processed = eeg_data_trials
                    if self.selected_channel_indices is not None:
                        eeg_data_processed = eeg_data_trials[:, self.selected_channel_indices, :]

                    for trial_idx in range(eeg_data_processed.shape[0]):
                        X_all.append(eeg_data_processed[trial_idx, :, :])
                        y_all.append(class_label)
                        subjects.append(subj_dir)

                except Exception as e:
                    print(f"{mat_file} 처리 중 오류 발생: {e}")

        if not X_all:
            raise ValueError("OPPD 데이터 로딩에 실패했습니다.")

        print(f"로드된 총 샘플(시행) 수: {len(X_all)}")
        return np.array(X_all), np.array(y_all), subjects

    def apply_wavelet_decomposition(self, eeg_data):
        wavelet_features = []
        for channel_idx in range(eeg_data.shape[0]):
            coeffs = pywt.wavedec(eeg_data[channel_idx, :], self.wavelet, level=self.wavelet_level)
            target_coeffs = coeffs[-2] if self.wavelet_level >= 2 else coeffs[-1]
            wavelet_features.append(target_coeffs)
        return np.array(wavelet_features)

    def apply_ovr_csp(self, X_train_cd_coeffs, y_train):
        n_classes = len(np.unique(y_train))
        n_samples, n_channels, n_cd_features = X_train_cd_coeffs.shape
        n_components_csp = n_channels // 2
        print(f"CSP: 채널 {n_channels}개에 대해 {n_components_csp}개 컴포넌트 사용")
        spatial_filters = {}

        for class_idx in range(n_classes):
            cov_target = np.zeros((n_channels, n_channels))
            cov_non_target = np.zeros((n_channels, n_channels))
            n_target, n_non_target = 0, 0

            for i in range(n_samples):
                cov = np.cov(X_train_cd_coeffs[i])
                trace = np.trace(cov)
                if trace == 0: continue
                cov /= trace

                if y_train[i] == class_idx:
                    cov_target += cov;
                    n_target += 1
                else:
                    cov_non_target += cov;
                    n_non_target += 1

            cov_target = cov_target / n_target if n_target > 0 else np.eye(n_channels)
            cov_non_target = cov_non_target / n_non_target if n_non_target > 0 else np.eye(n_channels)

            eigvals, eigvecs = eig(cov_target, cov_target + cov_non_target)
            eigvals, eigvecs = np.real(eigvals), np.real(eigvecs)
            idx = np.argsort(eigvals)[::-1]
            eigvecs = eigvecs[:, idx]

            current_n_components = min(n_components_csp,
                                       eigvecs.shape[1] // 2 if eigvecs.shape[1] > 1 else eigvecs.shape[1])
            if current_n_components > 0:
                spatial_filters[class_idx] = np.vstack(
                    (eigvecs[:, :current_n_components].T, eigvecs[:, -current_n_components:].T))
            else:
                spatial_filters[class_idx] = np.eye(n_channels)

        self.spatial_filter = spatial_filters
        return spatial_filters

    def extract_features(self, X_data, y_data, spatial_filters_to_use=None, is_training=True):
        times = {}
        n_samples_data = X_data.shape[0]
        if n_samples_data == 0: return np.array([]), np.array([]), {}

        start_time = time.time()
        wavelet_coeffs_all_samples = [self.apply_wavelet_decomposition(X_data[i]) for i in range(n_samples_data)]
        times["wavelet_decomposition_time"] = time.time() - start_time
        cd_coeffs_matrices = np.array(wavelet_coeffs_all_samples)

        current_spatial_filters = spatial_filters_to_use
        if is_training:
            start_time = time.time()
            current_spatial_filters = self.apply_ovr_csp(cd_coeffs_matrices, y_data)
            times["ovr_csp_learning_time"] = time.time() - start_time

        if current_spatial_filters is None:
            raise ValueError("CSP 필터가 없습니다. 훈련 데이터로 먼저 학습시켜야 합니다.")

        start_time = time.time()
        features_list = []
        n_unique_classes = len(np.unique(y_data))
        for cd_matrix_sample in cd_coeffs_matrices:
            sample_total_features = []
            for class_idx in range(n_unique_classes):
                filter_for_class = current_spatial_filters[class_idx]
                projected_data = np.dot(filter_for_class, cd_matrix_sample)
                var_vector = np.var(projected_data, axis=1)
                sum_var = np.sum(var_vector)
                norm_var = var_vector / sum_var if sum_var != 0 else np.zeros_like(var_vector)
                log_var_features = np.log(norm_var + 1e-9)
                sample_total_features.extend(log_var_features)
            features_list.append(sample_total_features)

        times["feature_extraction_from_CSP"] = time.time() - start_time
        return np.array(features_list), y_data, times

def run_cross_validation(data_dir, oppd_condition='eyes_open', n_folds=5,
                         correlation_threshold=0.95, random_state_cv=42):
    """OPPD 데이터셋에 대한 K-겹 교차 검증 실행"""
    print(f"\n===== OPPD 데이터셋 교차 검증 시작 (조건: {oppd_condition}) =====")

    init_clf = OlfactoryEEGClassifier(data_dir=data_dir, oppd_condition=oppd_condition, random_state=random_state_cv)
    X_raw, _, _ = init_clf.load_data()
    channels = init_clf.all_channel_names_loaded_

    sel_idx, sel_names = select_channels_by_correlation(X_raw, channels, threshold=correlation_threshold, verbose=True)
    print(f"선택된 채널 ({len(sel_idx)}개): {sel_names}")

    clf = OlfactoryEEGClassifier(data_dir=data_dir, oppd_condition=oppd_condition,
                                 selected_channel_indices=sel_idx, random_state=random_state_cv)
    X, y, _ = clf.load_data()
    print(f"최종 로드된 데이터 형태: {X.shape}, 클래스: {np.unique(y)}")

    classifiers = {
        'SVM': SVC(kernel='linear', C=0.5, random_state=random_state_cv, probability=True),
        'KNN': KNeighborsClassifier(n_neighbors=1),
        'NB': GaussianNB()
    }
    accuracies, fit_times, all_y_true, all_y_pred = ({n: [] for n in classifiers} for _ in range(4))
    train_wavelet_times, train_csp_learn_times, train_csp_feature_times = [], [], []
    test_wavelet_times, test_csp_feature_times = [], []

    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state_cv)
    for fold, (train_idx, test_idx) in enumerate(kf.split(X, y), 1):
        print(f"\n--- Fold {fold}/{n_folds} ---")
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        X_train_feats, y_train_proc, train_times = clf.extract_features(X_train, y_train, is_training=True)
        X_test_feats, y_test_proc, test_times = clf.extract_features(X_test, y_test, clf.spatial_filter,
                                                                     is_training=False)

        train_wavelet_times.append(train_times.get("wavelet_decomposition_time", 0))
        train_csp_learn_times.append(train_times.get("ovr_csp_learning_time", 0))
        train_csp_feature_times.append(train_times.get("feature_extraction_from_CSP", 0))
        test_wavelet_times.append(test_times.get("wavelet_decomposition_time", 0))
        test_csp_feature_times.append(test_times.get("feature_extraction_from_CSP", 0))

        for name, model in classifiers.items():
            t0 = time.time()
            model.fit(X_train_feats, y_train_proc)
            fit_times[name].append(time.time() - t0)
            y_pred = model.predict(X_test_feats)
            acc = accuracy_score(y_test_proc, y_pred)
            accuracies[name].append(acc)
            all_y_true[name].extend(y_test_proc)
            all_y_pred[name].extend(y_pred)
            print(f"{name} 정확도: {acc:.4f}")

    print("\n--- 교차 검증 요약 ---")
    for name in classifiers:
        mean_acc, std_acc = np.mean(accuracies[name]), np.std(accuracies[name])
        mean_time = np.mean(fit_times[name])
        print(f"{name}: 평균 정확도={mean_acc:.4f} (std={std_acc:.4f}), 평균 학습 시간={mean_time:.3f}s")

    print("\n--- 특징 추출 평균 시간 ---")
    print("Train Phase:")
    if train_wavelet_times: print(f"  - Wavelet Decomposition: {np.mean(train_wavelet_times):.3f}s")
    if train_csp_learn_times: print(f"  - OVR-CSP Filter Learning: {np.mean(train_csp_learn_times):.3f}s")
    if train_csp_feature_times: print(f"  - Feature Extraction from CSP: {np.mean(train_csp_feature_times):.3f}s")
    print("Test Phase:")
    if test_wavelet_times: print(f"  - Wavelet Decomposition: {np.mean(test_wavelet_times):.3f}s")
    if test_csp_feature_times: print(f"  - Feature Extraction from CSP: {np.mean(test_csp_feature_times):.3f}s")

    for name in classifiers:
        report = classification_report(all_y_true[name], all_y_pred[name],
                                       target_names=[str(c + 1) for c in np.unique(y)], digits=4, zero_division=0)
        print(f"\n--- {name} 상세 리포트 ---")
        print(report)

        cm = confusion_matrix(all_y_true[name], all_y_pred[name])
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y) + 1, yticklabels=np.unique(y) + 1)
        plt.title(f'[OPPD] Confusion Matrix for {name} ({oppd_condition})')
        plt.ylabel('Actual Label');
        plt.xlabel('Predicted Label')
        plt.show()

if __name__ == "__main__":
    oppd_data_dir = "./OPPD/Data"
    if not os.path.isdir(oppd_data_dir):
        print(f"경고: OPPD 데이터셋 경로를 찾을 수 없습니다: {oppd_data_dir}")
    else:
        run_cross_validation(data_dir=oppd_data_dir,
                             oppd_condition='eyes_closed',
                             n_folds=5,
                             correlation_threshold=0.90,
                             random_state_cv=42)