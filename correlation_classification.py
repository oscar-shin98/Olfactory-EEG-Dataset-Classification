# olfactory_eeg_classification.py
import os
import numpy as np
import pandas as pd
import pywt
from sklearn.svm import SVC
# from sklearn.metrics import accuracy_score, confusion_matrix, classification_report # 이 파일에서 직접 사용 안 함
# from sklearn.model_selection import train_test_split # 이 파일에서 직접 사용 안 함
from scipy.linalg import eig
import matplotlib.pyplot as plt
import seaborn as sns
import time
import mne


def select_channels_by_correlation(X_all_data, all_channel_names, threshold, verbose=True):
    """
    EEG 채널 간의 상관 관계를 기반으로 채널을 선택합니다.

    Parameters:
    -----------
    X_all_data : numpy.ndarray
        전체 EEG 데이터 (n_samples, n_channels, n_times)
    all_channel_names : list of str
        모든 채널의 이름 리스트
    threshold : float
        상관 관계 절댓값이 이 임계값을 초과하면 하나의 채널을 제거 대상으로 간주합니다.
    verbose : bool
        선택 과정을 출력할지 여부

    Returns:
    --------
    selected_indices : list
        선택된 채널의 인덱스 리스트
    selected_names : list
        선택된 채널의 이름 리스트
    """
    if X_all_data.ndim != 3 or X_all_data.shape[1] < 2:
        print("Warning: Not enough data or channels to perform correlation-based selection. Using all channels.")
        return list(range(X_all_data.shape[1])), all_channel_names if all_channel_names else [f"Ch{i + 1}" for i in
                                                                                              range(
                                                                                                  X_all_data.shape[1])]

    n_samples, n_channels, n_times = X_all_data.shape
    if not all_channel_names or len(all_channel_names) != n_channels:
        print(f"Warning: Channel names not provided or mismatch. Using generic names for {n_channels} channels.")
        all_channel_names = [f"Ch{i + 1}" for i in range(n_channels)]

    data_for_corr = X_all_data.transpose(1, 0, 2).reshape(n_channels, -1)
    corr_matrix = np.corrcoef(data_for_corr)

    if verbose:
        # plt.figure(figsize=(max(8, n_channels / 2), max(6, n_channels / 2.5)))  # 채널 수에 따라 크기 조절
        # sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', xticklabels=all_channel_names,
        #             yticklabels=all_channel_names)
        # plt.title("Channel Correlation Matrix")
        # plt.tight_layout()
        # plt.show()

        print(f"\nChannels pairs with |corr| > {threshold}:")
        for i in range(n_channels):
            for j in range(i + 1, n_channels):
                if abs(corr_matrix[i, j]) > threshold:
                    print(f"  {all_channel_names[i]}-{all_channel_names[j]}: {corr_matrix[i, j]:.3f}")


    #-------------------------------------------------------------------채널 제거 시 주석 해제
    channels_to_keep_indices = list(range(n_channels))
    channels_to_remove_indices = []
    channels_to_remove_indices = set()  # 중복 제거를 위해 set 사용

    for i in range(n_channels):
        if i in channels_to_remove_indices:
            continue
        for j in range(i + 1, n_channels):
            if j in channels_to_remove_indices:
                continue
            if abs(corr_matrix[i, j]) > threshold:
                # i, j 각각의 평균 상관계수 계산 (자기 자신은 제외)
                avg_corr_i = np.mean([abs(corr_matrix[i, k]) for k in range(n_channels) if k != i])
                avg_corr_j = np.mean([abs(corr_matrix[j, k]) for k in range(n_channels) if k != j])

                if avg_corr_i > avg_corr_j:
                    channels_to_remove_indices.add(i)
                    if verbose:
                        print(
                            f"Channel '{all_channel_names[i]}' (idx {i}) marked for removal due to higher average correlation ({avg_corr_i:.2f}) than '{all_channel_names[j]}' (idx {j}): {avg_corr_j:.2f}"
                        )
                    break  # i가 제거됐으므로 더 비교할 필요 없음
                else:
                    channels_to_remove_indices.add(j)
                    if verbose:
                        print(
                            f"Channel '{all_channel_names[j]}' (idx {j}) marked for removal due to higher average correlation ({avg_corr_j:.2f}) than '{all_channel_names[i]}' (idx {i}): {avg_corr_i:.2f}"
                        )

    # for i in range(n_channels):
    #     if i in channels_to_remove_indices:
    #         continue
    #     for j in range(i + 1, n_channels):
    #         if j in channels_to_remove_indices:
    #             continue
    #         if abs(corr_matrix[i, j]) > threshold:
    #             channels_to_remove_indices.append(j)
    #             if verbose:
    #                 print(
    #                     f"Channel '{all_channel_names[j]}' (idx {j}) marked for removal due to high correlation with '{all_channel_names[i]}' (idx {i}): {corr_matrix[i, j]:.2f}")

    final_selected_indices = [idx for idx in channels_to_keep_indices if idx not in channels_to_remove_indices]
    final_selected_names = [all_channel_names[i] for i in final_selected_indices]

    if verbose:
        print(f"\nOriginal number of channels: {n_channels}")
        print(f"Number of channels marked for removal: {len(channels_to_remove_indices)}")
        print(f"Number of channels selected: {len(final_selected_indices)}")
        print(f"Selected channel indices: {final_selected_indices}")
        # print(f"Selected channel names: {final_selected_names}") # 상세 출력은 주석 처리 가능

    if not final_selected_indices:
        print("Warning: All channels were marked for removal. Keeping the first channel by default to prevent errors.")
        return [0], [all_channel_names[0]]

    return final_selected_indices, final_selected_names
    # ------------------------------------------------------------------

    #-------------------------------------------------------------------채널 30개 유지 시 주석 해제
    # final_selected_indices = list(range(n_channels))
    # final_selected_names = all_channel_names.copy()
    # print(final_selected_indices)
    # print(final_selected_names)
    # if verbose:
    #     print(f"\nOriginal number of channels: {n_channels}")
    #     print("채널 제거 로직 비활성화 — 모든 채널 유지합니다.")
    #     print(f"Selected channel indices: {final_selected_indices}")
    #
    #return final_selected_indices, final_selected_names
    #------------------------------------------------------------------

    # 채널 5개 쓸 때 [FP1, CPZ, FZ, FP2, P3] [3, 28, 5, 7, 4] <--- 이거 주석처리
    #return [3,4,5,7,28], ['FP1','P3','FZ','FP2','CPZ']



class OlfactoryEEGClassifier:
    def __init__(self, data_dir, wavelet='db4', wavelet_level=5, selected_channel_indices=None, random_state=42):
        self.data_dir = data_dir
        self.wavelet = wavelet
        self.wavelet_level = wavelet_level
        self.selected_channel_indices = selected_channel_indices
        self.random_state = random_state
        self.classes = [chr(i) for i in range(ord('A'), ord('N'))]  # A to M (13 classes)
        self.clf = None
        self.spatial_filter = None
        self.all_channel_names_loaded_ = None  # 로드된 원본 채널 이름 저장용

    def load_data(self):
        X_all = []
        y_all = []
        subjects = []

        print(f"Data directory: {self.data_dir}")
        all_files = [f for f in os.listdir(self.data_dir) if f.endswith('.set')]
        if not all_files:
            raise FileNotFoundError(f"No .set files found in directory: {self.data_dir}")
        print(f"Found {len(all_files)} .set files.")

        for file_idx, file in enumerate(all_files):
            file_path = os.path.join(self.data_dir, file)
            parts = file.split('_')
            if len(parts) < 3:
                print(f"Unexpected filename format, skipping: {file}")
                continue
            subject_label = parts[0].strip()
            class_label = parts[1].strip().upper()

            try:
                raw = mne.io.read_raw_eeglab(file_path, preload=True, verbose=False)
                eeg_data_original = raw.get_data()

                if file_idx == 0:
                    # self.all_channel_names_loaded_ = raw.ch_names
                    self.all_channel_names_loaded_ = ['CP3', 'FC3', 'FCZ', 'FP1', 'P3', 'FZ', 'CZ', 'FP2',
                                                      'P4', 'F8', 'OZ', 'PZ', 'TP8', 'F7', 'F4', 'F3',
                                                      'CP4', 'T6', 'FC4', 'T4', 'C3', 'FT7', 'O2', 'C4',
                                                      'TP7', 'T5', 'O1', 'T3', 'CPZ', 'FT8']
                    print(
                        f"Original channel names from first file ({len(self.all_channel_names_loaded_)} channels): {self.all_channel_names_loaded_}")
                    if self.selected_channel_indices is not None:
                        print(f"Using pre-selected channel indices: {self.selected_channel_indices}")

                eeg_data_processed = eeg_data_original
                if self.selected_channel_indices is not None:
                    valid_indices = [idx for idx in self.selected_channel_indices if idx < eeg_data_original.shape[0]]
                    if len(valid_indices) != len(self.selected_channel_indices):
                        print(
                            f"Warning: Some selected channel indices were out of bounds for file {file}. Using valid subset: {valid_indices}")
                    if not valid_indices:
                        print(
                            f"Error: No valid selected channels for file {file} from {self.selected_channel_indices}. Skipping this file.")
                        continue
                    eeg_data_processed = eeg_data_original[valid_indices, :]

                if eeg_data_processed.size == 0:
                    print(f"Empty data after processing for file: {file_path}. Skipping.")
                else:
                    X_all.append(eeg_data_processed)
                    y_all.append(ord(class_label) - ord('A'))
                    subjects.append(subject_label)

            except Exception as e:
                print(f"Error reading or processing {file_path}: {e}")

        if not X_all:
            raise ValueError("No data loaded successfully. Check file paths, formats, or channel selection logic.")

        print(f"Total samples loaded: {len(X_all)}")
        if X_all:
            print(f"Number of channels in loaded data: {X_all[0].shape[0]}")
        return np.array(X_all), np.array(y_all), subjects

    def apply_wavelet_decomposition(self, eeg_data):
        wavelet_features = []
        for channel_idx in range(eeg_data.shape[0]):
            coeffs = pywt.wavedec(eeg_data[channel_idx, :], self.wavelet, level=self.wavelet_level)
            # cD_n, cD_{n-1}, ..., cD_1.  cD2 is coeffs[-2] if level >= 2
            if self.wavelet_level >= 2:
                target_coeffs = coeffs[-2]  # cD2
            else:  # level 1
                target_coeffs = coeffs[-1]  # cD1
            wavelet_features.append(target_coeffs)
        return np.array(wavelet_features)

    def compute_cd2_matrix(self, wavelet_features_all):
        # This function assumes wavelet_features_all is already the collection of desired coefficients (e.g., cD2)
        # from apply_wavelet_decomposition. It's essentially a pass-through or format conversion if needed.
        return np.array(wavelet_features_all)

    def apply_ovr_csp(self, X_train_cd_coeffs, y_train, n_components_ratio=0.25):
        n_classes = len(np.unique(y_train))
        n_samples, n_channels, n_cd_features = X_train_cd_coeffs.shape

        # OVR-CSP 필터 갯수 비율로 정할 때 주석 해제
        # n_components_csp = max(1, int(n_channels * n_components_ratio))
        # if 2 * n_components_csp > n_channels:
        #     n_components_csp = n_channels // 2
        # if n_components_csp == 0 and n_channels > 0:
        #     n_components_csp = 1  # Ensure at least one component if possible
        # if n_channels == 0:
        #     raise ValueError("Cannot apply CSP with 0 channels.")

        ## OVR-CSP 필터의 갯수는 채널의 절반으로 설정 0522
        n_components_csp = n_channels // 2

        print(
            f"CSP using {n_components_csp} components per side (total up to {2 * n_components_csp} filters for {n_channels} channels).")
        spatial_filters = {}

        for class_idx in range(n_classes):
            cov_target = np.zeros((n_channels, n_channels))
            cov_non_target = np.zeros((n_channels, n_channels))
            n_target, n_non_target = 0, 0

            ## 공분산 행렬 계산 (식 5)
            for i in range(n_samples):
                # X_train_cd_coeffs[i] is (n_channels, n_cd_features)
                cov = np.cov(X_train_cd_coeffs[i])
                if np.trace(cov) == 0:  # Avoid division by zero if trace is zero
                    # print(f"Warning: Trace of covariance is zero for sample {i}, class {y_train[i]}. Using identity matrix.")
                    # cov = np.eye(n_channels) # Or skip this sample's covariance
                    continue  # Skip this sample if covariance is ill-defined for CSP
                else:
                    cov /= np.trace(cov)

                if y_train[i] == class_idx:
                    cov_target += cov
                    n_target += 1
                else:
                    cov_non_target += cov
                    n_non_target += 1

            if n_target == 0:
                cov_target = np.eye(n_channels); print(f"Warning: No target samples for class {class_idx} in CSP.")
            else:
                cov_target /= n_target
            if n_non_target == 0:
                cov_non_target = np.eye(n_channels); print(
                    f"Warning: No non-target samples for class {class_idx} in CSP.")
            else:
                cov_non_target /= n_non_target

            C = cov_target + cov_non_target
            # eigvals, eigvecs = eig(C)  # Use eig for potentially non-symmetric, though cov should be symmetric
            # eigvals = np.real(eigvals)  # Ensure real
            # eigvecs = np.real(eigvecs)
            eigvals, eigvecs = np.linalg.eigh(C)
            idx = np.argsort(eigvals)[::-1]
            eigvals = eigvals[idx]
            eigvecs = eigvecs[:, idx]

            epsilon = 1e-9
            valid_eigvals = np.where(eigvals > epsilon, eigvals, epsilon)
            W_transform_matrix = np.diag(1.0 / np.sqrt(valid_eigvals)).dot(eigvecs.T)

            S1 = W_transform_matrix.dot(cov_target).dot(W_transform_matrix.T)
            # eigvals_S1, eigvecs_S1 = eig(S1)
            # eigvals_S1 = np.real(eigvals_S1)
            # eigvecs_S1 = np.real(eigvecs_S1)
            eigvals_S1, eigvecs_S1 = np.linalg.eigh(S1)
            idx_S1 = np.argsort(eigvals_S1)[::-1]
            eigvecs_S1 = eigvecs_S1[:, idx_S1]
            Q = eigvecs_S1.T.dot(W_transform_matrix)

            current_n_components = min(n_components_csp, Q.shape[0] // 2 if Q.shape[0] > 1 else Q.shape[0])
            if current_n_components == 0 and Q.shape[0] > 0:
                spatial_filter_for_class = Q[:Q.shape[0], :]
            elif Q.shape[0] >= 2 * current_n_components and current_n_components > 0:
                spatial_filter_for_class = np.vstack((Q[:current_n_components, :], Q[-current_n_components:, :]))
            elif Q.shape[0] > 0:  # Not enough for 2*current_n_components, take what's available
                spatial_filter_for_class = Q[:Q.shape[0], :]
            else:
                print(f"Warning: Not enough components for CSP for class {class_idx}. Using identity matrix.")
                spatial_filter_for_class = np.eye(n_channels)
            spatial_filters[class_idx] = spatial_filter_for_class
        self.spatial_filter = spatial_filters
        return spatial_filters

    def extract_features(self, X_data, y_data, spatial_filters_to_use=None, is_training=True):

        # 실행 시간을 저장할 딕셔너리
        times = {
            "wavelet_decomposition_time": 0.0,
            "ovr_csp_learning_time": 0.0,  # CSP 필터 학습 시간 (훈련 시에만)
            "feature_extraction_from_CSP": 0.0  # 필터 적용 및 최종 특징 추출 시간
        }

        n_samples_data = X_data.shape[0]
        if n_samples_data == 0:
            return np.array([]), np.array([])

        n_unique_classes = len(np.unique(y_data))
        print(
            f"Extracting features for {n_samples_data} samples, {X_data.shape[1]} channels each. Training: {is_training}")

        wavelet_coeffs_all_samples = []
        start_time = time.time()
        for i in range(n_samples_data):
            wf = self.apply_wavelet_decomposition(X_data[i])
            wavelet_coeffs_all_samples.append(wf)
        times["wavelet_decomposition_time"] = time.time() - start_time
        print(f'Wavelet decomposition completed in {times["wavelet_decomposition_time"]:.2f}s')

        # cd_coeffs_matrices shape: (n_samples, n_channels, n_cd_features)
        cd_coeffs_matrices = self.compute_cd2_matrix(np.array(wavelet_coeffs_all_samples))

        current_spatial_filters = spatial_filters_to_use
        if is_training:
            print("Applying OVR-CSP spatial filtering (training)...")
            start_time = time.time()
            current_spatial_filters = self.apply_ovr_csp(cd_coeffs_matrices, y_data)
            self.spatial_filter = current_spatial_filters  # Store filters from training
            times["ovr_csp_learning_time"] = time.time() - start_time
            print(f'OVR-CSP completed in {times["ovr_csp_learning_time"]:.2f}s')

        if current_spatial_filters is None or not current_spatial_filters:
            print("Error: Spatial filters are not available. Extracting basic log-variance features per channel.")
            features_list = []
            num_dummy_filters_per_class = X_data.shape[1]  # Use number of channels as dummy filter count
            for cd_matrix_sample in cd_coeffs_matrices:

                # 로그 정규화 분산
                var_vector = np.var(cd_matrix_sample, axis=1)
                sum_var = np.sum(var_vector)
                if sum_var == 0:
                    norm_var = np.zeros_like(var_vector)
                else:
                    norm_var = var_vector / sum_var
                epsilon = 1e-9
                log_var_features = np.log(norm_var + epsilon)

                # var_vector = np.var(cd_matrix_sample, axis=1)
                # log_var_features = var_vector  # 변수 이름은 그대로 두거나 feature_vector 등으로 변경 가능

                # Repeat for each class to match expected feature dimension if CSP was used
                repeated_features = np.tile(log_var_features, n_unique_classes)[
                                    :num_dummy_filters_per_class * n_unique_classes]
                features_list.append(repeated_features)
            return np.array(features_list), y_data

        features_list = []
        start_time = time.time()
        # Determine expected feature length from the first valid filter
        expected_filters_per_class = 0
        for class_idx_key in sorted(current_spatial_filters.keys()):
            if current_spatial_filters[class_idx_key].size > 0:
                expected_filters_per_class = current_spatial_filters[class_idx_key].shape[0]
                break
        if expected_filters_per_class == 0:  # Fallback if all filters are empty
            expected_filters_per_class = X_data.shape[1]  # Number of channels
            print(
                f"Warning: All CSP filters are empty. Using {expected_filters_per_class} as fallback feature count per class.")

        for cd_matrix_sample in cd_coeffs_matrices:  # cd_matrix_sample: (n_channels, n_cd_features)
            sample_total_features = []
            for class_idx_key in range(n_unique_classes):  # Iterate 0 to num_classes-1
                log_var_class_features = np.zeros(expected_filters_per_class)  # Initialize with zeros
                if class_idx_key in current_spatial_filters and current_spatial_filters[class_idx_key].size > 0:
                    filter_for_class = current_spatial_filters[class_idx_key]  # (n_csp_filters, n_channels)

                    if filter_for_class.shape[1] != cd_matrix_sample.shape[0]:
                        print(
                            f"Warning: Mismatch in spatial filter channels ({filter_for_class.shape[1]}) and data channels ({cd_matrix_sample.shape[0]}) for class {class_idx_key}.")
                        # Features will remain zeros as initialized
                    else:
                        projected_data = np.dot(filter_for_class, cd_matrix_sample)  # (n_csp_filters, n_cd_features)
                        var_vector = np.var(projected_data, axis=1)  # (n_csp_filters,)
                        sum_var = np.sum(var_vector)
                        if sum_var == 0:
                            norm_var = np.zeros_like(var_vector)
                        else:
                            norm_var = var_vector / sum_var
                        epsilon = 1e-9
                        current_log_var = np.log(norm_var + epsilon)
                        # Ensure consistent length
                        len_to_fill = min(len(current_log_var), expected_filters_per_class)
                        log_var_class_features[:len_to_fill] = current_log_var[:len_to_fill]
                else:
                    print(
                        f"Warning: No spatial filter for class {class_idx_key} or filter is empty. Using zero features for this class component.")
                sample_total_features.extend(log_var_class_features)
            features_list.append(sample_total_features)
        times["feature_extraction_from_CSP"] = time.time() - start_time
        print(f'Feature extraction from CSP completed in {times["feature_extraction_from_CSP"]:.2f}s')
        return np.array(features_list), y_data, times