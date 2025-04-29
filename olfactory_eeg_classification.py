# import os
# import numpy as np
# import pandas as pd
# import pywt
# from sklearn.svm import SVC
# from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# from sklearn.model_selection import train_test_split
# from scipy.linalg import eig
# import matplotlib.pyplot as plt
# import seaborn as sns
# import time
# class OlfactoryEEGClassifier:
#     def __init__(self, data_dir, wavelet='db4', wavelet_level=5, test_size=0.3, random_state=42):
#         """
#         Initialize the EEG classifier for olfactory signals
#
#         Parameters:
#         -----------
#         data_dir : str
#             Directory containing the preprocessed EEG data
#         wavelet : str
#             Wavelet family to use for decomposition
#         wavelet_level : int
#             Level of wavelet decomposition
#         test_size : float
#             Proportion of the dataset to include in the test split
#         random_state : int
#             Random seed for reproducibility
#         """
#         self.data_dir = data_dir
#         self.wavelet = wavelet
#         self.wavelet_level = wavelet_level
#         self.test_size = test_size
#         self.random_state = random_state
#         self.subjects = [f"Sub. {i}" for i in range(1, 12)]
#         self.classes = [chr(i) for i in range(ord('A'), ord('N'))]  # A through M
#         self.clf = None
#         self.spatial_filter = None
#
#     def load_data(self):
#         """Load all data from directory structure"""
#         X_all = []
#         y_all = []
#
#         print(f"Data directory: {self.data_dir}")  # 데이터 폴더 경로 확인
#
#         for subject in self.subjects:
#             print(f"Processing subject: {subject}")  # 어떤 subject를 읽는지 확인
#             for class_label in self.classes:
#                 class_dir = os.path.join(self.data_dir, subject, class_label)
#                 if not os.path.exists(class_dir):
#                     print(f"Directory not found: {class_dir}")  # 폴더가 존재하지 않음
#                     continue
#
#                 csv_files = [f for f in os.listdir(class_dir) if f.endswith('.csv')]
#                 print(f"Found {len(csv_files)} files in {class_dir}")  # 몇 개의 파일을 찾았는지 확인
#
#                 for file in csv_files:
#                     file_path = os.path.join(class_dir, file)
#                     try:
#                         eeg_data = pd.read_csv(file_path, header=None).values.T
#                         print(f"File: {file_path}, Shape: {eeg_data.shape}")  # 각 파일의 데이터 형태 확인
#                         if eeg_data.shape[0] == 0 or eeg_data.shape[1] == 0:
#                             print(f"Empty file: {file_path}")  # 빈 파일이면 출력
#                         else:
#                             X_all.append(eeg_data)
#                             y_all.append(ord(class_label) - ord('A'))  # Convert letter to numeric label
#                     except Exception as e:
#                         print(f"Error reading {file_path}: {e}")  # 파일 읽기 오류 출력
#
#         print(f"Total samples loaded: {len(X_all)}")
#         return np.array(X_all), np.array(y_all)
#
#     def apply_wavelet_decomposition(self, eeg_data):
#         """
#         Apply 5-level wavelet decomposition using DWT with db4
#
#         Parameters:
#         -----------
#         eeg_data : numpy.ndarray
#             EEG data with shape (channels, time_points)
#
#         Returns:
#         --------
#         wavelet_features : numpy.ndarray
#             Wavelet coefficients
#         """
#         wavelet_features = []
#
#         for channel in range(eeg_data.shape[0]):
#             # Apply wavelet decomposition
#             coeffs = pywt.wavedec(eeg_data[channel, :], self.wavelet, level=self.wavelet_level)
#
#             # Extract approximate and detailed coefficients
#             # feature_vector = np.concatenate([coef for coef in coeffs])
#             # wavelet_features.append(feature_vector) -modify 04.04
#             # cD2 계수만 추출 (레벨 2 세부 계수)
#             cd2 = coeffs[self.wavelet_level - 1]  # 5단계 분해의 경우 coeffs[3]이어야 함
#             wavelet_features.append(cd2)
#
#         return np.array(wavelet_features)
#
#     def compute_cd2_matrix(self, wavelet_features_all):
#         """
#         Compute the cD2 Wavelet Coefficients Matrix
#
#         Parameters:
#         -----------
#         wavelet_features_all : list
#             List of wavelet features for all samples
#
#         Returns:
#         --------
#         cd2_matrices : list
#             List of cD2 matrices for all samples
#         """
#         cd2_matrices = []
#
#         for wavelet_features in wavelet_features_all:
#             # Extract the cD2 coefficients (assuming they're at a specific position in the wavelet decomposition)
#             # Note: This is a simplified implementation - you may need to adjust based on the actual paper details
#             cd2_length = len(wavelet_features[0]) // (2**(self.wavelet_level - 2))
#             cd2_start = len(wavelet_features[0]) - cd2_length
#             cd2_end = cd2_start + cd2_length
#
#             cd2_matrix = np.zeros((wavelet_features.shape[0], cd2_length))
#             for i in range(wavelet_features.shape[0]):
#                 cd2_matrix[i, :] = wavelet_features[i, cd2_start:cd2_end]
#
#             cd2_matrices.append(cd2_matrix)
#
#         return cd2_matrices
#
#     def apply_ovr_csp(self, X_train, y_train, n_components=2):
#         """
#         Apply One-vs-Rest Common Spatial Pattern (OVR-CSP) with whitening transformation.
#
#         Parameters:
#         -----------
#         X_train : numpy.ndarray
#             Training data with shape (n_samples, n_channels, n_times)
#         y_train : numpy.ndarray
#             Training labels
#         n_components : int
#             Number of CSP components to select from each end
#
#         Returns:
#         --------
#         spatial_filters : dict
#             Dictionary with class labels as keys and spatial filters as values
#         """
#         n_classes = len(np.unique(y_train))
#         n_channels = X_train[0].shape[0]
#
#         spatial_filters = {}
#
#         # For each class, compute OVR-CSP with whitening
#         for class_idx in range(n_classes):
#             # Compute mean covariance matrices for target vs. non-target
#             cov_target = np.zeros((n_channels, n_channels))
#             cov_non_target = np.zeros((n_channels, n_channels))
#
#             n_target = 0
#             n_non_target = 0
#
#             for i, X in enumerate(X_train):
#                 cov = np.cov(X)
#                 if y_train[i] == class_idx:
#                     cov_target += cov
#                     n_target += 1
#                 else:
#                     cov_non_target += cov
#                     n_non_target += 1
#
#             if n_target > 0:
#                 cov_target /= n_target
#             if n_non_target > 0:
#                 cov_non_target /= n_non_target
#
#             # Composite covariance matrix
#             C = cov_target + cov_non_target
#
#             # Whitening transformation using eigen decomposition (symmetric matrix => use np.linalg.eigh)
#             eigvals, eigvecs = np.linalg.eigh(C)
#             # Sort eigenvalues in descending order
#             idx = np.argsort(eigvals)[::-1]
#             eigvals = eigvals[idx]
#             eigvecs = eigvecs[:, idx]
#             # Compute whitening matrix: W = diag(1/sqrt(eigvals)) * U^T
#             W = np.diag(1.0 / np.sqrt(eigvals)).dot(eigvecs.T)
#
#             # Whiten the target and non-target covariance matrices
#             S1 = W.dot(cov_target).dot(W.T)
#             S2 = W.dot(cov_non_target).dot(W.T)
#
#             # Eigen decomposition on the whitened target covariance matrix S1
#             eigvals_S1, eigvecs_S1 = np.linalg.eigh(S1)
#             # Sort in descending order
#             idx_S1 = np.argsort(eigvals_S1)[::-1]
#             eigvals_S1 = eigvals_S1[idx_S1]
#             eigvecs_S1 = eigvecs_S1[:, idx_S1]
#
#             # Compute projection matrix Q = U_S^T * W, where U_S are eigenvectors of S1
#             Q = eigvecs_S1.T.dot(W)
#
#             # Select first n_components and last n_components rows to form the spatial filter
#             spatial_filter = np.vstack((Q[:n_components, :], Q[-n_components:, :]))
#             spatial_filters[class_idx] = spatial_filter
#
#         self.spatial_filter = spatial_filters
#         return spatial_filters
#
#     def extract_features(self, X_data, y_data, spatial_filters=None, is_training=True):
#         """
#         Extract features using wavelet decomposition and spatial filtering with normalized variance.
#
#         Parameters:
#         -----------
#         X_data : numpy.ndarray
#             EEG data with shape (n_samples, n_channels, n_times)
#         y_data : numpy.ndarray
#             Class labels
#         spatial_filters : dict, optional
#             Spatial filters to apply (used for test data)
#         is_training : bool
#             Whether this is training or testing phase
#
#         Returns:
#         --------
#         features : numpy.ndarray
#             Extracted features for classification
#         y_data : numpy.ndarray
#             Class labels
#         """
#         n_samples = len(X_data)
#         n_classes = len(np.unique(y_data))
#         print(f"Total samples: {n_samples}, Total classes: {n_classes}")
#
#         # Apply wavelet decomposition to obtain cD2 matrices for each sample
#         cd2_matrices = []
#         start_time = time.time()
#         for i in range(n_samples):
#             cd2_matrices.append(self.apply_wavelet_decomposition(X_data[i]))
#         print(f"Wavelet decomposition completed in {time.time() - start_time:.2f}s")
#
#         # Apply OVR-CSP for spatial filtering (training phase)
#         if is_training:
#             print("Applying OVR-CSP spatial filtering...")
#             start_time = time.time()
#             spatial_filters = self.apply_ovr_csp(cd2_matrices, y_data)
#             print(f"OVR-CSP spatial filtering completed in {time.time() - start_time:.2f}s")
#
#         # Apply spatial filtering and extract features with normalized variance
#         features = []
#         start_time = time.time()
#         for i, cd2_matrix in enumerate(cd2_matrices):
#             if i % 10 == 0:
#                 elapsed = time.time() - start_time
#                 # Uncomment for progress logging if needed
#                 # print(f"Feature extraction: {i}/{n_samples} samples processed (Elapsed: {elapsed:.2f}s)")
#
#             sample_features = []
#             for class_idx in range(n_classes):
#                 if class_idx in spatial_filters:
#                     # Apply spatial filter: filtered = P * cd2_matrix
#                     filtered = np.dot(spatial_filters[class_idx], cd2_matrix)
#
#                     # Compute variance for each row of filtered signals
#                     var_vector = np.var(filtered, axis=1)
#                     # Normalize the variance: divide each variance by the sum of variances
#                     norm_var = var_vector / np.sum(var_vector)
#                     # Log transform the normalized variance
#                     log_var = np.log(norm_var)
#                     sample_features.extend(log_var)
#             features.append(sample_features)
#         print(f"Feature extraction completed in {time.time() - start_time:.2f}s")
#
#         return np.array(features), y_data
#
#     def train(self):
#         """Train the complete classification pipeline"""
#         print("Loading data...")
#         X_all, y_all = self.load_data()
#
#         print(f"Data loaded: {len(X_all)} samples with {len(np.unique(y_all))} classes")
#
#         # Split into training and testing sets
#         X_train, X_test, y_train, y_test = train_test_split(
#             X_all, y_all, test_size=self.test_size, random_state=self.random_state, stratify=y_all
#         )
#
#         print("Extracting features for training data...")
#         X_train_features, y_train = self.extract_features(X_train, y_train, is_training=True)
#
#         print("Training SVM classifier...")
#         self.clf = SVC(kernel='linear', C=1.0, random_state=self.random_state)
#         self.clf.fit(X_train_features, y_train)
#
#         print("Extracting features for testing data...")
#         X_test_features, y_test = self.extract_features(X_test, y_test, self.spatial_filter, is_training=False)
#
#         print("Evaluating classifier...")
#         y_pred = self.clf.predict(X_test_features)
#
#         accuracy = accuracy_score(y_test, y_pred)
#         conf_matrix = confusion_matrix(y_test, y_pred)
#         class_report = classification_report(y_test, y_pred)
#
#         print(f"Accuracy: {accuracy:.4f}")
#         print("Confusion Matrix:")
#         print(conf_matrix)
#         print("Classification Report:")
#         print(class_report)
#
#         # Plot confusion matrix
#         plt.figure(figsize=(10, 8))
#         sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
#                    xticklabels=self.classes[:len(np.unique(y_all))],
#                    yticklabels=self.classes[:len(np.unique(y_all))])
#         plt.xlabel('Predicted')
#         plt.ylabel('True')
#         plt.title('Confusion Matrix')
#         plt.tight_layout()
#         plt.savefig('confusion_matrix.png')
#         plt.show()
#
#         return accuracy, conf_matrix, class_report
#
#     def predict(self, X_new):
#         """
#         Predict class labels for new data
#
#         Parameters:
#         -----------
#         X_new : numpy.ndarray
#             New EEG data with shape (n_samples, n_channels, n_times)
#
#         Returns:
#         --------
#         y_pred : numpy.ndarray
#             Predicted class labels
#         """
#         if self.clf is None:
#             raise ValueError("Classifier not trained. Call train() first.")
#
#         # Apply wavelet decomposition
#         wavelet_features = []
#         for i in range(len(X_new)):
#             wavelet_features.append(self.apply_wavelet_decomposition(X_new[i]))
#
#         # Compute cD2 matrices
#         cd2_matrices = self.compute_cd2_matrix(wavelet_features)
#
#         # Apply spatial filtering and extract features
#         features = []
#         n_classes = len(self.spatial_filter)
#
#         for cd2_matrix in cd2_matrices:
#             # Apply all class-specific spatial filters
#             class_features = []
#             for class_idx in range(n_classes):
#                 if class_idx in self.spatial_filter:
#                     # Apply spatial filter
#                     filtered = np.dot(self.spatial_filter[class_idx], cd2_matrix)
#
#                     # Extract features (variance of filtered signals)
#                     var = np.var(filtered, axis=1)
#                     log_var = np.log(var)
#                     class_features.extend(log_var)
#
#             features.append(class_features)
#
#         # Make predictions using trained classifier
#         y_pred = self.clf.predict(np.array(features))
#
#         return y_pred
#
# # Example usage
# if __name__ == "__main__":
#     data_directory = "processed_dataset"  # Path to your dataset
#
#     # Initialize classifier
#     eeg_classifier = OlfactoryEEGClassifier(data_directory)
#
#     # Train and evaluate
#     accuracy, conf_matrix, class_report = eeg_classifier.train()
#
#     print(f"Final accuracy: {accuracy:.4f}")
import os
import numpy as np
import pandas as pd
import pywt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from scipy.linalg import eig
import matplotlib.pyplot as plt
import seaborn as sns
import time
import mne  # EEGLAB 파일 읽기에 사용


class OlfactoryEEGClassifier:
    def __init__(self, data_dir, wavelet='db4', wavelet_level=5, test_size=0.3, random_state=42):
        """
        Initialize the EEG classifier for olfactory signals

        Parameters:
        -----------
        data_dir : str
            Directory containing the preprocessed EEG data (.set/.fdt files)
        wavelet : str
            Wavelet family to use for decomposition
        wavelet_level : int
            Level of wavelet decomposition
        test_size : float
            Proportion of the dataset to include in the test split
        random_state : int
            Random seed for reproducibility
        """
        self.data_dir = data_dir
        self.wavelet = wavelet
        self.wavelet_level = wavelet_level
        self.test_size = test_size
        self.random_state = random_state
        # 13 classes (A to M)
        self.classes = [chr(i) for i in range(ord('A'), ord('N'))]
        self.clf = None
        self.spatial_filter = None

    def load_data(self):
        """Load all .set files from the preprocessed directory.
           Also, extract subject information from file names.
           Returns:
             X_all: numpy array of EEG data (n_samples, n_channels, n_times)
             y_all: numpy array of numeric class labels
             subjects: list of subject labels corresponding to each sample
        """
        X_all = []
        y_all = []
        subjects = []

        print(f"Data directory: {self.data_dir}")
        all_files = [f for f in os.listdir(self.data_dir) if f.endswith('.set')]
        print(f"Found {len(all_files)} .set files.")

        for file in all_files:
            file_path = os.path.join(self.data_dir, file)
            # 파일명 예: "Sub. 1_A_001.set" → subject 정보는 첫 번째 토큰, 클래스는 두 번째 토큰
            parts = file.split('_')
            if len(parts) < 3:
                print(f"Unexpected filename format, skipping: {file}")
                continue
            subject_label = parts[0].strip()  # 예: "Sub. 1" 또는 "Sub.1"
            class_label = parts[1].strip()  # 예: "A", "B", ..., "M"
            try:
                # EEGLAB 파일 읽기 (내부적으로 .fdt 파일을 참조)
                raw = mne.io.read_raw_eeglab(file_path, preload=True, verbose=False)
                eeg_data = raw.get_data()  # shape: (n_channels, n_times)
                if eeg_data.size == 0:
                    print(f"Empty file: {file_path}")
                else:
                    X_all.append(eeg_data)
                    y_all.append(ord(class_label.upper()) - ord('A'))
                    subjects.append(subject_label)
                    print(f"Loaded {file_path}, shape: {eeg_data.shape}")
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

        print(f"Total samples loaded: {len(X_all)}")
        return np.array(X_all), np.array(y_all), subjects

    def apply_wavelet_decomposition(self, eeg_data):
        """
        Apply wavelet decomposition using DWT.

        Parameters:
        -----------
        eeg_data : numpy.ndarray
            EEG data with shape (channels, time_points)

        Returns:
        --------
        wavelet_features : numpy.ndarray
            For each channel, a selected set of detail coefficients (here, using index self.wavelet_level-1)
        """
        wavelet_features = []

        for channel in range(eeg_data.shape[0]):
            coeffs = pywt.wavedec(eeg_data[channel, :], self.wavelet, level=self.wavelet_level)
            cd_level = self.wavelet_level - 1
            if cd_level < len(coeffs):
                cd_coeff = coeffs[cd_level]
            else:
                cd_coeff = coeffs[-1]
            wavelet_features.append(cd_coeff)

        return np.array(wavelet_features)

    def compute_cd2_matrix(self, wavelet_features_all):
        """
        Compute the cD2 Wavelet Coefficients Matrix.

        Parameters:
        -----------
        wavelet_features_all : list or np.ndarray
            List of wavelet features for all samples

        Returns:
        --------
        cd2_matrices : list
            List of cD2 matrices for all samples
        """
        cd2_matrices = []
        for wavelet_features in wavelet_features_all:
            # wavelet_features: shape (n_channels, feature_length)
            cd2_length = len(wavelet_features[0]) // (2 ** (self.wavelet_level - 2))
            cd2_start = len(wavelet_features[0]) - cd2_length
            cd2_end = cd2_start + cd2_length
            cd2_matrix = np.zeros((wavelet_features.shape[0], cd2_length))
            for i in range(wavelet_features.shape[0]):
                cd2_matrix[i, :] = wavelet_features[i][cd2_start:cd2_end]
            cd2_matrices.append(cd2_matrix)
        return cd2_matrices

    def apply_ovr_csp(self, X_train, y_train, n_components=2):
        """
        Apply One-vs-Rest Common Spatial Pattern (OVR-CSP) with whitening.

        Parameters:
        -----------
        X_train : numpy.ndarray
            Training data with shape (n_samples, n_channels, n_times)
        y_train : numpy.ndarray
            Training labels
        n_components : int
            Number of components from each side

        Returns:
        --------
        spatial_filters : dict
            Dictionary with class indices as keys and spatial filter matrices as values
        """
        n_classes = len(np.unique(y_train))
        n_channels = X_train[0].shape[0]
        spatial_filters = {}

        for class_idx in range(n_classes):
            cov_target = np.zeros((n_channels, n_channels))
            cov_non_target = np.zeros((n_channels, n_channels))
            n_target = 0
            n_non_target = 0

            for i, X in enumerate(X_train):
                cov = np.cov(X)
                if y_train[i] == class_idx:
                    cov_target += cov
                    n_target += 1
                else:
                    cov_non_target += cov
                    n_non_target += 1

            if n_target > 0:
                cov_target /= n_target
            if n_non_target > 0:
                cov_non_target /= n_non_target

            C = cov_target + cov_non_target
            eigvals, eigvecs = np.linalg.eigh(C)
            idx = np.argsort(eigvals)[::-1]
            eigvals = eigvals[idx]
            eigvecs = eigvecs[:, idx]
            W = np.diag(1.0 / np.sqrt(eigvals)).dot(eigvecs.T)

            S1 = W.dot(cov_target).dot(W.T)
            eigvals_S1, eigvecs_S1 = np.linalg.eigh(S1)
            idx_S1 = np.argsort(eigvals_S1)[::-1]
            eigvals_S1 = eigvals_S1[idx_S1]
            eigvecs_S1 = eigvecs_S1[:, idx_S1]

            Q = eigvecs_S1.T.dot(W)
            spatial_filter = np.vstack((Q[:n_components, :], Q[-n_components:, :]))
            spatial_filters[class_idx] = spatial_filter

        self.spatial_filter = spatial_filters
        return spatial_filters

    def extract_features(self, X_data, y_data, spatial_filters=None, is_training=True):
        """
        Extract features using wavelet decomposition and spatial filtering.
        For each sample, perform DWT to obtain coefficients, then compute a cD2 matrix,
        apply OVR-CSP (if training) and extract normalized log-variance as features.

        Parameters:
        -----------
        X_data : numpy.ndarray
            EEG data with shape (n_samples, n_channels, n_times)
        y_data : numpy.ndarray
            Class labels
        spatial_filters : dict, optional
            Spatial filters to use (for test phase)
        is_training : bool
            Flag for training or testing

        Returns:
        --------
        features : numpy.ndarray
            Feature matrix for classification
        y_data : numpy.ndarray
            Class labels
        """
        n_samples = len(X_data)
        n_classes = len(np.unique(y_data))
        print(f"Total samples: {n_samples}, Total classes: {n_classes}")

        wavelet_features_all = []
        start_time = time.time()
        for i in range(n_samples):
            wf = self.apply_wavelet_decomposition(X_data[i])
            wavelet_features_all.append(wf)
        print(f"Wavelet decomposition completed in {time.time() - start_time:.2f}s")

        cd2_matrices = self.compute_cd2_matrix(np.array(wavelet_features_all))

        if is_training:
            print("Applying OVR-CSP spatial filtering...")
            start_time = time.time()
            spatial_filters = self.apply_ovr_csp(cd2_matrices, y_data)
            print(f"OVR-CSP completed in {time.time() - start_time:.2f}s")

        features = []
        start_time = time.time()
        for cd2_matrix in cd2_matrices:
            sample_features = []
            for class_idx in range(n_classes):
                if class_idx in spatial_filters:
                    filtered = np.dot(spatial_filters[class_idx], cd2_matrix)
                    var_vector = np.var(filtered, axis=1)
                    norm_var = var_vector / np.sum(var_vector)
                    log_var = np.log(norm_var)
                    sample_features.extend(log_var)
            features.append(sample_features)
        print(f"Feature extraction completed in {time.time() - start_time:.2f}s")

        return np.array(features), y_data

    def train(self):
        """Train the complete pipeline: data loading, feature extraction, SVM training, and evaluation."""
        print("Loading data...")
        X_all, y_all, _ = self.load_data()
        print(f"Data loaded: {len(X_all)} samples with {len(np.unique(y_all))} classes")

        X_train, X_test, y_train, y_test = train_test_split(
            X_all, y_all, test_size=self.test_size, random_state=self.random_state, stratify=y_all
        )

        print("Extracting features for training data...")
        X_train_features, y_train = self.extract_features(X_train, y_train, is_training=True)

        print("Training SVM classifier...")
        self.clf = SVC(kernel='linear', C=1.0, random_state=self.random_state)
        self.clf.fit(X_train_features, y_train)

        print("Extracting features for testing data...")
        X_test_features, y_test = self.extract_features(X_test, y_test, self.spatial_filter, is_training=False)

        print("Evaluating classifier...")
        y_pred = self.clf.predict(X_test_features)
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred)

        print(f"Accuracy: {accuracy:.4f}")
        print("Confusion Matrix:")
        print(conf_matrix)
        print("Classification Report:")
        print(class_report)

        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.classes[:len(np.unique(y_all))],
                    yticklabels=self.classes[:len(np.unique(y_all))])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        plt.show()

        return accuracy, conf_matrix, class_report

    def predict(self, X_new):
        """
        Predict class labels for new EEG data.

        Parameters:
        -----------
        X_new : numpy.ndarray
            New EEG data with shape (n_samples, n_channels, n_times)

        Returns:
        --------
        y_pred : numpy.ndarray
            Predicted class labels
        """
        if self.clf is None:
            raise ValueError("Classifier not trained. Call train() first.")

        wavelet_features_all = []
        for i in range(len(X_new)):
            wf = self.apply_wavelet_decomposition(X_new[i])
            wavelet_features_all.append(wf)
        cd2_matrices = self.compute_cd2_matrix(np.array(wavelet_features_all))

        features = []
        n_classes = len(self.spatial_filter)
        for cd2_matrix in cd2_matrices:
            sample_features = []
            for class_idx in range(n_classes):
                if class_idx in self.spatial_filter:
                    filtered = np.dot(self.spatial_filter[class_idx], cd2_matrix)
                    var_vector = np.var(filtered, axis=1)
                    log_var = np.log(var_vector)
                    sample_features.extend(log_var)
            features.append(sample_features)
        y_pred = self.clf.predict(np.array(features))
        return y_pred


# Example usage when running this module directly.
if __name__ == "__main__":
    data_directory = "preprocessed_5th"  # Folder containing .set files
    eeg_classifier = OlfactoryEEGClassifier(data_directory)
    accuracy, conf_matrix, class_report = eeg_classifier.train()
    print(f"Final accuracy: {accuracy:.4f}")
