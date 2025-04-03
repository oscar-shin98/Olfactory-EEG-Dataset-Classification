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
class OlfactoryEEGClassifier:
    def __init__(self, data_dir, wavelet='db4', wavelet_level=5, test_size=0.3, random_state=42):
        """
        Initialize the EEG classifier for olfactory signals
        
        Parameters:
        -----------
        data_dir : str
            Directory containing the preprocessed EEG data
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
        self.subjects = [f"Sub. {i}" for i in range(1, 12)]
        self.classes = [chr(i) for i in range(ord('A'), ord('N'))]  # A through M
        self.clf = None
        self.spatial_filter = None

    def load_data(self):
        """Load all data from directory structure"""
        X_all = []
        y_all = []

        print(f"Data directory: {self.data_dir}")  # 데이터 폴더 경로 확인

        for subject in self.subjects:
            print(f"Processing subject: {subject}")  # 어떤 subject를 읽는지 확인
            for class_label in self.classes:
                class_dir = os.path.join(self.data_dir, subject, class_label)
                if not os.path.exists(class_dir):
                    print(f"Directory not found: {class_dir}")  # 폴더가 존재하지 않음
                    continue

                csv_files = [f for f in os.listdir(class_dir) if f.endswith('.csv')]
                print(f"Found {len(csv_files)} files in {class_dir}")  # 몇 개의 파일을 찾았는지 확인

                for file in csv_files:
                    file_path = os.path.join(class_dir, file)
                    try:
                        eeg_data = pd.read_csv(file_path, header=None).values.T
                        print(f"File: {file_path}, Shape: {eeg_data.shape}")  # 각 파일의 데이터 형태 확인
                        if eeg_data.shape[0] == 0 or eeg_data.shape[1] == 0:
                            print(f"Empty file: {file_path}")  # 빈 파일이면 출력
                        else:
                            X_all.append(eeg_data)
                            y_all.append(ord(class_label) - ord('A'))  # Convert letter to numeric label
                    except Exception as e:
                        print(f"Error reading {file_path}: {e}")  # 파일 읽기 오류 출력

        print(f"Total samples loaded: {len(X_all)}")
        return np.array(X_all), np.array(y_all)

    def apply_wavelet_decomposition(self, eeg_data):
        """
        Apply 5-level wavelet decomposition using DWT with db4
        
        Parameters:
        -----------
        eeg_data : numpy.ndarray
            EEG data with shape (channels, time_points)
            
        Returns:
        --------
        wavelet_features : numpy.ndarray
            Wavelet coefficients
        """
        wavelet_features = []
        
        for channel in range(eeg_data.shape[0]):
            # Apply wavelet decomposition
            coeffs = pywt.wavedec(eeg_data[channel, :], self.wavelet, level=self.wavelet_level)
            
            # Extract approximate and detailed coefficients
            feature_vector = np.concatenate([coef for coef in coeffs])
            wavelet_features.append(feature_vector)
            
        return np.array(wavelet_features)
    
    def compute_cd2_matrix(self, wavelet_features_all):
        """
        Compute the cD2 Wavelet Coefficients Matrix
        
        Parameters:
        -----------
        wavelet_features_all : list
            List of wavelet features for all samples
            
        Returns:
        --------
        cd2_matrices : list
            List of cD2 matrices for all samples
        """
        cd2_matrices = []
        
        for wavelet_features in wavelet_features_all:
            # Extract the cD2 coefficients (assuming they're at a specific position in the wavelet decomposition)
            # Note: This is a simplified implementation - you may need to adjust based on the actual paper details
            cd2_length = len(wavelet_features[0]) // (2**(self.wavelet_level - 2))
            cd2_start = len(wavelet_features[0]) - cd2_length
            cd2_end = cd2_start + cd2_length
            
            cd2_matrix = np.zeros((wavelet_features.shape[0], cd2_length))
            for i in range(wavelet_features.shape[0]):
                cd2_matrix[i, :] = wavelet_features[i, cd2_start:cd2_end]
                
            cd2_matrices.append(cd2_matrix)
            
        return cd2_matrices
    
    def apply_ovr_csp(self, X_train, y_train, n_components=2):
        """
        Apply One-vs-Rest Common Spatial Pattern for spatial filtering
        
        Parameters:
        -----------
        X_train : numpy.ndarray
            Training data with shape (n_samples, n_channels, n_times)
        y_train : numpy.ndarray
            Training labels
        n_components : int
            Number of CSP components to select
            
        Returns:
        --------
        spatial_filters : dict
            Dictionary with class labels as keys and spatial filters as values
        """
        n_classes = len(np.unique(y_train))
        n_channels = X_train[0].shape[0]
        
        # Initialize spatial filters dictionary
        spatial_filters = {}
        
        # For each class, compute OVR-CSP
        for class_idx in range(n_classes):
            # Compute mean covariance matrices for target vs. non-target
            cov_target = np.zeros((n_channels, n_channels))
            cov_non_target = np.zeros((n_channels, n_channels))
            
            n_target = 0
            n_non_target = 0
            
            for i, X in enumerate(X_train):
                # Compute covariance matrix
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
            
            # Solve generalized eigenvalue problem for CSP
            evals, evecs = eig(cov_target, cov_target + cov_non_target)
            
            # Sort eigenvalues and eigenvectors
            idx = np.argsort(evals)[::-1]  # In descending order
            evecs = evecs[:, idx]
            
            # Select n_components eigenvectors from each end
            spatial_filter = np.vstack((evecs[:, :n_components].T, evecs[:, -n_components:].T))
            spatial_filters[class_idx] = spatial_filter
        
        self.spatial_filter = spatial_filters
        return spatial_filters
    
    def extract_features(self, X_data, y_data, spatial_filters=None, is_training=True):
        """
        Extract features using wavelet decomposition and spatial filtering
        
        Parameters:
        -----------
        X_data : numpy.ndarray
            EEG data with shape (n_samples, n_channels, n_times)
        y_data : numpy.ndarray
            Class labels
        spatial_filters : dict, optional
            Spatial filters to apply (used for test data)
        is_training : bool
            Whether this is training or testing phase
            
        Returns:
        --------
        features : numpy.ndarray
            Extracted features for classification
        y_data : numpy.ndarray
            Class labels
        """
        n_samples = len(X_data)
        n_classes = len(np.unique(y_data))
        print(f"Total samples: {n_samples}, Total classes: {n_classes}")
        # Apply wavelet decomposition
        wavelet_features = []
        start_time = time.time()
        for i in range(n_samples):
            if i % 10 == 0:
                elapsed = time.time() - start_time
                #print(f"Wavelet decomposition: {i}/{n_samples} samples processed (Elapsed: {elapsed:.2f}s)")
            wavelet_features.append(self.apply_wavelet_decomposition(X_data[i]))
        print(f"Wavelet decomposition completed in {time.time() - start_time:.2f}s")

        # Compute cD2 matrices
        print("Computing cD2 matrices...")
        start_time = time.time()
        cd2_matrices = self.compute_cd2_matrix(wavelet_features)
        print(f"cD2 matrices computation completed in {time.time() - start_time:.2f}s")

        # Apply OVR-CSP for spatial filtering
        if is_training:
            print("Applying OVR-CSP spatial filtering...")
            start_time = time.time()
            spatial_filters = self.apply_ovr_csp(cd2_matrices, y_data)
            print(f"OVR-CSP spatial filtering completed in {time.time() - start_time:.2f}s")

        # Apply spatial filtering and extract features
        features = []
        start_time = time.time()
        for i, cd2_matrix in enumerate(cd2_matrices):
            if i % 10 == 0:
                elapsed = time.time() - start_time
                #print(f"Feature extraction: {i}/{n_samples} samples processed (Elapsed: {elapsed:.2f}s)")

            # Apply all class-specific spatial filters
            class_features = []
            for class_idx in range(n_classes):
                if class_idx in spatial_filters:
                    # Apply spatial filter
                    filtered = np.dot(spatial_filters[class_idx], cd2_matrix)

                    # Extract features (e.g., variance of filtered signals)
                    var = np.var(filtered, axis=1)
                    log_var = np.log(var)
                    class_features.extend(log_var)

            features.append(class_features)
        print(f"Feature extraction completed in {time.time() - start_time:.2f}s")
        return np.array(features), y_data
    
    def train(self):
        """Train the complete classification pipeline"""
        print("Loading data...")
        X_all, y_all = self.load_data()
        
        print(f"Data loaded: {len(X_all)} samples with {len(np.unique(y_all))} classes")
        
        # Split into training and testing sets
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
        
        # Plot confusion matrix
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
        Predict class labels for new data
        
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
        
        # Apply wavelet decomposition
        wavelet_features = []
        for i in range(len(X_new)):
            wavelet_features.append(self.apply_wavelet_decomposition(X_new[i]))
        
        # Compute cD2 matrices
        cd2_matrices = self.compute_cd2_matrix(wavelet_features)
        
        # Apply spatial filtering and extract features
        features = []
        n_classes = len(self.spatial_filter)
        
        for cd2_matrix in cd2_matrices:
            # Apply all class-specific spatial filters
            class_features = []
            for class_idx in range(n_classes):
                if class_idx in self.spatial_filter:
                    # Apply spatial filter
                    filtered = np.dot(self.spatial_filter[class_idx], cd2_matrix)
                    
                    # Extract features (variance of filtered signals)
                    var = np.var(filtered, axis=1)
                    log_var = np.log(var)
                    class_features.extend(log_var)
            
            features.append(class_features)
        
        # Make predictions using trained classifier
        y_pred = self.clf.predict(np.array(features))
        
        return y_pred

# Example usage
if __name__ == "__main__":
    data_directory = "processed_dataset"  # Path to your dataset
    
    # Initialize classifier
    eeg_classifier = OlfactoryEEGClassifier(data_directory)
    
    # Train and evaluate
    accuracy, conf_matrix, class_report = eeg_classifier.train()
    
    print(f"Final accuracy: {accuracy:.4f}")
