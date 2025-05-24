import matplotlib.pyplot as plt
import numpy as np

# 데이터 정의
thresholds = ['1', '0.99', '0.98', '0.95', '0.9', '0.85', '0.65', '임의']
x = np.arange(len(thresholds))

wavelet_times = [6.393, 3.623, 3.740, 2.431, 1.791, 1.589, 1.251, 0.673]
ovr_csp_times = [4.471, 3.177, 3.171, 2.776, 2.696, 4.602, 1.764, 1.386]
feature_times = [6.618, 3.576, 3.597, 2.600, 2.951, 5.059, 1.556, 1.058]
svm_accuracies = [97.62, 97.70, 97.74, 97.60, 96.44, 95.42, 89.59, 74.81]
#svm_times = [1.312, 0.888, 1.097, 0.780, 0.833, 0.759, 0.839, 0.984]
knn_accuracies = [96.60, 96.30, 96.76, 96.72, 96.00, 95.66, 95.02, 92.03]
#knn_times = [0.002, 0.003, 0.003, 0.002, 0.002, 0.0015, 0.001, 0.001]

# 1. Wavelet Decomposition Time
plt.figure(figsize=(8, 5))
plt.plot(x, wavelet_times, marker='o')
plt.title("Wavelet Decomposition Time")
plt.ylabel("Time (s)")
plt.xticks(x, thresholds)
plt.grid(False)
plt.show()

# 2. OVR-CSP Training Time
plt.figure(figsize=(8, 5))
plt.plot(x, ovr_csp_times, marker='o', color='orange')
plt.title("OVR-CSP Training Time")
plt.ylabel("Time (s)")
plt.xticks(x, thresholds)
plt.show()

# 3. Feature Extraction from CSP Time
plt.figure(figsize=(8, 5))
plt.plot(x, feature_times, marker='o', color='green')
plt.title("Feature Extraction from CSP Time")
plt.ylabel("Time (s)")
plt.xticks(x, thresholds)
plt.show()

# 4. SVM Accuracy & Time
plt.figure(figsize=(8, 5))
plt.bar(x, svm_accuracies, label="SVM Accuracy (%)")
#plt.plot(x, svm_times, color='red', marker='o', label="SVM Time (s)")
plt.title("SVM Accuracy")
plt.ylabel("Accuracy (%)")
plt.xticks(x, thresholds)
plt.legend()
plt.show()

# 5. KNN Accuracy & Time
plt.figure(figsize=(8, 5))
plt.bar(x, knn_accuracies, color='lightgreen', label="KNN Accuracy (%)")
#plt.plot(x, knn_times, color='red', marker='o', label="KNN Time (s)")
plt.title("KNN Accuracy")
plt.ylabel("Accuracy (%)")
plt.xticks(x, thresholds)
plt.legend()
#plt.grid(True)
plt.show()
