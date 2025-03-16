import numpy as np
import matplotlib.pyplot as plt
from mne import create_info, EpochsArray
from mne.time_frequency import tfr_multitaper
from mne.stats.cluster_level import permutation_cluster_test
import scipy.stats as stats
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import precision_score, recall_score, f1_score
from scipy.stats import ttest_ind

# Simulated EEG data for demonstration purposes
n_channels = 64
n_samples = 1000
n_epochs = 50



sfreq = 500  # Sampling frequency
t = np.arange(n_samples) / sfreq
# Simulate EEG data with oscillatory components and noise
theta = 0.5 * np.sin(2 * np.pi * 5 * t)  # 5 Hz theta
alpha = 0.3 * np.sin(2 * np.pi * 10 * t)  # 10 Hz alpha
noise = 0.2 * np.random.randn(n_epochs, n_channels, n_samples)
data = noise + theta + alpha

# Create MNE Epochs object
info = create_info(ch_names=[f'EEG{i}' for i in range(n_channels)], sfreq=sfreq, ch_types='eeg')
epochs = EpochsArray(data, info)

# Compute time-frequency representation (TFR)
freqs = np.arange(4, 40, 1)  # Frequency range
tfr = tfr_multitaper(epochs, freqs=freqs, n_cycles=freqs / 2., time_bandwidth=2.0, return_itc=False)

# Plot ERDS map for a single channel
channel_idx = 0
tfr_data = tfr.data[channel_idx]
plt.imshow(tfr_data, aspect='auto', origin='lower',
           extent=[epochs.times[0], epochs.times[-1], freqs[0], freqs[-1]],
           cmap='RdBu_r')
plt.colorbar(label='Power Change (%)')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.title(f'Time-Frequency ERDS Map for Channel {channel_idx}')
plt.show()









features = []
for epoch in data:
    for channel in epoch:
        band_power_theta = np.mean(channel[4:8])  # Theta band power
        band_power_alpha = np.mean(channel[8:12])  # Alpha band power
        alpha_to_theta_ratio = band_power_alpha / band_power_theta
        theta_to_alpha_ratio = band_power_theta / band_power_alpha
        mean_val = np.mean(channel)
        variance_val = np.var(channel)
        skewness_val = stats.skew(channel)
        kurtosis_val = stats.kurtosis(channel)
        entropy_val = stats.entropy(np.abs(channel))
        features.append([band_power_theta, band_power_alpha, alpha_to_theta_ratio,
                         theta_to_alpha_ratio, mean_val, variance_val,
                         skewness_val, kurtosis_val, entropy_val])

# Convert features to numpy array for further processing


```
# Start of Selection
features = np.array(features)
print("Feature shape:", repr(features.shape))  # Verify dimensions (n_epochs * n_channels x feature_count)

# Simulated labels for binary classification (e.g., workload levels)
labels = np.random.randint(0, 2, size=(features.shape[0]))

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Select top k features using ANOVA F-test
selector = SelectKBest(score_func=f_classif, k=5)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

selected_features_indices = selector.get_support(indices=True)
print("Selected feature indices:", selected_features_indices)
```

---

### **5. Machine Learning Classification**
```python
# Train a Random Forest classifier on selected features
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train_selected, y_train)

# Evaluate classifier performance on test set using accuracy metric
accuracy = clf.score(X_test_selected, y_test)
print("Classifier accuracy:", accuracy)

# Perform k-fold cross-validation to assess model stability and generalization performance
cv_scores = cross_val_score(clf, X_train_selected, y_train, cv=5)
print("Cross-validation accuracy:", cv_scores.mean())

# Predict on test set and compute precision/recall/F1-score metrics
y_pred = clf.predict(X_test_selected)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print("Precision: {}, Recall: {}, F1-score: {}".format(precision, recall, f1))

### **6. Statistical Analysis**
condition_1_features = features[labels == 0]
condition_2_features = features[labels == 1]
t_statistic, p_value = ttest_ind(condition_1_features[:, selected_features_indices],
                                 condition_2_features[:, selected_features_indices], axis=0)

print("T-statistics:", t_statistic)
print("P-values:", p_value)
# Implement cluster-based permutation test
from mne.stats import permutation_cluster_test

# Reshape data for cluster-based test (assuming we have multiple channels/features)
condition_1_data = condition_1_features[:, selected_features_indices].T  # Transpose for channels x times format
condition_2_data = condition_2_features[:, selected_features_indices].T

# Run permutation cluster test
f_obs, clusters, cluster_pv, h0 = permutation_cluster_test(
    [condition_1_data, condition_2_data],
    n_permutations=1000,
    threshold=2.0,  # Threshold for cluster formation
    tail=0,  # Two-tailed test
    n_jobs=1,
    verbose=True
)

print("Significant clusters (p < 0.05):", [i for i, p in enumerate(cluster_pv) if p < 0.05])
print("Cluster p-values:", cluster_pv)
```

