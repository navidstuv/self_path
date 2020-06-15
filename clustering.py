from utils.utils import GMM, random_proj, stats,kmean, HDBscan

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


train_features = np.load('features_val_mag_feature_extractor.npy')
test_features = np.load('features_val_feature_extractor.npy')
test_label = np.load('true_val_mag_feature_extractor.npy')
print(np.amax(train_features))
print(np.amax(test_features))

###for random project
# feature_trunc_train = random_proj(train_features)
# feature_trunc_test = random_proj(test_features)

###Normalize
# scaler = StandardScaler()
# scaler.fit(train_features)
# train_features = scaler.transform(train_features)
# test_features = scaler.transform(test_features)

###PCA
pca = PCA(n_components=2)
pca.fit(train_features)
feature_trunc_train = pca.transform(train_features)
# feature_trunc_test = pca.transform(test_features)

plt.scatter(feature_trunc_train[:,0], feature_trunc_train[:,1])
plt.show()

pred = HDBscan(feature_trunc_train)
print(accuracy_score(test_label, pred))
# _ = stats(pred, test_label)