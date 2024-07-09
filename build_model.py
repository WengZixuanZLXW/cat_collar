from sklearn.cluster import MeanShift, estimate_bandwidth
import numpy as np
from joblib import dump, load
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

attributes = np.genfromtxt("./attributes_data.csv", delimiter=',')

bandwidth = estimate_bandwidth(attributes, quantile = 0.2)

ms = MeanShift(max_iter=1000, bandwidth=bandwidth)
ms.fit(attributes)

dump(ms, "./model_plot/meanshift.model")
np.savetxt("./model_plot/centers.csv", ms.cluster_centers_, delimiter=",")

print("After {} iteration, the model has {} cluster centers.".format(ms.n_iter_, ms.cluster_centers_.size/31))

# # 使用 t-SNE 进行降维
# tsne = TSNE(n_components=2, random_state=42)
# X_embedded = tsne.fit_transform(attributes)

# # 绘制聚类结果的散点图
# plt.figure(figsize=(8, 6))
# plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=ms.labels_, cmap='viridis', s=50)
# plt.title('t-SNE Visualization of Mean Shift Clustering')
# plt.xlabel('t-SNE Feature 1')
# plt.ylabel('t-SNE Feature 2')
# plt.colorbar()
# plt.show()

# print("plot finished")
