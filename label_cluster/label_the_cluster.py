from sklearn.cluster import MeanShift
import numpy as np
from joblib import dump, load

#load model
ms:MeanShift = load("./meanshift.model")
#process valid data
valid_data = np.genfromtxt("./label_cluster/valid.csv", delimiter=',', dtype="object")
attributes = valid_data[:,:-1]
label = np.array(valid_data[:,-1], dtype = "str")
#get result and visualize
prediction = ms.predict(attributes)
compare = np.column_stack((prediction, label))
np.savetxt("./label_cluster/predict_label.csv", compare, fmt='%s')
print(compare)