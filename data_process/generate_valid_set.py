import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis

#order = ["mean_XYZ", "sum_XYZ", 'min', 'max', "standard deviation_XYZ", "skewness_XYZ", "kurtosis_XYZ", "correlation_XY+XZ+YZ", "vector magnitude_mean, sum, min, max, sd, skewness, kurtosis"]



#Return a nparray with shape N*32; 31 attributes + 1 label
def abstract_attributes(filepath):
    original_data = pd.read_csv(filepath)
    original_data = original_data.to_numpy()

    accel_data = []
    for data in original_data[:,2]:
        str_list = data.split(' ')[-1]
        list = eval(str_list)
        accel_data.append(list)

    attributes_list = []
    for point in accel_data:
        data_point = np.array(point)
        mean_x = (data_point[:,0].mean())
        mean_y = (data_point[:,1].mean())
        mean_z = (data_point[:,2].mean())
        sum_x = (data_point[:,0].sum())
        sum_y = (data_point[:,1].sum())
        sum_z = (data_point[:,2].sum())    
        min_x = (data_point[:,0].min())    
        min_y = (data_point[:,1].min())  
        min_z = (data_point[:,2].min())
        max_x = (data_point[:,0].max())    
        max_y = (data_point[:,1].max())  
        max_z = (data_point[:,2].max())  
        sd_x = (np.std(data_point[:,0]))
        sd_y = (np.std(data_point[:,1]))
        sd_z = (np.std(data_point[:,1]))
        skew_x = (skew(data_point[:,0]))
        skew_y = (skew(data_point[:,1]))
        skew_z = (skew(data_point[:,2]))
        kurt_x = (kurtosis(data_point[:,0]))
        kurt_y = (kurtosis(data_point[:,1]))
        kurt_z = (kurtosis(data_point[:,2]))
        corre_xy = np.corrcoef(data_point[:,0], data_point[:,1])[0,1]
        corre_xz = np.corrcoef(data_point[:,0], data_point[:,2])[0,1]
        corre_yz = np.corrcoef(data_point[:,1], data_point[:,2])[0,1]
        vm_mean = np.linalg.norm([mean_x,mean_y,mean_z])
        vm_sum = np.linalg.norm([sum_x,sum_y,sum_z])
        vm_min = np.linalg.norm([min_x,min_y,min_z])
        vm_max = np.linalg.norm([max_x,max_y,max_z])
        vm_sd = np.linalg.norm([sd_x,sd_y,sd_z])
        vm_kurt = np.linalg.norm([kurt_x,kurt_y,kurt_z])
        vm_skew = np.linalg.norm([skew_x,skew_y,skew_z])

        attributes = [mean_x,mean_y,mean_z,sum_x,sum_y,sum_z,min_x,min_y,min_z,max_x,max_y,max_z,sd_x,sd_y,sd_z,skew_x,skew_y,skew_z,kurt_x,kurt_y,kurt_z,corre_xy,corre_xz,corre_yz,vm_mean,vm_sum,vm_min,vm_max,vm_sd,vm_kurt,vm_skew]
        attributes_list.append(attributes)

    attributes = np.array(attributes_list)
    label = np.array(original_data[:,1], dtype="str").reshape(-1,1)
    result = np.hstack((attributes_list, label), dtype="object") 
    return result

if __name__ == "__main__":
    result1 = abstract_attributes("./original_data/cat_label.csv")
    result2 = abstract_attributes("./original_data/6.4_label.csv")
    result = np.concatenate((result1, result2), axis=0)


    np.savetxt("./label_cluster/valid.csv", result, delimiter=',', fmt='%s')
    print("Process complete, save as valid_set.csv. One attributes list represent 165 XYZ accel datapoint.")
