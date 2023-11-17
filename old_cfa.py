import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors

class CFA:
    def __init__(self, fd, tol):
        """
        Initialize the CFA algorithm with feature differences (fd) and tolerance (tol) thresholds.

        :param fd: Maximum number of features that can differ to consider data points as paired.
        :param tol: Specified tolerance percentage for feature values.
        """
        self.fd = fd
        self.tol = tol

    def calculate_tolerance(self, X):
        """
        Calculate the tolerance for each feature.

        :param X: Feature dataset.
        :return: Tolerance values for each feature.
        """
        std_dev = np.std(X, axis=0)
        return self.tol * std_dev

    def separate_classes(self, X, y):
        """
        Separate the dataset into majority and minority classes.

        :param X: Feature dataset.
        :param y: Class labels.
        :return: Majority and minority class data.
        """
       
        unique_classes, counts = np.unique(y, return_counts=True)
        majority_class_index = np.argmax(counts)
        majority_class = unique_classes[majority_class_index]
        majority_data = X[y == majority_class]
        minority_data = X[y != majority_class]
        
        return majority_data, minority_data

    def compute_cf_set(self, X, neighbors_model):
        """
        Compute the CF-Set for each instance in X.

        :param X: Dataset to compute CF-Set.
        :param neighbors_model: Nearest neighbors model for finding nearest neighbor.
        :return: Array of indices of nearest neighbors.
        """
        distances, indices = neighbors_model.kneighbors(X)
        
        return indices[:, 0]

    def check_feature_differences(self, x, y, tolerance):
        """
        Check the number of feature differences between two instances within the tolerance.

        :param x: First instance.
        :param y: Second instance.
        :param tolerance: Tolerance values for each feature.
        :return: Number of features that differ.
        """
        return np.sum(np.abs(x - y) > tolerance)

    def generate_synthetic_instance(self, x, xp, p):
        """
        Generate a synthetic counterfactual instance.

        :param x: Native instance.
        :param p: Counterfactual instance from CF-Set.
        :return: Synthetic instance.
        """
        return x +(xp - p)

    def run_cfa(self, X, y):
        """
        Run the CFA algorithm on a given dataset.

        :param X: Feature dataset.
        :param y: Class labels.
        :return: Set of synthetic counterfactual instances.
        """
        tolerance = self.calculate_tolerance(X)
        majority_data, minority_data = self.separate_classes(X, y)

        nn_model = NearestNeighbors(n_neighbors=1).fit(majority_data)
        cf_set_indices = self.compute_cf_set(minority_data, nn_model)
        
        # Pair Majo - Mino
        paired_instances_indices = []
        paired_instances = []
        
        for i, idx in enumerate(cf_set_indices):
            #print(majority_data[idx],'is located at idx', np.where((X == majority_data[idx]).all(axis=1))[0][0],'in X')
            #print(np.where(X == majority_data[idx])[0][0], np.where(X == majority_data[i])[0][0])
            paired_instances_indices.append((np.where((X == majority_data[idx]).all(axis=1))[0][0], np.where((X == minority_data[i]).all(axis=1))[0][0]))
            paired_instances.append((majority_data[idx], minority_data[i])) # (majority_data instance, minority_data instance)
            
        print('X',X)
        print('pi',paired_instances_indices)
        
       
        # Pair Non paired Majo - Paired Majo
        npmaj_pmaj = []
        npmaj_pmaj_indices = []
        
        #print('paired_instances from majo', np.array(paired_instances)[:,0])
        #print('majority_data', majority_data)
        for i,np_maj in enumerate(majority_data):
            #print('np_maj: ', np_maj, 'paired_instances: ', np.array(paired_instances)[:,0])
            
            
           
            if not np.any(np.all(np.array(paired_instances)[:,0] == np.array(np_maj), axis=1)):
                # print('np_maj', np_maj, 'not in paired_instances', np.array(paired_instances)[:,0])
                
                # Check the data in majority data that is paired the closest to np maj
                nn_model = NearestNeighbors(n_neighbors=1).fit(np.array(paired_instances)[:,0])
                _, indices = nn_model.kneighbors(np.array([np_maj]))
                # print('indices: ', indices)
                # print(i,majority_data[i])
                print(np.where((X==np.array(majority_data)[i]).all(axis=1))[0][0])
                npmaj_pmaj_indices.append((np.where((X==np.array(majority_data)[i]).all(axis=1))[0][0], np.where((X==np.array(paired_instances)[indices[0][0]][0]).all(axis=1))[0][0]))
                npmaj_pmaj.append((majority_data[i], paired_instances[indices[0][0]][0]))
                
        print('npmaj_pmaj_indices: ', npmaj_pmaj_indices)
        print('npmaj_pmaj: ', npmaj_pmaj)
                
                
        
                
        
        # Transfer Non paire Majo - Paired Majo - Mino   
        synthetic_instances = []  
        for id, p in zip(npmaj_pmaj_indices, npmaj_pmaj):
            print("id: ", id)
            print("p: ", p)
            
            
            print('pi',paired_instances)
            print('pi',np.array(paired_instances)[:,0])
            print('p1',p[1])
            print(np.where(np.all(np.array([elem[0] for elem in paired_instances]) == p[1], axis=1))[0])
            
            
            
        
            if self.check_feature_differences(p[1],np.array(paired_instances)[np.where((np.array(paired_instances)[:,0] == p[1]).all(axis=1))[0]][0][1], tolerance) <= self.fd:
                
                print('PHASE 2----------------------')
                print(p[1], np.array(paired_instances)[np.where((np.array(paired_instances)[:,0] == p[1]).all(axis=1))[0]][0][1])
                synthetic_instances.append(self.generate_synthetic_instance(p[0], np.array(paired_instances)[np.where((np.array(paired_instances)[:,0] == p[1]).all(axis=1))[0]][0][1], p[1]))
             
        return np.array(synthetic_instances)

