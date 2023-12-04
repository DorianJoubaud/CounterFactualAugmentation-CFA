import numpy as np
from sklearn.neighbors import NearestNeighbors
import pandas as pd

class CFA:
    def __init__(self, fd, tol):
        """
        Initialize the Counterfactual Analysis (CFA) algorithm.
        
        :param fd: Feature dimension of the dataset.
        :param tol: Tolerance factor for standard deviation in feature values.
        """
        self.fd = fd
        self.tol = tol
        self.mino_label = None
        

    def calculate_tolerance(self, X):
        """
        Calculate the tolerance level for each feature based on standard deviation.
        
        :param X: Dataset features.
        :return: Array of tolerance levels for each feature.
        """
        
        return self.tol * np.std(X, axis=0) + np.mean(X, axis=0)

    def separate_classes(self, X, y):
        """
        Separate the dataset into majority and minority classes.

        :param X: Feature dataset.
        :param y: Class labels.
        :return: Majority and minority class data.
        """
       
        unique_classes, counts = np.unique(y, return_counts=True)
        self.mino_label = unique_classes[np.argmin(counts)]
        majority_class_index = np.argmax(counts)
        majority_class = unique_classes[majority_class_index]
        majority_data = X[y == majority_class]
        minority_data = X[y != majority_class]
        
        return majority_data, minority_data

    def compute_cf_set(self, X, neighbors_model):
        """
        Compute the Counterfactual Set (CF-Set) for each instance in the minority class.
        
        :param X: Minority class instances.
        :param neighbors_model: Nearest neighbors model trained on majority class.
        :return: Indices of nearest neighbors in the majority class.
        """
        return neighbors_model.kneighbors(X, return_distance=False)[:, 0]
    
   
    
    def check_id_fd_accepted(self, instance, counterfactual, fd):
        """
        Check the number of feature that are considered as the same between an instance and its counterfactual. 
        Two features are considered the same if they are within the tolerance level. Otherwise, they are considered not the same.
        To be  accepted, the number of features considered the same must be superior or equal to the feature diff (fd)
        
        :param instance: The instance.
        :param counterfactual: The counterfactual instance.
        :param fd: The feature diff
        :return: True if the feature dimensions are the same, False otherwise.
        """
       
       
        return np.sum(np.abs(instance - counterfactual) <= self.tol) >= fd
      

    def generate_synthetic_instance(self, non_paired_majority, counterfactual, closest_paired_majority):
        """
        Generate a synthetic instance by adjusting a non-paired majority instance towards the minority class.
        
        :param non_paired_majority: A majority instance not paired with any minority instance.
        :param counterfactual: A minority instance serving as a counterfactual example.
        :param closest_paired_majority: The closest majority instance that is paired with a minority instance.
        :return: A synthetic instance.
        """
        return non_paired_majority + (counterfactual - closest_paired_majority)

    def find_nearest_paired(self, instance, paired_instances):
        """
        Find the nearest paired instance for a given non-paired instance.
        
        :param instance: The non-paired instance.
        :param paired_instances: List of paired majority instances.
        :return: The nearest paired majority instance.
        """
        nn_model = NearestNeighbors(n_neighbors=1).fit(paired_instances)
        _, idx = nn_model.kneighbors([instance])
        return paired_instances[idx[0][0]]

    def run_cfa(self, X, y, get_synt_labels=False):
        # Calculate the tolerance levels for feature adjustment
        tolerance = self.calculate_tolerance(X)

        # Separating the dataset into majority and minority classes
        majority_data, minority_data = self.separate_classes(X, y)

        # Track paired and used majority instances
        paired_majority_indices = list()
        closest_majority_instance = list()
        paired_minority_indices = list()
        used_majority_indices = list()

        synthetic_instances = []
        for i, minority_instance in enumerate(minority_data):
            # Train nearest neighbors model for the minority instance
            nn_model = NearestNeighbors(n_neighbors=len(majority_data)).fit(majority_data)
            distances, indices = nn_model.kneighbors([minority_instance])
            

            for idx in indices[0]:
                if idx not in paired_majority_indices:
                    # Found an unpaired closest majority instance
                    closest_majority_instance.append(majority_data[idx])
                    paired_majority_indices.append(idx)
                    paired_minority_indices.append(i)
                    break
            
            
            
            
       

            # Generate and store the synthetic instance for each non-paired majority instance
        nn_model = NearestNeighbors(n_neighbors=1).fit(closest_majority_instance)
        for np_maj_idx, np_maj_instance in enumerate(majority_data):
            if np_maj_idx not in paired_majority_indices and np_maj_idx not in used_majority_indices:
                    
                    
                    distances, indices = nn_model.kneighbors([np_maj_instance])
                    
                    
                    majority_instace_closest_to_npaired= majority_data[np.where((majority_data == closest_majority_instance[indices[0][0]]).all(axis=1))[0][0]]
                    
                    
                    minority_data_closest_to_majority = minority_data[paired_minority_indices[np.where((closest_majority_instance == majority_instace_closest_to_npaired).all(axis=1))[0][0]]]
                   
                    # check if the number of feature that are considered as the same between an instance and its counterfactual is superior or equal to the feature diff (fd)
                    
                    if self.check_id_fd_accepted(minority_data_closest_to_majority, majority_instace_closest_to_npaired, self.fd):
                       
                        
                        synthetic_instance = self.generate_synthetic_instance(np_maj_instance, minority_data_closest_to_majority, majority_instace_closest_to_npaired)
                        synthetic_instances.append(synthetic_instance)
                        used_majority_indices.append(np_maj_idx)
         
         
        # if no synthetic instance is generated, return the original dataset
           
                        
        if get_synt_labels:
            if len(synthetic_instances) == 0:
                print('No synthetic instance generated')
                print('Please try to increase the tolerance level or decrease the feature diff')
                return X, y, np.zeros(X.shape[0])
            
            df = pd.DataFrame(X)
            df['Class'] = y
            df_cfa = pd.DataFrame(synthetic_instances)
            df_cfa['Class'] = self.mino_label
            df_cfa['Synthetic'] = 1
            df_cfa = pd.concat([df, df_cfa], ignore_index=True)
            df_cfa['Synthetic'] = df_cfa['Synthetic'].fillna(0)
            

            return np.array(df_cfa.drop(columns=['Class']).drop(columns = ['Synthetic'])), np.array(df_cfa['Class']), np.array(df_cfa['Synthetic'])

        else:
            if len(synthetic_instances) == 0:
                print('No synthetic instance generated')
                print('Please try to increase the tolerance level or decrease the feature diff')
                
                return X, y
            
            
            return np.concatenate((X, synthetic_instances)), np.concatenate((y, np.ones(np.array(synthetic_instances).shape[0])))
            