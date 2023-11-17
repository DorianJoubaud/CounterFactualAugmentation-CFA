import numpy as np
from sklearn.neighbors import NearestNeighbors

class CFA:
    def __init__(self, fd, tol):
        """
        Initialize the Counterfactual Analysis (CFA) algorithm.
        
        :param fd: Feature dimension of the dataset.
        :param tol: Tolerance factor for standard deviation in feature values.
        """
        self.fd = fd
        self.tol = tol

    def calculate_tolerance(self, X):
        """
        Calculate the tolerance level for each feature based on standard deviation.
        
        :param X: Dataset features.
        :return: Array of tolerance levels for each feature.
        """
        return self.tol * np.std(X, axis=0)

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
        Compute the Counterfactual Set (CF-Set) for each instance in the minority class.
        
        :param X: Minority class instances.
        :param neighbors_model: Nearest neighbors model trained on majority class.
        :return: Indices of nearest neighbors in the majority class.
        """
        return neighbors_model.kneighbors(X, return_distance=False)[:, 0]

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

    def run_cfa(self, X, y):
        """
        Execute the CFA algorithm to generate synthetic instances.
        
        :param X: Feature matrix.
        :param y: Target labels.
        :return: List of synthetic instances.
        """
        # Calculate the tolerance levels for feature adjustment
        tolerance = self.calculate_tolerance(X)

        # Separating the dataset into majority and minority classes
        majority_data, minority_data = self.separate_classes(X, y)

        # Train a nearest neighbors model on the majority class
        nn_model = NearestNeighbors(n_neighbors=1).fit(majority_data)

        # Compute the CF-Set for minority class instances
        cf_set_indices = self.compute_cf_set(minority_data, nn_model)

        # Pair each minority instance with its nearest majority instance
        paired_instances = [(majority_data[idx], minority_data[i]) for i, idx in enumerate(cf_set_indices)]

        synthetic_instances = []
        for np_maj in majority_data:
            # Check if a majority instance is not already paired
            if not any(np.array_equal(np_maj, maj) for maj, _ in paired_instances):
                # Find the nearest paired majority instance
                closest_paired_majority = self.find_nearest_paired(np_maj, [maj for maj, _ in paired_instances])
                # Retrieve the paired minority instance
                _, paired_minority = paired_instances[np.where(np.array([maj for maj, _ in paired_instances]) == closest_paired_majority)[0][0]]
                # Generate and store the synthetic instance
                synthetic_instance = self.generate_synthetic_instance(np_maj, paired_minority, closest_paired_majority)
                synthetic_instances.append(synthetic_instance)

        return np.array(synthetic_instances)
