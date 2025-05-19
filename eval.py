import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics.cluster import contingency_matrix
from scipy.optimize import linear_sum_assignment
from itertools import combinations
from collections import Counter

def clustering_metrics(true_labels, pred_labels):
    """
    Compute clustering metrics including Accuracy, ARI, and NMI.
    
    Args:
        true_labels: Ground truth class labels (list or numpy array).
        pred_labels: Predicted cluster labels (list or numpy array).
        
    Returns:
        dict: Dictionary containing accuracy, ARI, and NMI scores.
    """
    # Convert inputs to numpy arrays if they aren't already
    true_labels = np.asarray(true_labels)
    pred_labels = np.asarray(pred_labels)
    
    # --- Accuracy (with optimal label alignment) ---
    # Compute contingency matrix and find optimal mapping
    contingency = contingency_matrix(true_labels, pred_labels)
    row_ind, col_ind = linear_sum_assignment(-contingency)
    accuracy = contingency[row_ind, col_ind].sum() / len(true_labels)
    
    # --- ARI and NMI (no alignment needed) ---
    ari = adjusted_rand_score(true_labels, pred_labels)
    nmi = normalized_mutual_info_score(true_labels, pred_labels)
    
    return {
        'accuracy': accuracy,
        'ARI': ari,
        'NMI': nmi
    }




class Omega: # code from: https://github.com/isaranto/omega_index/blob/master/omega_index/Omega.py
    def __init__(self, comms1, comms2):
        self.nodes1 = self.get_node_assignment(comms1)
        self.nodes2 = self.get_node_assignment(comms2)
        self.nodes = list(set().union([node for i, com in comms2.items() for node in com],
                                      [node for i, com in comms1.items() for node in com]))
        J, K, N, obs, tuples1, tuples2 = self.observed()
        exp = self.expected(J, K, N, tuples1, tuples2)
        self.omega_score = self.calc_omega(obs, exp)

    def get_node_assignment(self, comms):
        """
        returns a dictionary with node-cluster assignments of the form {node_id :[cluster1, cluster_3]}
        :param comms:
        :return:
        """
        nodes = {}
        for i, com in comms.items():
            for node in com:
                try:
                    nodes[node].append(i)
                except KeyError:
                    nodes[node] = [i]
        return nodes

    def num_of_common_clusters(self, u, v, nodes_dict):
        """
        return the number of clusters in which the pair u,v appears in the
        :param u:
        :param v:
        :param nodes_dict:
        :return:
        """
        try:
            _sum = len(set(nodes_dict[u]) & set(nodes_dict[v]))
        except KeyError:
            _sum = 0
        return _sum

    def observed(self):
        N = 0
        tuples1 = {}
        J = 0
        for u, v in combinations(self.nodes, 2):
            N += 1
            n = self.num_of_common_clusters(u, v, self.nodes1)
            tuples1[(u, v)] = self.num_of_common_clusters(u, v, self.nodes1)
            J = n if n > J else J
        tuples2 = {}
        K = 0
        for u, v in combinations(self.nodes, 2):
            n = self.num_of_common_clusters(u, v, self.nodes2)
            tuples2[(u, v)] = self.num_of_common_clusters(u, v, self.nodes2)
            K = n if n > K else K
        obs = 0
        A = {j: 0 for j in range(min(J, K)+1)}
        for (u, v), n in tuples1.items():
            try:
                if n == tuples2[(u, v)]:
                    A[n] += 1
            except KeyError:
                pass
        obs = sum(A[j]/N for j in range(min(J, K)+1))
        return J, K, N, obs, tuples1, tuples2

    def expected(self, J, K, N, tuples1, tuples2):
        N1 = Counter(tuples1.values())
        N2 = Counter(tuples2.values())
        exp = sum((N1[j]*N2[j])/(N**2) for j in range(min(J, K)+1))
        return exp

    def calc_omega(self, obs, exp):
        if exp == obs == 1:
            return 1.0
        else:
            return (obs-exp)/(1-exp)
        

def fuzzy_clustering_metrics(true_communities, pred_communities):

    """
    Compute metrics for fuzzy/overlapping clustering, including Omega Index.
    
    Args:
        true_communities: Ground truth communities as a dictionary where
                         keys are community IDs and values are lists of items.
                         Example: {"com1": ["item1", "item2"], "com2": ["item2", "item3"]}
        pred_communities: Predicted communities in the same format as true_communities.
        
    Returns:
        dict: Dictionary containing Omega Index score.
    """
    # Try to import omega_index package
    
    # Calculate Omega Index
    omega = Omega(pred_communities, true_communities)
    omega_score = omega.omega_score
    
    return {
        'omega_index': omega_score
    }


# Example usage for fuzzy clustering
if __name__ == "__main__":
    # Example from the original compute_clustering_metrics
    true_labels = [0, 0, 1, 1, 2, 2]
    pred_labels = [1, 1, 0, 0, 0, 2]  # Cluster IDs are arbitrary!

    metrics = clustering_metrics(true_labels, pred_labels)
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"ARI: {metrics['ARI']:.3f}")
    print(f"NMI: {metrics['NMI']:.3f}")
    
    # Example for fuzzy clustering with overlapping communities
    # Using a more realistic example with some common clusters
    true_communities = {
        "com1": [0, 1, 2],
        "com2": [3, 4, 5],
        "com3": [5, 6, 7],
        "com4": [8, 9]
    }
    
    pred_communities = {
        "cluster1": [0, 1, 2],         # Exactly matches com1
        "cluster2": [3, 4, 5, 6],      # Includes all of com2 plus one element from com3
        "cluster3": [6, 7, 8],         # Overlaps with com3 and com4
        "cluster4": [9]                # Partial match with com4
    }
    

    fuzzy_metrics = fuzzy_clustering_metrics(true_communities, pred_communities)
    print(f"Omega Index: {fuzzy_metrics['omega_index']:.3f}" if fuzzy_metrics['omega_index'] is not None else "Omega Index: Not available")