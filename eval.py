import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics.cluster import contingency_matrix
from scipy.optimize import linear_sum_assignment
from itertools import combinations
from collections import Counter
import subprocess
import re

def parse_minigraph_result(stat_path):
    nga50 = 0
    ng50 = 0
    length = 0
    rdup = 0
    with open(stat_path) as f:
        for line in f.readlines():
            if line.startswith('NG50'):
                try:
                    ng50 = int(re.findall(r'NG50\s*(\d+)', line)[0])
                except IndexError:
                    ng50 = 0
            if line.startswith('NGA50'):
                try:
                    nga50 = int(re.findall(r'NGA50\s*(\d+)', line)[0])
                except IndexError:
                    nga50 = 0
            if line.startswith('Length'):
                try:
                    length = int(re.findall(r'Length\s*(\d+)', line)[0])
                except IndexError:
                    length = 0
            if line.startswith('Rdup'):
                try:
                    rdup = float(re.findall(r'Rdup\s*(\d+\.\d+)', line)[0])
                except IndexError:
                    rdup = 0

    return ng50, nga50, length, rdup

def parse_yak_result(yakres_path):
    """
    Yak triobinning result files have following info:
    C       F  seqName     type      startPos  endPos    count
    C       W  #switchErr  denominator  switchErrRate
    C       H  #hammingErr denominator  hammingErrRate
    C       N  #totPatKmer #totMatKmer  errRate
    """
    switch_err = None
    hamming_err = None

    with open(yakres_path, 'r') as file:
        # Read all the lines and reverse them
        lines = file.readlines()
        reversed_lines = reversed(lines)

        for line in reversed_lines:
            if line.startswith('W'):
                switch_err = float(line.split()[3])
            elif line.startswith('H'):
                hamming_err = float(line.split()[3])

            if switch_err is not None and hamming_err is not None:
                break

    return switch_err, hamming_err

def run_yak(mat_yak, pat_yak, asm, outfile, yak_path, threads=8):
    cmd = f'{yak_path} trioeval -t{threads} {pat_yak} {mat_yak} {asm} > {outfile}'.split(' ')
    with open(outfile, 'w') as f:
        p = subprocess.Popen(cmd, stdout=f)
    return p

def parse_minigraph_for_full(report, save_path=None, directory=None, filename='0_minigraph.txt'):
    stat_path = report
    with open(stat_path) as f:
        report = f.read()
        print(report)

def run_minigraph(ref, asm, paf, minigraph_path=None):
    if minigraph_path:
        cmd = f'{minigraph_path} -t32 -xasm -g10k -r10k --show-unmap=yes {ref} {asm}'.split(' ')
    else:
        cmd = f'minigraph -t32 -xasm -g10k -r10k --show-unmap=yes {ref} {asm}'.split(' ')
    with open(paf, 'w') as f:
        p = subprocess.Popen(cmd, stdout=f)
    return p

def parse_pafs(idx, report, paf, paf_path=None):
    if paf_path and paf_path != 'None':
        cmd = f'k8 {paf_path} asmstat {idx} {paf}'.split()
    else:
        cmd = f'paftools.js asmstat {idx} {paf}'.split()

    with open(report, 'w') as f:
        p = subprocess.Popen(cmd, stdout=f)
    return p

def parse_real_results(mat_report, pat_report, mat_phs, pat_phs):
    ng50_m, nga50_m, length_m, rdup_m = parse_minigraph_result(mat_report)
    ng50_p, nga50_p, length_p, rdup_p = parse_minigraph_result(pat_report)
    switch_err_m, hamming_err_m = parse_yak_result(mat_phs)
    switch_err_p, hamming_err_p = parse_yak_result(pat_phs)
    print(f'Results:')
    print(f'Length: M:{"{:,}".format(length_m)} P:{"{:,}".format(length_p)} Avg:{"{:,}".format((length_m + length_p) // 2)}')
    print(f'Rdup: M:{rdup_m:.4f} P:{rdup_p:.4f} Avg:{(rdup_m + rdup_p) / 2:.4f}')
    print(f'NG50: M:{"{:,}".format(ng50_m)} P:{"{:,}".format(ng50_p)} Avg:{"{:,}".format((ng50_m + ng50_p) // 2)}')
    print(f'NGA50: M:{"{:,}".format(nga50_m)} P:{"{:,}".format(nga50_p)} Avg:{"{:,}".format((nga50_m + nga50_p) // 2)}')
    print(f'YAK Switch Err: M:{switch_err_m * 100:.4f}% P:{switch_err_p * 100:.4f}% Avg:{(switch_err_m + switch_err_p) / 2 * 100:.4f}%')
    print(f'YAK Hamming Err: M:{hamming_err_m * 100:.4f}% P:{hamming_err_p * 100:.4f}% Avg:{(hamming_err_m + hamming_err_p) / 2 * 100:.4f}%')
    #print(f'MERYL Switch Err: M:{mat_switch_error:.4f}% P:{pat_switch_error:.4f}% Avg:{(mat_switch_error + pat_switch_error) / 2:.4f}%')


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