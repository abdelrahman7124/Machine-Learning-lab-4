import numpy as np


# Rule = RandIndex - ExpectedRandIndex / (max(RnadIndex) - ExpectedRandIndex)
# max RandIndex = 1
def AdjustedRandIndex(X, labels):  
    n = len(labels)
    if n != len(X):
        raise ValueError("labels and X must have the same length.")
    
    TP = TN = FP = FN = 0
    for i in range(n):
        for j in range(i + 1, n):
            same_true = labels[i] == labels[j]
            same_pred = X[i] == X[j] 
            
            if same_true and same_pred:
                TP += 1
            elif not same_true and not same_pred:
                TN += 1
            elif same_true and not same_pred:
                FN += 1 
            else:  
                FP += 1 
    
    rand_index = (TP + TN) / (TP + TN + FP + FN)
    
    N = n * (n - 1) / 2  
    a = TP + FN  
    b = TP + FP  
    
    expected_rand_index = (a * b + (N - a) * (N - b)) / (N * N)
    
    ari = (rand_index - expected_rand_index) / (1 - expected_rand_index)
    
    return ari


## RULE = MI(X, Y) / 0.5 * (H(X) + H(Y))
### MI(X, Y) = sigma (from 1 to len x ) { sigma (from 1 to len y ) [ P(x,y) log ( P(x,y) / P(x)P(y) )] }
def normalized_mutual_info(X, labels):

    labels = np.array(labels)
    X = np.array(X)
    n_samples = len(labels)
    
    true_classes = np.unique(labels)
    pred_classes = np.unique(X)
    
    # Compute probabilities P(x) and P(y)
    pi = np.array([np.sum(labels == c) / n_samples for c in true_classes])
    pj = np.array([np.sum(X == c) / n_samples for c in pred_classes])
    
    # Compute Mutual Information (MI)
    mi = 0.0
    for i, c_true in enumerate(true_classes):
        for j, c_pred in enumerate(pred_classes):
            # compute P(x,y)
            p_ij = np.sum((labels == c_true) & (X == c_pred)) / n_samples
            if p_ij > 0:  
                mi += p_ij * np.log(p_ij / (pi[i] * pj[j]))
    
    # Compute entropies
    hi = -np.sum([p * np.log(p) for p in pi if p > 0])
    hj = -np.sum([p * np.log(p) for p in pj if p > 0])
    
    # Normalized Mutual Information
    nmi = mi / ((hi + hj) / 2)
    
    return nmi


## Rule = 1/N * sum (from i=1 to N) { max (from j != i) [ number of data points in cluster i with true label j ] }
def purity(X,labels):
    labels = np.array(labels)
    X = np.array(X)
    n_samples = len(labels)
    
    unique_pred_classes = np.unique(X)
    total_correct = 0
    
    for c_pred in unique_pred_classes:
        mask = (X == c_pred)
        true_labels_in_cluster = labels[mask]
        if len(true_labels_in_cluster) == 0:
            continue
        most_common_label = np.bincount(true_labels_in_cluster).argmax()
        correct_count = np.sum(true_labels_in_cluster == most_common_label)
        total_correct += correct_count
    
    purity_score = total_correct / n_samples
    return purity_score

def confusionMatrix(cluster_labels, true_labels):
    unique_clusters = np.unique(cluster_labels)
    confusion_matrices = []
    
    for cluster in unique_clusters:
        # Get indices of points in this cluster
        cluster_mask = (cluster_labels == cluster)
        
        # Find majority class in this cluster
        cluster_true_labels = true_labels[cluster_mask]
        unique, counts = np.unique(cluster_true_labels, return_counts=True)
        majority_class = unique[np.argmax(counts)]
        
        
        # True Positives: in cluster AND majority class
        tp = np.sum((cluster_labels == cluster) & (true_labels == majority_class))
        
        # False Positives: in cluster BUT NOT majority class
        fp = np.sum((cluster_labels == cluster) & (true_labels != majority_class))
        
        # False Negatives: NOT in cluster BUT is majority class
        fn = np.sum((cluster_labels != cluster) & (true_labels == majority_class))
        
        # True Negatives: NOT in cluster AND NOT majority class
        tn = np.sum((cluster_labels != cluster) & (true_labels != majority_class))
        
        cm = np.array([[tp, fp],
                       [fn, tn]])
        confusion_matrices.append(cm) 
    return confusion_matrices






    


