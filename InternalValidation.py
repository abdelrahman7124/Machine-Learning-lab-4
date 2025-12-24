import numpy as np
def calculate_silhouette_score(X, labels):
    n_samples = X.shape[0]
    unique_labels = np.unique(labels)

    if len(unique_labels) <= 1 or len(unique_labels) == n_samples:
        return 0.0
    distances = np.sqrt(((X[:, np.newaxis, :] - X[np.newaxis, :, :]) ** 2).sum(axis=2))
    s_values = []
    for i in range(n_samples):
        # a(i): mean intra-cluster distance
        cluster_i = labels[i]
        #indecies of other points in the same cluster
        mask_same = (labels == cluster_i)
        mask_same[i] = False

        if np.sum(mask_same) == 0:
            a_i = 0
        else:
            a_i = np.mean(distances[i, mask_same])

        # b(i): min mean inter-cluster distance
        b_i = np.inf
        for label in unique_labels:
            if label == cluster_i:
                continue
            #indecies of other points in the other cluster
            mask_other = (labels == label)
            if np.sum(mask_other) > 0:
                mean_dist_other = np.mean(distances[i, mask_other])
                if mean_dist_other < b_i:
                    b_i = mean_dist_other

        if max(a_i, b_i) == 0:
            s_i = 0
        else:
            s_i = (b_i - a_i) / max(a_i, b_i)

        s_values.append(s_i)

    return np.mean(s_values)

def calculate_davies_bouldin_index(X, labels):
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    
    if n_clusters <= 1:
        return 0.0
    
    centroids = []
    for label in unique_labels:
        cluster_points = X[labels == label]
        centroids.append(np.mean(cluster_points, axis=0))
    centroids = np.array(centroids)
    
    # Calculate average intra-cluster distances (S_i)
    S = []
    for i, label in enumerate(unique_labels):
        cluster_points = X[labels == label]
        distances = np.sqrt(((cluster_points - centroids[i]) ** 2).sum(axis=1))
        S.append(np.mean(distances))
    
    # Calculate Davies-Bouldin Index
    db_values = []
    for i in range(n_clusters):
        max_ratio = 0
        for j in range(n_clusters):
            if i != j:
                # Distance between centroids
                centroid_dist = np.sqrt(((centroids[i] - centroids[j]) ** 2).sum())
                if centroid_dist > 0:
                    ratio = (S[i] + S[j]) / centroid_dist
                    max_ratio = max(max_ratio, ratio)
        db_values.append(max_ratio)
    
    return np.mean(db_values)


def calculate_calinski_harabasz_index(X, labels):

    n_samples = X.shape[0]
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    
    if n_clusters <= 1 or n_clusters == n_samples:
        return 0.0
    
    # Overall centroid
    overall_centroid = np.mean(X, axis=0)
    
    # Between-cluster dispersion (BCSS)
    bcss = 0
    for label in unique_labels:
        cluster_points = X[labels == label]
        n_k = len(cluster_points)
        cluster_centroid = np.mean(cluster_points, axis=0)
        bcss += n_k * np.sum((cluster_centroid - overall_centroid) ** 2)
    
    # Within-cluster dispersion (WCSS)
    wcss = 0
    for label in unique_labels:
        cluster_points = X[labels == label]
        cluster_centroid = np.mean(cluster_points, axis=0)
        wcss += np.sum((cluster_points - cluster_centroid) ** 2)
    
    # Calinski-Harabasz Index
    if wcss == 0:
        return 0.0
    
    ch_index = (bcss / (n_clusters - 1)) / (wcss / (n_samples - n_clusters))
    
    return ch_index