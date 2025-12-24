import numpy as np
from enum import Enum
from scipy.stats import multivariate_normal

## ///
##AutoEncoder Implementation
##///

# Activation functions and their derivatives
def relu(x):
    return np.maximum(0, x)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def tanh(x):
    return np.tanh(x)

def relu_der(x):
    return (x > 0).astype(float)

def sigmoid_der(x):
    s = sigmoid(x)
    return s * (1 - s)

def tanh_der(x):
    return 1 - np.tanh(x) ** 2

# Dense Layer Class
class DenseLayer:
    def __init__(self, input_dim, output_dim, activation):
        self.W = np.random.randn(input_dim, output_dim) * np.sqrt(2 / input_dim)
        self.b = np.zeros((1, output_dim))
        if activation == 'tanh':
            self.activation = tanh
            self.activation_der = tanh_der
        elif activation == 'sigmoid':
            self.activation = sigmoid
            self.activation_der = sigmoid_der
        elif activation == 'relu':
            self.activation = relu
            self.activation_der = relu_der

    def forward(self, X):
        self.X = X
        self.Z = X @ self.W + self.b

        self.A = self.activation(self.Z)

        return self.A

    def backward(self, dA, l2_lambda):
        dz = self.activation_der(self.Z)
        dZ = dA * dz

        dW = self.X.T @ dZ + l2_lambda * self.W
        db = np.sum(dZ, axis=0, keepdims=True)
        dX = dZ @ self.W.T

        return dX, dW, db



#autoencoder Class
class Autoencoder:
    def __init__(self, layers, bottleneck_idx):
        self.layers = layers
        self.bottleneck_idx = bottleneck_idx

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X
    
    def encode(self, X):
        encoded = X
        # Forward pass only up to bottleneck layer
        for i, layer in enumerate(self.layers):
            encoded = layer.forward(encoded)
            if i == self.bottleneck_idx:  
                return encoded
        return encoded

    def backward(self, dLoss, l2_lambda):
        # reverse to loop through layers from the last 
        for layer in reversed(self.layers):
            dLoss, dW, db = layer.backward(dLoss, l2_lambda)
            layer.dW = dW
            layer.db = db
            
    def mse_loss(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2) / 2
    
    def create_batches(self, X, batch_size):
        batches = []
        for i in range(0, len(X), batch_size):
            batches.append(X[i:i + batch_size])
        return batches

    def train(self, X, epochs, batch_size, learning_rate, l2_lambda):
        losses = []

        for epoch in range(epochs):
            epoch_loss = 0
            for batch in self.create_batches(X, batch_size):
                y_pred = self.forward(batch)

                loss = self.mse_loss(batch, y_pred)
                epoch_loss += loss

                dLoss = (y_pred - batch) / batch.shape[0]

                self.backward(dLoss, l2_lambda)

                for layer in self.layers:
                    layer.W -= learning_rate * layer.dW
                    layer.b -= learning_rate * layer.db

            epoch_loss /= (len(X) / batch_size)
            losses.append(epoch_loss)

            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {epoch_loss}")

        return losses
    
    
    
## -------------------------------------------------------------
##PCA Implementation
##--------------------------------------------------------------
class PCA:
    def __init__(self, desired_dim):
        self.desired_dim =desired_dim 
        self.principal_components = None
        self.mean_ = None
        self.explained_variance_ratio_ = None

    def fit(self, X):
        # 1. Zero center the data
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_

        # Rowvar=False because rows are samples, cols are features
        cov_matrix = np.cov(X_centered, rowvar=False)

        # eigh is used for symmetric matrices
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        sorted_indices = np.argsort(eigenvalues)[::-1]
        sorted_eigenvalues = eigenvalues[sorted_indices]
        sorted_eigenvectors = eigenvectors[:, sorted_indices]

        self.principal_components = sorted_eigenvectors[:, :self.desired_dim]

        # 6. Explained variance ratio
        total_variance = np.sum(sorted_eigenvalues)
        self.explained_variance_ratio_ = sorted_eigenvalues[:self.desired_dim] / total_variance

    def transform(self, X):
        X_centered = X - self.mean_
        return np.dot(X_centered, self.principal_components)

    def inverse_transform(self, X_transformed):
        return np.dot(X_transformed, self.principal_components.T) + self.mean_

    def compute_reconstruction_error(self, X, X_reconstructed):
        #MSE
        return np.mean(np.square(X - X_reconstructed))
## -------------------------------------------------------------
## Kmeans Implementation
##--------------------------------------------------------------

class initMethod(Enum):
    RANDOM = 'random'
    KMEANS_PP = 'kmeans++'
class KMeans:
    def __init__(self, num_clusters, max_iter = 300, tol = 1e-6, init_method = initMethod.RANDOM, random_state = None):

        self.init_method = init_method
        self.centroids_ = None
        self.num_clusters =num_clusters 
        self.max_iter = max_iter
        self.tol = tol
        self.inertia_history_ = []
        self.random_state = random_state
        if random_state:
            np.random.seed(random_state)

    def _initialize_centroids(self, X):
        n_samples, n_features = X.shape

        if self.init_method == initMethod.RANDOM:

            #replace=False to avoid duplicate centroids
            indices = np.random.choice(n_samples, self.num_clusters, replace=False)
            self.centroids_ = X[indices]

        elif self.init_method == initMethod.KMEANS_PP: 
            #first one is random
            centroids = [X[np.random.randint(n_samples)]]

            for  i in range(1, self.num_clusters):
                #for everypoint x get the squared distances from every centroid and take the minimum
                distances = np.array([min([np.sum((x - c) ** 2) for c in centroids]) for x in X])
                weighted_prob = distances / np.sum(distances)
                next_centroid_idx = np.random.choice(n_samples, p=weighted_prob)
                centroids.append(X[next_centroid_idx])

            self.centroids_ = np.array(centroids)

    def euclidean_distance(self, X, centroids):
        # X: n*d , centroids: K*d
        num_points = X.shape[0]
        num_centroids = centroids.shape[0]
        distances = np.zeros((num_points, num_centroids))
        for i in range(num_points):
            for j in range(num_centroids):
                diff = X[i] - centroids[j]
                distances[i, j] = np.sqrt(np.sum(diff**2))

        return distances #Points*centroids
    def fit(self, X):
        self.inertia_history_ = []
        self._initialize_centroids(X)

        for i in range(self.max_iter):
            distances = self.euclidean_distance(X, self.centroids_)
            labels = np.argmin(distances, axis=1) # n elements and each has the cluster(centroid) number(index)

            # compute inertia
            inertia = np.sum((X - self.centroids_[labels]) ** 2)
            self.inertia_history_.append(inertia)

            new_centroids = np.zeros((self.num_clusters, X.shape[1])) # empty NP array
            for i in range(self.num_clusters):
                points_in_cluster = X[labels == i]
                if len(points_in_cluster) > 0:
                    new_centroids[i] = np.mean(points_in_cluster, axis=0)
                else:
                    # If a cluster is empty, keep the old position -- if the cluster have no points assigned to it
                    new_centroids[i] = self.centroids_[i]

            # Check convergence
            change_in_centroids = np.sum((self.centroids_ - new_centroids) ** 2)
            if change_in_centroids < self.tol:
                self.centroids_ = new_centroids
                break

            self.centroids_ = new_centroids

        self.labels_ = labels
        self.inertia_ = self.inertia_history_[-1]
        
## -------------------------------------------------------------
## GMM Implementation
##--------------------------------------------------------------
class GaussianMixtureModel:
    def __init__(self, n_components=3, covariance_type='full', tol=1e-4, max_iter=100, reg_covar=1e-6):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.tol = tol
        self.max_iter = max_iter
        self.reg_covar = reg_covar
        
        self.means_ = None
        self.covariances_ = None
        self.weights_ = None
        self.log_likelihood_history_ = []
        self.converged_ = False

    def _initialize_parameters(self, X):
        n_samples, n_features = X.shape
        indices = np.random.choice(n_samples, self.n_components, replace=False)
        self.means_ = X[indices]
        self.weights_ = np.full(self.n_components, 1 / self.n_components)
        
        if self.covariance_type == 'full':
            self.covariances_ = np.array([np.eye(n_features) for _ in range(self.n_components)])
        elif self.covariance_type == 'tied':
            self.covariances_ = np.eye(n_features)
        elif self.covariance_type == 'diag':
            self.covariances_ = np.ones((self.n_components, n_features))
        elif self.covariance_type == 'spherical':
            self.covariances_ = np.ones(self.n_components)

    def _e_step(self, X):
        n_samples, n_features = X.shape
        weighted_log_probs = np.zeros((n_samples, self.n_components))
        
        for k in range(self.n_components):
            if self.covariance_type == 'full':
                cov = self.covariances_[k] + np.eye(n_features) * self.reg_covar
            elif self.covariance_type == 'tied':
                cov = self.covariances_ + np.eye(n_features) * self.reg_covar
            elif self.covariance_type == 'diag':
                cov = np.diag(self.covariances_[k] + self.reg_covar)
            elif self.covariance_type == 'spherical':
                cov = np.eye(n_features) * (self.covariances_[k] + self.reg_covar)
            
            try:
                weighted_log_probs[:, k] = np.log(self.weights_[k] + 1e-10) + \
                                           multivariate_normal.logpdf(X, mean=self.means_[k], cov=cov)
            except np.linalg.LinAlgError:
                weighted_log_probs[:, k] = -np.inf

        log_prob_norm = self._log_sum_exp(weighted_log_probs)
        log_likelihood = np.sum(log_prob_norm)
        
        with np.errstate(under='ignore'):
            responsibilities = np.exp(weighted_log_probs - log_prob_norm[:, np.newaxis])
            
        return log_likelihood, responsibilities

    def _m_step(self, X, responsibilities):
        n_samples, n_features = X.shape
        Nk = responsibilities.sum(axis=0) + 1e-10
        
        self.weights_ = Nk / n_samples
        self.means_ = np.dot(responsibilities.T, X) / Nk[:, np.newaxis]
        
        if self.covariance_type == 'full':
            for k in range(self.n_components):
                diff = X - self.means_[k]
                self.covariances_[k] = np.dot(responsibilities[:, k] * diff.T, diff) / Nk[k]
        elif self.covariance_type == 'tied':
            avg_cov = np.zeros((n_features, n_features))
            for k in range(self.n_components):
                diff = X - self.means_[k]
                avg_cov += np.dot(responsibilities[:, k] * diff.T, diff)
            self.covariances_ = avg_cov / n_samples
        elif self.covariance_type == 'diag':
            for k in range(self.n_components):
                diff = X - self.means_[k]
                self.covariances_[k] = np.sum(responsibilities[:, k][:, np.newaxis] * (diff ** 2), axis=0) / Nk[k]
        elif self.covariance_type == 'spherical':
            for k in range(self.n_components):
                diff = X - self.means_[k]
                diff_sq_norm = np.sum(diff**2, axis=1)
                self.covariances_[k] = np.sum(responsibilities[:, k] * diff_sq_norm) / (Nk[k] * n_features)

    def fit(self, X):
        self._initialize_parameters(X)
        self.log_likelihood_history_ = []
        
        for i in range(self.max_iter):
            prev_ll = self.log_likelihood_history_[-1] if self.log_likelihood_history_ else -np.inf
            
            ll, resp = self._e_step(X)
            
            if np.isnan(ll):
                print(f"Warning: Training diverged at iteration {i}. Stopping.")
                break
                
            self.log_likelihood_history_.append(ll)
            
            if np.abs(ll - prev_ll) < self.tol:
                self.converged_ = True
                break
            
            self._m_step(X, resp)
        return self

    def _log_sum_exp(self, log_probs):
        max_log = np.max(log_probs, axis=1, keepdims=True)
        max_log[np.isinf(max_log)] = 0 
        return np.squeeze(max_log) + np.log(np.sum(np.exp(log_probs - max_log), axis=1))

    def bic(self, X):
        if not self.log_likelihood_history_: return np.inf
        n, d = X.shape
        return self._n_parameters(d) * np.log(n) - 2 * self.log_likelihood_history_[-1]

    def aic(self, X):
        if not self.log_likelihood_history_: return np.inf
        d = X.shape[1]
        return 2 * self._n_parameters(d) - 2 * self.log_likelihood_history_[-1]
    
    def _n_parameters(self, n_features):
        if self.covariance_type == 'full':
            cov_p = self.n_components * n_features * (n_features + 1) / 2
        elif self.covariance_type == 'diag':
            cov_p = self.n_components * n_features
        elif self.covariance_type == 'tied':
            cov_p = n_features * (n_features + 1) / 2
        elif self.covariance_type == 'spherical':
            cov_p = self.n_components
        return int(cov_p + self.n_components * n_features + (self.n_components - 1))
