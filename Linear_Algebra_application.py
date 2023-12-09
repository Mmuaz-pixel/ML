import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 

# The most important problem linear algebra solve is too much columns by dimensionality reduction 

from sklearn.decomposition import PCA 

#PCA, Principal component analysis is a dimensional reduction technique used to reduce the number of features in such a way that many columns can be projected in one column having influence of all combined 

rng = np.random.RandomState(2)

X = np.dot(rng.rand(2,2), rng.randn(2,200)).T

# .dot performs dot product 
# .rand(2,2) makes a 2x2 matrics with values between 0 and 1 
# .randn(2,200) makes a 2x200 matric with random values such that mean = 0 and S.D = 1
# .T is the transpose 

# print(X.shape) # 200x2

# plt.scatter(X[:,0], X[:,1])
# plt.axis('equal')
# plt.show()

pca = PCA(n_components=2)
pca.fit(X)

def draw_vectors(v0, v1, ax=None):
    if ax is None:
        ax = plt.gca()
    arrow_props = dict(arrowstyle='->', 
                       linewidth = 2, 
                       shrinkA = 0, 
                       shrinkB = 0
                       )
    ax.annotate('', v1, v0, arrowprops=arrow_props)

# print("components: ", pca.components_)
# print("variance: ",pca.explained_variance_)
# print("mean: ",pca.mean_)
# print(pca.n_components_)

plt.scatter(X[:,0], X[:,1])
plt.scatter(pca.components_[:,0], pca.components_[:,1], color='blue', label='components')
plt.scatter(pca.explained_variance_[0], pca.explained_variance_[1], color='red', label='variance')
plt.scatter(pca.mean_[0], pca.mean_[1], color='orange', label='mean')

for length, vectors in zip(pca.explained_variance_, pca.components_): # length = expalined variance 
    v = vectors * 3 * np.sqrt(length) # this step is primarliy for visualization purpose 

    print("Length: ", length)
    print("component: ", vectors)
    print("v: ", v)
    draw_vectors(pca.mean_, pca.mean_ + v)

plt.legend()
plt.axis('equal')
plt.show()


pca = PCA(n_components=1)
pca.fit(X)
X_PCA = pca.transform(X) # transforms the 2 dimensional data to 1 (making a single principal component)
print("X PCA: ", X_PCA) 
X_new = pca.inverse_transform(X_PCA) # it again derives the 2 dimensional data but using the principal component this time 
print("X NEW: ", X_new)
plt.scatter(X[:,0], X[:,1])
plt.scatter(X_new[:,0], X_new[:,1], alpha=0.8)
plt.axis('equal')
plt.show()