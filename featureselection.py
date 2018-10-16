import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt



def featureselection():

    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data" #comment it
    df = pd.read_csv(url, names=['sepal length', 'sepal width', 'petal length', 'petal width', 'target'])  #replaces
    print(df)

    features = ['sepal length', 'sepal width', 'petal length', 'petal width']   #replace features with names we give for features

    # Separating out the features
    x = df.loc[:, features].values
    # Separating out the target
    y = df.loc[:, ['target']].values  #replace target with timestamp

    x = StandardScaler().fit_transform(x)

    from sklearn.decomposition import PCA

    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(x)

    principalDf = pd.DataFrame(data=principalComponents, columns=['Feature 1', 'Feature 2'])

    print(pca.explained_variance_)  # eigenvalue
    print(pca.components_)  # eigenvector
    #plt.plot(pca.components_)
    print(x[:,0])
    plt.scatter(x[:, 0], x[:, 1], alpha=0.2)
    for length, vector in zip(pca.explained_variance_, pca.components_):
        v = vector * 3 * np.sqrt(length)
        draw_vector(pca.mean_, pca.mean_ + v)
    plt.axis('equal')
    plt.show()

def draw_vector(v0, v1, ax=None):
    ax = ax or plt.gca()
    arrowprops=dict(arrowstyle='->',
                    linewidth=2,
                    shrinkA=0, shrinkB=0)
    ax.annotate('', v1, v0, arrowprops=arrowprops)

featureselection()