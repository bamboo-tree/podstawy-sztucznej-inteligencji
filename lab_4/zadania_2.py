import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

example = np.random.randn(500, 2)
example[:250,:] -= 2
example[250:,:] += 2
example_pcaed = PCA(2).fit_transform(example)

fig, ax = plt.subplots(1, 2, figsize=(20,10))
ax[0].scatter(example[:,0], example[:,1])
ax[1].scatter(example_pcaed[:,0], example_pcaed[:,1])

lim = 6
ax[0].set_xlim([-lim, lim])
ax[0].set_ylim([-lim, lim])
ax[1].set_xlim([-lim, lim])
ax[1].set_ylim([-lim, lim])
ax[0].set_xlabel('x')
ax[0].set_ylabel('y')
ax[1].set_xlabel('PC 1')
ax[1].set_ylabel('PC 2')
ax[0].set_title('Dane pierwotne')
ax[1].set_title('Dane po PCA')