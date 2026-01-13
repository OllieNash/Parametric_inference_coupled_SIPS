import matplotlib.pyplot as plt  


def plot_likelihood(estimates, estiamte, likelihood):
    plt.plot(estimates,likelihood,color='blue',marker='o',markersize=1,linestyle='-')
    #plt.ylim(-20,5)
    plt.xlabel("Parameter values")
    plt.ylabel("Likelihood")
    plt.title('Likelihood vs parameter')
    plt.axvline(estiamte, color='g', linestyle='dashed', linewidth=2, label='Estimated gamma')
    plt.show()


