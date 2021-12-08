import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


# We use D^2-sampling algorithm proposed in paper 'Practical Coreset Constructions for Machine Learning'.
def D2Sampling(data_vectors, k):
    B = []
    # randomly select k points to form B
    B.append(data_vectors[np.random.choice(len(data_vectors))])
    
    for _ in range(k - 1):
        p = np.zeros(len(data_vectors))
        for ind, x in enumerate(data_vectors):
            p[ind] = min_distance_to_B(x, B) ** 2
        p = p / sum(p) 
        B.append(data_vectors[np.random.choice(len(data_vectors), p=p)])
    
    return B

# Calculate the minimal distance from one point to B
def min_distance_to_B(x, B,require_index = False):
    min_dist = np.inf
    closest_point_index = -1
    for ind,b in enumerate(B):
        dist = np.linalg.norm(x - b)
        if dist < min_dist:
            min_dist = dist
            closest_point_index = ind
    if require_index == True:
        return min_dist,closest_point_index
    else:
        return min_dist

def BFL16_algorithm(data_vector, B, m):
    """
    BFL16 in paper 'New Frameworks for Offline and Streaming Coreset Constructions'
    """

    num_points_in_clusters = {i: 0 for i in range(len(B))}
    sum_distance_to_closest_cluster = 0
    for p in data_vector:
        min_dist, closest_index = min_distance_to_B(p, B,require_index=True)
        num_points_in_clusters[closest_index] += 1
        sum_distance_to_closest_cluster += min_dist ** 2

    # Set the probability of each points based on its squared distance to its closest coreset element and number of
    # points in its cluster. 
    Prob = np.zeros(len(data_vector))
    for i, p in enumerate(data_vector):
        min_dist, closest_index = min_distance_to_B(p, B,require_index=True)
        Prob[i] += min_dist ** 2 / (2 * sum_distance_to_closest_cluster)
        Prob[i] += 1 / (2 * len(B) * num_points_in_clusters[closest_index])

    # If the points is closer to the cluster center, it will have more chance to be chosen.
    chosen_indices = np.random.choice(len(data_vector), size=m, p=Prob) 
    weights = [1 / (m * Prob[i]) for i in chosen_indices]

    return [data_vector[i] for i in chosen_indices], weights 

def cost_2Means(data_vectors, coreset_vectors, sample_weight=None):
    kmeans = KMeans(n_clusters=2).fit(coreset_vectors, sample_weight=sample_weight)
    cost = 0
    for x in data_vectors:
        cost += min_distance_to_B(x, kmeans.cluster_centers_) ** 2
    return cost

def coreset_cost(data_vectors,B,M):
    '''
    Do the same thing as shown in the pervious part - to construct coreset.
    '''
    # Construct 10 coresets for the same size and select the best one (with lowest cost).
    coreset, coreset_weights = [None] * 10, [None] * 10
    for i in range(10):
        coreset[i], coreset_weights[i] = BFL16_algorithm(data_vectors, B=B, m=M)
    core_cost = [cost_2Means(data_vectors, coreset_vectors=coreset[i], sample_weight=coreset_weights[i]) for i in range(10)]
    return core_cost

def plot_cost(cost_whole,cost_3,cost_5,cost_10,cost_20):
    x = np.arange(5)
    costs = [cost_whole, np.min(cost_3), np.min(cost_5), np.min(cost_10), np.min(cost_20)]
    scale = np.floor(np.log10(max(costs)))

    labels = ['whole', 'm=3', 'm=5', 'm=10', 'm=20']
    color = ['#006d2c', '#74c476', '#74c476', '#74c476', '#74c476']
    hatch=['', '/', '.', '\\', '']
    # plt.bar(x, costs, yerr=yerr, color=color, hatch=hatch)

    def autolabel(rects,cost):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height() / (10 ** scale)
            ax.annotate('{0:.2f}'.format(cost),
                        xy=(rect.get_x() + rect.get_width() / 2, height * (10 ** scale)),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)



    fig, ax = plt.subplots(figsize=(9, 2), dpi=300)
    for i in range(len(x)):
        rects = ax.bar(x[i], costs[i], label=labels[i], color=color[i], hatch=hatch[i])#, yerr=[[yerr[0][i]], [yerr[1][i]]])
        autolabel(rects,costs[i])

    ax.set_ylim([1,1e2])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel('Cost (lower is better)')
    ax.set_title('Iris - coreset')


def plot_scatter(iris_df,B_df = None,core_df = None,D1=None,D2=None):
    # coreset
    colours = ['blue', 'orange']
    species = ['setosa', 'versicolor']

    for i in range(0, 2):    
        species_df = iris_df[iris_df['label'] == i]    
        plt.scatter(        
            species_df[D1],        
            species_df[D2],
            color=colours[i],        
            alpha=0.5,        
            label=species[i]   
        )
    if B_df is not None:
        #species_df = iris_df[iris_df['species'] == i]    
        plt.scatter(        
            B_df[D1],        
            B_df[D2],
            color='red',        
            alpha=0.7,        
            label='cluster centerids'  
        )
        
        
    
    
    if core_df is not None:
        plt.scatter(        
            core_df[D1],        
            core_df[D2],
            color='green',        
            alpha=1,        
            label='coreset'  
        )

    plt.xlabel(D1)
    plt.ylabel(D2)
    plt.title('Iris dataset: '+ D1 + 'VS.' + D2)
    plt.legend(loc='lower right')
    plt.show()

def cost_on_classical(one,other,data_vectors):
    
    cc = [one,other]
    cost_after_qaoa = 0
    for x in data_vectors:
        cost_after_qaoa += min_distance_to_B(x, cc) ** 2
    return cost_after_qaoa

def plot_cost_qaoa(cost_whole,cost_qaoa):
    x = np.arange(2)
    costs = [cost_whole, cost_qaoa ]
    scale = np.floor(np.log10(max(costs)))

    labels = ['on whole dataset', 'on 2 cluster centerids from QAOA']
    color = ['#006d2c', '#74c476']
    hatch=['', '/']
    
    def autolabel(rects,cost):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height() / (10 ** scale)
            ax.annotate('{0:.2f}'.format(cost),
                        xy=(rect.get_x() + rect.get_width() / 2, height * (10 ** scale)),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)
    
    fig, ax = plt.subplots(figsize=(9, 2), dpi=300)
    for i in range(len(x)):
        rects = ax.bar(x[i], costs[i], label=labels[i], color=color[i], hatch=hatch[i])#, yerr=[[yerr[0][i]], [yerr[1][i]]])
        autolabel(rects,costs[i])

    ax.set_ylim([1,1e2])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel('Cost (lower is better)')
    ax.set_title('Cost - using 2Means Clustering VS. using 2 cluster centerids from QAOA')
