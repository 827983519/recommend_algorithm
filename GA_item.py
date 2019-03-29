from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import pairwise_distances

def fresh_property(population_current):
    sum = 0
    for i in range(Population_size):
        population_current[i][1]['fitness'] = objective_function(population_current[i])
        sum += population_current[i][1]['fitness']

    population_current[0][1]['rate_fit'] = population_current[0][1]['fitness'] / sum
    population_current[0][1]['cumm_fit'] = population_current[0][1]['rate_fit']

    for i in range(Population_size):
        population_current[i][1]['rate_fit'] = population_current[i][1]['fitness'] / sum
        population_current[i][1]['cumm_fit'] = population_current[i][1]['rate_fit'] + population_current[i-1][1]['cumm_fit']                                                     
            
    
def objective_function(individual):
    distance = 0
    min_distance = 100000000
    for row in range(len(pca_data)):
        for i in range(Cluster_number):
            euclidean_distance = np.abs(pca_data.iloc[row].values - individual[0][i]).sum()
            if euclidean_distance < min_distance:
                min_distance = euclidean_distance
        distance += min_distance
    return distance
    

def select(population_current,population_next):
    for i in range(Population_size):
        rand = np.random.rand(1)
        if rand <= population_current[0][1]['cumm_fit']:
            population_next[i] = population_current[0]
        else:
            for j in range(Population_size):
                if population_current[j][1]['cumm_fit'] <= rand and population_current[j+1][1]['cumm_fit'] >= rand:
                    population_next[i] = population_current[j+1]
                    break
            
def crossover(population_next):
    for i in range(Population_size):
        rand = np.random.rand(1)
        if rand <= Probability_crossover:
            rand_cluster = np.random.randint(Cluster_number)
            p1_num = np.random.randint(Population_size)
            p2_num = np.random.randint(Population_size)
            p1 = population_next[p1_num]
            p2 = population_next[p2_num]
            c1 = p1
            c2 = p2
            c1[0] = np.vstack([p1[0][:rand_cluster,:],p2[0][rand_cluster:,:]])
            c2[0] = np.vstack([p2[0][:rand_cluster,:],p1[0][rand_cluster:,:]]) 
            test_c = [[],[]]            
            test_c[0].extend([objective_function(c1),objective_function(c2),objective_function(p1),objective_function(p2)])      
            test_c[1].extend([c1,c2,p1,p2])
            population_next[p1_num] = test_c[1][test_c[0].index(min(test_c[0]))]
            test_c[1] = test_c[1][:test_c[0].index(min(test_c[0]))] + test_c[1][test_c[0].index(min(test_c[0]))+1:]
            test_c[0].remove(min(test_c[0]))   
            population_next[p2_num] = test_c[1][test_c[0].index(min(test_c[0]))]

def mutation(population_next):
    for i in range(Population_size):
        rand = np.random.rand(1)
        if rand <= Probability_mutation:
            mutation_array = np.ones([Cluster_number,Dimension_number])
            for k in range(Cluster_number):
                rand_pick = np.random.randint(Population_size)
                mutation_array[k] = population_next[rand_pick][0][k]
            if objective_function([mutation_array]) < objective_function(population_next[i]):
                population_next[i][0] = mutation_array

def user_predict(train_data,label,euc,test_data,k,user_mean,movie_mean):
    prediction_set = []
    new = 0
    last = 0
    
    for i in range(len(test_data)):
        user = int(test_data.iloc[i].userId)
        movie = int(test_data.iloc[i].movieId)
        new = user

        if movie not in train_data.index:
            prediction_set.append(user_mean[user])
            continue 


        if new != last:
            cluster_label = label.loc[movie].label
            mean = movie_mean[movie]
            k_similar_index = []
            if len(label[label['label'] == cluster_label]) < k:
                k_similar_index.extend(euc[movie][label[label['label'] == cluster_label].index].index)
            else:
                k_similar_index.extend(euc[movie][label[label['label'] == cluster_label].index].sort_values(ascending=True)[:k].index)

        add_up = 0
        add_down = 0  

        for similar_index in k_similar_index:
            if train_data[user][similar_index] != 0:
                similar_mean = movie_mean[similar_index]
                add_up = add_up + euc[movie][similar_index] * (train_data[user][similar_index] - similar_mean)
                add_down = add_down + abs(euc[movie][similar_index])

        if add_down == 0:
            prediction = mean
        else:
            prediction = mean+add_up/add_down
            if(prediction > 5):
                prediction = 5
            if(prediction < 0):
                predition = 0
        prediction_set.append(prediction) 
        last = new      
    return prediction_set



error_set = []

#ga_file = open('ga','w')

rating = pd.read_csv('ratings.dat',sep='::',header=None,names=['userId','movieId','rating','timestamp'])

train,test = train_test_split(rating,test_size = 0.2,random_state=0)

test = test.sort_index()
item_based = train.pivot_table(index='movieId', columns='userId', values='rating')
user_mean = item_based.mean(axis=0)
movie_mean = item_based.mean(axis=1)

item_based = item_based.fillna(0)
pca = PCA(n_components=500)
pca_data = pd.DataFrame(pca.fit_transform(item_based),index=item_based.index)
min_max = []
min_max.append(pca_data.max())
min_max.append(pca_data.min())

cluster_num_set = [i for i in range(3,16,2)]

for cluster_num in range(3,15,1):
    print('cluster_num',cluster_num)
    Population_size = 50
    Cluster_number = cluster_num
    Dimension_number = 500
    iteration_num = 140
    Probability_crossover = 0.5
    Probability_mutation = 0.0001;
    population_current = []
    population_next = []
    for i in range(Population_size):
        gene_array = np.array([])
        for j in range(Dimension_number):
            gene = np.random.uniform(min_max[0][j],min_max[1][j],(Cluster_number,1))
            if len(gene_array) == 0:
                gene_array = gene
            else:
                gene_array = np.hstack([gene_array,gene])
        population_current.append([gene_array,{'rate_fit':0,'cumm_fit':0,'fitness':0}])

    population_next = population_current[:]    
    fresh_property(population_current)
    for i in range(iteration_num):
        print('iteration',i)
        select(population_current,population_next)
        crossover(population_next)
        mutation(population_next)
        fresh_property(population_next)
        population_current = population_next[:]
    kmeans = KMeans(n_clusters=cluster_num,init=population_next[0][0])
    kmeans.fit(pca_data)
    label = pd.DataFrame(kmeans.labels_,index = item_based.index,columns=['label'])
    route = str(cluster_num)+'.csv'
    label.to_csv(route)
    '''
    euc = pd.DataFrame(pairwise_distances(item_based,metric='euclidean'),index=item_based.index,columns=item_based.index)
    error = np.sqrt(mean_squared_error(test.rating,user_predict(item_based,label,euc,test,100,user_mean,movie_mean)))
    ga_file.write('cluseter_num ')
    ga_file.write(str(cluster_num))
    ga_file.write(': ')
    ga_file.write(str(error))
    ga_file.write('\n')

ga_file.close()
'''
        
