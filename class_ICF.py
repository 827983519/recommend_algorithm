from sklearn.metrics import pairwise_distances
import copy

class ICF:
    def __init__(self,neighbor=10,similarity='pearson'):
        self.neighbor = neighbor
        self.similarity = similarity
    
    def jaccard(self,n1,n2):
        up = len(np.intersect1d(n1,n2))
        down = len(np.union1d(n1,n2))
        return up/down

    def calculate_jaccard(self,data):
        index_table = dict(zip(data.index.values,[0 for i in range(len(data.index))]))
        for index,row in data.iterrows():
            index_table[index] =row.dropna().index.values
        jaccard_table = pd.DataFrame(index=data.index,columns=data.index).to_dict()
        for i in data.index.values:
            print(i)
            for j in data.index.values:
                if i==j:
                    jaccard_table[i][j]=1
                    continue
                if jaccard_table[i][j] >0:
                    jaccard_table[j][i] = jaccard_table[i][j]
                    continue
                jaccard_table[j][i] = self.jaccard(index_table[j],index_table[i])
        return jaccard_table

    def calculate_similarity(self,data):
        if self.similarity == 'pearson':
            return  pd.DataFrame(1- pairwise_distances(data.fillna(0),metric='correlation'),columns=data.index,index=data.index).to_dict()
        if self.similarity == 'IIF':
            IIF = pd.read_csv('IIF.csv',index_col=0)
            IIF = pd.DataFrame(IIF.values,index=IIF.index,columns=IIF.index)
            return IIF.to_dict()
        
        if self.similarity == 'jaccard':
            return self.calculate_jaccard(data)

    def find_nearset_neighbor(self,movieId):
        top_neighbor = sorted(self.similarity_set[movieId].items(), key=lambda e:e[1], reverse=True)[1:1+self.neighbor]
        similar_index = [i[0] for i in top_neighbor]
        return similar_index

    def fit(self,data):     
        self.movie_mean = data.mean(axis=1).to_dict()
        self.user_mean = data.mean(axis=0).to_dict()
        self.dataset = data.fillna(0).to_dict()
        self.similarity_set = self.calculate_similarity(data)
        
     
    def calculate_Fscore(self,test_data,Feature_name,topN):
        self.precision = []
        self.recall = []
        self.Fscore = []
        test_data = test_data[Feature_name]
        user_feature = Feature_name[0]
        movie_fearure = Feature_name[1]
        for user in self.predict_set:
            top = sorted(self.predict_set[user].items(),key = lambda items:items[1],reverse=True)[:topN]
            top_N = [i[0] for i in top]
            test_set = test.loc[test[user_feature]==user,movie_fearure].values
            inter = len(np.intersect1d(top_N,test_set))
            precision = inter/topN
            #recall = inter/len(test_set)
           # fscore = (1+0.25)*(precision*recall)/(0.25*precision+recall)
            self.precision.append(precision)
            #self.recall.append(recall)
           # self.Fscore.append(fscore)
        
    def predict(self,test_data,Feature_name):
        predict_set = []
        user_feature = Feature_name[0]
        movie_fearure = Feature_name[1]
        test_data = test_data[Feature_name]
        test_data = test_data.to_dict()
      
        for user,movie in zip(test_data[user_feature].values(),test_data[movie_fearure].values()):
            if user not in self.dataset:
                predict_set.append(self.movie_mean[movie])
                continue

            if movie not in self.dataset[user]:
                predict_set.append(self.user_mean[user])
                continue
        
            k_similar = self.find_nearset_neighbor(movie)        
            mean = self.movie_mean[movie]
            add_up = 0
            add_down = 0
            for k_index in k_similar:
                if self.dataset[user][k_index] != 0:         
                    similar_mean = self.movie_mean[k_index]
                    add_up += self.similarity_set[movie][k_index]*(self.dataset[user][k_index]-similar_mean)
                    add_down += abs(self.similarity_set[movie][k_index])

            if add_down == 0:
                prediction = mean
                
            else:
                prediction = mean+add_up/add_down
                if(prediction > 5):
                    prediction = 5

                if(prediction < 0):
                    predition = 0
            predict_set.append(prediction) 
        return predict_set
    
    
    def predict_whole(self): 
        predict_dict = copy.deepcopy(self.dataset)
        for movie in self.dataset[list(self.dataset.keys())[0]]:
            print(movie) 
            k_similar = self.find_nearset_neighbor(movie)
            mean = self.movie_mean[movie]
            now = movie
            for user in self.dataset.keys():  
                add_up = 0
                add_down = 0
                for k_index in k_similar:
                    if self.dataset[user][k_index] != 0:
                        similar_mean = self.movie_mean[k_index]
                        add_up += self.similarity_set[movie][k_index]*(self.dataset[user][k_index]-similar_mean)
                        add_down += abs(self.similarity_set[movie][k_index])

                if add_down == 0:
                    prediction = mean

                else:
                    prediction = mean+add_up/add_down
                    if(prediction > 5):
                        prediction = 5

                    if(prediction < 0):
                        predition = 0
                predict_dict[user][movie] = prediction
        self.predict_dict = predict_set
        return self.predict_dict



