from sklearn.metrics import pairwise_distances
import copy

class ICF_topN: 
    
    def R_jaccard(self,I1,I2):
        intersect = 1.0 * len(np.intersect1d(I1,I2))
        if intersect == 0:
            return 0
        I1_g = 1.0 * len(I1) - intersect
        I2_g = 1.0 * len(I2) - intersect

        down = 1 + 1/intersect + I1_g/(1 + I1_g) + 1/(1 + I2_g)
        return 1/down


    def calculate_Rjaccard(self,data):
        index_table = dict(zip(data.index.values,[0 for i in range(len(data.index))]))
        for index,row in data.iterrows():
            index_table[index] =row.dropna().index.values
        rjaccard_table = pd.DataFrame(index=data.index,columns=data.index).to_dict()
        for i in rjaccard_table:
            for j in rjaccard_table:
                if i == j:
                    rjaccard_table[i][j]=1
                    rjaccard_table[j][i]=1
                    continue
                if rjaccard_table[i][j] > 0:
                    rjaccard_table[j][i] = rjaccard_table[i][j]
                    continue
                rjaccard_table[j][i] = self.R_jaccard(index_table[j],index_table[i])
        return rjaccard_table

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
        
        if self.similarity == 'rjaccard':
            return self.calculate_Rjaccard(data)

    def find_nearset_neighbor(self,movieId):
        top_neighbor = sorted(self.similarity_set[movieId].items(), key=lambda e:e[1], reverse=True)[1:1+self.neighbor]
        similar_index = [i[0] for i in top_neighbor]
        return similar_index

    def fit(self,data):     
        self.origin_data = data
        self.dataset = data.fillna(0).to_dict()
        self.similarity_set = self.calculate_similarity(data)
       
    
    def recommend(self,userId,topN):
        top = sorted(self.predict_set[userId].items(),key = lambda items:items[1],reverse=True)[:topN]
        top_N = [i[0] for i in top]
        return top_N
     
    def calculate_Fscore(self,test_data,topN):
        self.precision = []
        self.recall = []
        self.Fscore = []
        test_data = test_data[['userId','movieId']]
        for user in self.predict_set:
            #top = sorted(self.predict_set[user].items(),key = lambda items:items[1],reverse=True)[:topN]
            #top_N = [i[0] for i in top]
            top_N = self.recommend(user,topN) 
            test_set = test_data.loc[test_data['userId']==user,'movieId'].values
            if len(test_set)==0:
                continue
            inter = len(np.intersect1d(top_N,test_set))
            precision = inter/topN
            #recall = inter/len(test_set)
            #fscore = (1+0.25)*(precision*recall)/(0.25*precision+recall)
            self.precision.append(precision)
            #self.recall.append(recall)
            # self.Fscore.append(fscore)
    
    
    def predict_whole(self,user_list): 
        user_list = user_list.to_dict()
        predict_set = copy.deepcopy(self.dataset)
        for movie in self.dataset[list(self.dataset.keys())[0]]:
            print(movie) 
            k_similar = self.find_nearset_neighbor(movie)
            for user in self.dataset.keys(): 
                if predict_set[user][movie] > 0:
                    predict_set[user][movie] = 0
                    continue
                u_list = user_list['User_list'][user]
                combine = np.intersect1d(u_list,k_similar)
                p = 0
                for k_index in combine:
                    if self.dataset[user][k_index] > 0:
                        p += self.similarity_set[movie][k_index]*1

                predict_set[user][movie] = p
        self.predict_set = predict_set
        return self.predict_set

