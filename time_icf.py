import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


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
        rjaccard_table = pd.DataFrame(np.zeros((len(data.index),len(data.index)))+100,index=data.index,columns=data.index).to_dict()
        for i in rjaccard_table:
            for j in rjaccard_table:
                if i == j:
                    rjaccard_table[i][j]=1
                    rjaccard_table[j][i]=1
                    continue
                if rjaccard_table[i][j] != 100:
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

    def calculate_similarity(self,data,IIF_t):
        if self.similarity == 'pearson':
            return  IIF.to_dict()#pd.DataFrame(1- pairwise_distances(data.fillna(0),metric='correlation'),columns=data.index,index=data.index).to_dict()
        if self.similarity == 'IIF':
#             IIF = pd.read_csv('IIF_ta.csv',index_col=0)
#             IIF = pd.DataFrame(IIF.values,index=IIF.index,columns=IIF.index)
            return IIF_t.to_dict()
        
        if self.similarity == 'jaccard':
#             return self.calculate_jaccard(data)
            return IIF_t.to_dict()
        
        if self.similarity == 'rjaccard':
            return self.calculate_Rjaccard(data)

    def find_nearset_neighbor(self,movieId):
        top_neighbor = sorted(self.similarity_set[movieId].items(), key=lambda e:e[1], reverse=True)[1:1+self.neighbor]
        similar_index = [i[0] for i in top_neighbor]
        return similar_index

    def fit(self,data,IIF_t,time_table): 
        self.time_table = time_table
        self.origin_data = data.copy()
        self.dataset = data.fillna(0).to_dict()
        self.similarity_set = self.calculate_similarity(data,IIF_t)
        self.user_list = pd.DataFrame(index=item_based.columns,columns=['User_list']).to_dict()
        for i in data.columns:
            self.user_list['User_list'][i] = data[i].dropna().index.values
       
    
    def recommend(self,userId,topN):
        top = sorted(self.predict_set[userId].items(),key = lambda items:items[1],reverse=True)[:topN]
        top_N = [i[0] for i in top]
        return top_N
     
        

    def calculate_Fscore(self,test_data,topN):
        self.precision = []
        self.recall = []
        self.Fscore = []
        self.coverage = 0
        total_movie = set()
        test_data = test_data[['userId','movieId','rating']]
        for user in set(test_data.userId):
            #top = sorted(self.predict_set[user].items(),key = lambda items:items[1],reverse=True)[:topN]
            #top_N = [i[0] for i in top]
            if user not in self.predict_set:
                continue
            top_N = self.recommend(user,topN)
            total_movie.update(top_N)
            test_set = test_data[(test_data['userId']==user) & (test_data['rating']>=3)]['movieId'].values#test_data.loc[test_data['userId']==user,'movieId'].values
            if len(test_set)==0:
                continue
            inter = len(np.intersect1d(top_N,test_set))
            precision = inter/topN
            recall = inter/len(test_set)
            if recall == 0:
                fscore = 0
            else:
                fscore = (1+0.25)*(precision*recall)/(0.25*precision+recall)
            self.precision.append(precision)
            self.recall.append(recall)
            self.Fscore.append(fscore)
            self.coverage = len(total_movie)/650#len(self.similarity_set)
    
    
    def predict_whole(self): 
        user_list = self.user_list
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
                    if self.dataset[user][k_index] > 3:
                        p += self.similarity_set[movie][k_index]*1

                predict_set[user][movie] = p
        self.predict_set = predict_set
        return self.predict_set
    
    def predict_whole_time(self,current,b):
        user_list = self.user_list
        predict_set = copy.deepcopy(self.dataset)
        for movie in self.dataset[list(self.dataset.keys())[0]]:
            k_similar = self.find_nearset_neighbor(movie)
            for user in self.dataset.keys(): 
                if predict_set[user][movie] > 0:
                    predict_set[user][movie] = 0
                    continue
                u_list = user_list['User_list'][user]
                combine = np.intersect1d(u_list,k_similar)
                p = 0
                for k_index in combine:
                    if self.dataset[user][k_index] >= 3:
                        f = 1/(1 + b*abs(current - self.time_table[user][k_index]))
                        p += self.similarity_set[movie][k_index]*f
                predict_set[user][movie] = p
        self.predict_set = predict_set
        return self.predict_set


def IIF(I1,I2,data,time_table,j,k,alpha):
    intersect = np.intersect1d(I1,I2)
    if len(intersect) == 0:
        return 0
    Nu = len(I1)
    Nv = len(I2)
    up = 0
    for i in intersect:
        f = 1/(1 + alpha*abs(time_table[i][k] -time_table[i][j]))
        up += 1/(np.log(1+1.0*(len(data['User_list'][i])))) * f
    return up/np.sqrt(1.0*Nu*Nv)

def IIF_table(movie_list,user_list,time_table,alpha):
    data = user_list.to_dict()
    a = dict(zip(list(movie_list['Movie_list'].keys()),[1000 for i in range(len(movie_list['Movie_list']))]))
    table = dict(zip(list(movie_list['Movie_list'].keys()),[copy.deepcopy(a) for i in range(len(movie_list['Movie_list']))]))
    for i in table:
#         print(i)
        for j in table:
            if i == j:
                table[j][i] = 1
                continue
            if table[i][j] != 1000:
                table[j][i] = table[i][j]   
                continue
            table[j][i] = IIF(movie_list['Movie_list'][j],movie_list['Movie_list'][i],data,time_table,j,i,alpha)
    return pd.DataFrame(table,index=list(table.keys()),columns=list(table.keys()))

def jaccard(n1,n2,time_table,j,k,alpha):
    inter = np.intersect1d(n1,n2)
    if len(inter) == 0:
        return 0
    up = 0
    for i in inter:
        up+= 1*1/(1 + alpha*abs(time_table[i][k] - time_table[i][j]))
    down = len(np.union1d(n1,n2))
    return up/down

def calculate_jaccard(movie_list,user_list,time_table,alpha):
    a = dict(zip(list(movie_list['Movie_list'].keys()),[1000 for i in range(len(movie_list['Movie_list']))]))
    table = dict(zip(list(movie_list['Movie_list'].keys()),[copy.deepcopy(a) for i in range(len(movie_list['Movie_list']))]))
    for i in table:
#         print(i)
        for j in table:
            if i == j:
                table[j][i] = 1
                continue
            if table[i][j] != 1000:
                table[j][i] = table[i][j]   
                continue
            table[j][i] = jaccard(movie_list['Movie_list'][j],movie_list['Movie_list'][i],time_table,j,i,alpha)
    return pd.DataFrame(table,index=list(table.keys()),columns=list(table.keys()))

rating = pd.read_csv('./ml-1m/ratings.dat',sep='::',header=None,names=['userId','movieId','rating','timestamp'])
movie = pd.read_csv('./ml-1m/movies.dat',sep='::',header=None,names=['movieId','title','genres'])
rating = rating[(rating['userId']<3500) & (rating['movieId'] < 700)]
movie = movie[movie['movieId']<700]


from sklearn.model_selection import train_test_split
train,test = train_test_split(rating,test_size = 0.2,random_state=0)
item_based = train.pivot_table(index='movieId', columns='userId', values='rating')
# rating = rating.sort_values('timestamp')
time_table = train.pivot_table(index='movieId', columns='userId', values='timestamp')



movie_list = pd.DataFrame(index=item_based.index,columns=['Movie_list'])
for i in item_based.index.values:
    print(i)
    movie_list['Movie_list'][i] = item_based.loc[i].dropna().index.values
    
user_list = pd.DataFrame(index=item_based.columns,columns=['User_list'])
for i in item_based.columns:
    user_list['User_list'][i] = item_based[i].dropna().index.values



IIF_t = IIF_table(movie_list,user_list,time_table,0.001)
jaccard_t = calculate_jaccard(movie_list,user_list,time_table,0.001)






