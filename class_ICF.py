from sklearn.metrics import pairwise_distances

class ICF:
    def __init__(self,neighbor=10,similarity='pearson'):
        self.neighbor = neighbor
        self.similarity = similarity

    def calculate_similarity(self,data):
        if self.similarity == 'pearson':
            return  pd.DataFrame(1- pairwise_distances(data,metric='correlation'),columns=data.index,index=data.index).to_dict()

    def find_nearset_neighbor(self,movieId):
        top_neighbor = sorted(self.similarity_set[movieId].items(), key=lambda e:e[1], reverse=True)[1:1+self.neighbor]
        similar_index = [i[0] for i in top_neighbor]
        return similar_index

    def fit(self,data):
        
        self.movie_mean = data.mean(axis=1).to_dict()
        self.user_mean = data.mean(axis=0).to_dict()
        data = data.fillna(0)
        self.dataset = data.to_dict()
        self.similarity_set = self.calculate_similarity(data)
        

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


        
