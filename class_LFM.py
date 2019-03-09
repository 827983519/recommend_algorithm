class LFM:
    def __init__(self,classcount=10,itercount=30,alpha=0.01,lamda=0.06):
        self.classcount = classcount
        self.itercount = itercount
        self.alpha = alpha
        self.lamda = lamda 

    def initPara(self,data,userId,itemId):
        arrayp = np.random.rand(len(userId), self.classcount)/np.sqrt(self.classcount)
        arrayq = np.random.rand(self.classcount, len(itemId))/np.sqrt(self.classcount)
        self.p = pd.DataFrame(arrayp, columns=range(0,self.classcount), index=userId).to_dict()
        self.q = pd.DataFrame(arrayq, columns=itemId, index=range(0,self.classcount)).to_dict()
        self.u = data[self.rating_feature].mean()
        self.bi = dict(zip(itemId,[0 for i in range(len(itemId))]))
        self.bu = dict(zip(userId,[0 for i in range(len(userId))]))

        
    def lfmPredict(self,user,movie):
        p_q = sum([self.p[f][user] * self.q[movie][f] for f in range(self.classcount)])
        predict = p_q + self.u + self.bi[movie] + self.bu[user]
        return predict

    def fit(self,data,feature_name):
        self.user_feature = feature_name[0]
        self.movie_feature = feature_name[1]
        self.rating_feature = feature_name[2]
        item_based = data.pivot_table(index=self.movie_feature, columns=self.user_feature, values=self.rating_feature)
        self.user_mean = item_based.mean(axis=0).to_dict()
        self.movie_mean = item_based.mean(axis=1).to_dict()
        self.initPara(data,data[self.user_feature].unique(),data[self.movie_feature].unique())
        self.data = data[feature_name].to_dict()
        
        for step in range(self.itercount):
            print(step)
            for user,movie,rating in zip(self.data[self.user_feature].values(),self.data[self.movie_feature].values(),self.data[self.rating_feature].values()):
                eui = float(rating - self.lfmPredict(user,movie))
                self.bu[user] += self.alpha *(eui - self.bu[user])
                self.bi[movie] += self.alpha *(eui - self.bi[movie])
                p_now = [self.p[f][user] for f in range(self.classcount)]
                for i in range(self.classcount):
                    self.p[i][user] += self.alpha * (eui * self.q[movie][i] - self.lamda * self.p[i][user])
                    self.q[movie][i] += self.alpha * (eui * p_now[i] -self.lamda * self.q[movie][i])
            self.alpha *= 0.9

    def predict(self,test):
        predict_set = []
        test = test[[self.user_feature,self.movie_feature]]
        test = test.to_dict()
        for user,movie in zip(test[self.user_feature].values(),test[self.movie_feature].values()):
            if movie not in self.data[self.movie_feature].values():
                predict_set.append(self.user_mean[user])
                continue
            if user not in self.data[self.user_feature].values():
                predict_set.append(self.movie_mean[movie])
                continue
            predict_set.append(self.lfmPredict(user,movie))
        return predict_set
            
