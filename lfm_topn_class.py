class LFM_topN:
    def __init__(self,classcount=10,itercount=30,alpha=0.01,lamda=0.03,ratio=1):
        self.classcount = classcount
        self.itercount = itercount
        self.alpha = alpha
        self.lamda = lamda 
        self.ratio = ratio

    def getUserNegativeItem(self,userId,positiveItemList):
        ret = dict()
        for i in positiveItemList['User_list'][userId]:
            ret[i] = 1
        n = 0
        for i in range(0, len(positiveItemList['User_list'][userId]) * 10):
            item = self.items_pool[np.random.randint(0, len(self.items_pool) - 1)]
            if item in ret:
                 continue
            ret[item] = 0
            n += 1
            if n > self.ratio*len(positiveItemList['User_list'][userId]):
                break
        negativeItemList = []
        for i in ret:
            if ret[i] == 0:
                negativeItemList.append(i)
        return negativeItemList


    def get_user_item(self,data):
        item_based = data.pivot_table(index=self.movie_feature, columns=self.user_feature, values=self.rating_feature)
        positiveItemList = pd.DataFrame(index=item_based.columns,columns=['User_list'])
        for i in item_based.columns:
            positiveItemList['User_list'][i] = item_based[i].dropna().index.values
        positiveItemList = positiveItemList.to_dict()       
        user_item = {}
        for user_Id in item_based.columns.values:
            negativeItemList = self.getUserNegativeItem(user_Id, positiveItemList)
            user_item[user_Id] = [positiveItemList['User_list'][user_Id],negativeItemList]
        return user_item

    def initPara(self,data,userId,itemId):
        self.p = np.random.rand(len(userId), self.classcount)/np.sqrt(self.classcount)
        self.q = np.random.rand(len(itemId), self.classcount)/np.sqrt(self.classcount)
        self.user_list = {}
        self.movie_list = {}
        
        j = 0
        for i in userId:
            self.user_list[i] = j
            j+=1
            
        j = 0
       # print(176 in itemId)
        for i in itemId:
            self.movie_list[i] = j
            j+=1

        
    def lfmPredict(self,user,movie):
        p_q = np.dot(self.p[self.user_list[user]],self.q[self.movie_list[movie]])
        predict = self.sigmoid(p_q)
        return predict
    
    def sigmoid(self,x):
        return 1.0/(1+np.exp(-x))

    def fit(self,data,feature_name):
        self.user_feature = feature_name[0]
        self.movie_feature = feature_name[1]
        self.rating_feature = feature_name[2]
        movie_list = list(set(data[self.movie_feature].values))
        user_list = list(set(data[self.user_feature].values))
        self.initPara(data,user_list,movie_list)
        self.items_pool = list(data.movieId.values)
        self.user_item = self.get_user_item(data)

        for step in range(self.itercount):
            print(step)
            for user in self.user_item:   
                for movie in self.user_item[user][0]:
                    eui = 1 - self.lfmPredict(user,movie)
                    du = self.alpha * (eui * self.q[self.movie_list[movie]] - self.lamda * self.p[self.user_list[user]])
                    di = self.alpha * (eui * self.p[self.user_list[user]] - self.lamda * self.q[self.movie_list[movie]])
                    self.p[self.user_list[user]] += du
                    self.q[self.movie_list[movie]] += di
                
                for movie in self.user_item[user][1]:
                    eui = 0 - self.lfmPredict(user,movie)
                    du = self.alpha * (eui * self.q[self.movie_list[movie]] - self.lamda * self.p[self.user_list[user]])
                    di = self.alpha * (eui * self.p[self.user_list[user]] - self.lamda * self.q[self.movie_list[movie]])
                    self.p[self.user_list[user]] += du
                    self.q[self.movie_list[movie]] += di
            self.alpha *= 0.9
        
    def recommend(self,userID,TopN=10):
        predict_movie = {}
        for movie in self.movie_list:
            predict_movie[movie] = self.lfmPredict(userID,movie)
        sort_predict_movie = sorted(predict_movie.items(), key = lambda item:item[1],reverse=True)[:150]
        predict_list = []
        for index in sort_predict_movie:
            if index[0] not in self.user_item[userID][0]:
                predict_list.append(index[0])
                if len(predict_list) == TopN:
                    break

        return predict_list
    
    def calculate_Fscore(self,test_data,topN):
        self.precision = []
        self.recall = []
        self.Fscore = []
        test_data = test_data[['userId','movieId']]
        for user in set(test_data.userId):
            if user not in self.user_item:
                continue
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
