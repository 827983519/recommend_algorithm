from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

def Genre_table(movie):
    genre_set = set()
    for i in movie.genres.values:
        genre_set.update(i.split('|'))
 #   genre_set.remove('(no genres listed)')
    Genre_matrix = pd.DataFrame(np.zeros([len(genre_set),len(genre_set)]),index=genre_set,columns=genre_set)
    frequency_list = {}
    
    for index in Genre_matrix.index:
        frequency_list[index] = 0 
        
    for i in movie.genres:
        if i == '(no genres listed)':
            continue
        genre_list = i.split('|')
        for j in range(len(genre_list)):
            criterion = genre_list[j]
            frequency_list[criterion] += 1 
            for k in genre_list[j+1:]:
                Genre_matrix.loc[criterion][k] += 1
    for i in frequency_list.keys():
        Genre_matrix.loc[i] = Genre_matrix.loc[i]/frequency_list[i]
    return Genre_matrix


rating = pd.read_csv('./ml-1m/ratings.dat',sep='::',header=None,names=['userId','movieId','rating','timestamp'])
movie = pd.read_csv('./ml-1m/movies.dat',sep='::',header=None,names=['movieId','title','genres'])

rating = rating[(rating['userId']<3500) & (rating['movieId'] < 700)]
movie = movie[movie['movieId']<700]
train,test = train_test_split(rating,test_size = 0.2,random_state=0)

Genre_matrix = Genre_table(movie)

#Conpute user preference
user_based = train.pivot_table(index='userId', columns='movieId', values='rating')
user_pre_table = pd.DataFrame(index=user_based.index,columns=['preference_genre'])
user_pre = {}
for i in user_based.index:
    user_pre[i] = [(user_based.loc[i][user_based.loc[i].notnull()]).index.values]

genre_number = 3

for userid in user_pre.keys():
    user_movie_list = {}
    for index in Genre_matrix.index:
        user_movie_list[index] = 0 
    for movieid in user_pre[userid][0]:
        genre_list = movie[movie['movieId']==movieid].genres.values[0]
        if genre_list == '(no genres listed)':
            continue
        genre_list = genre_list.split('|')
        for genre in genre_list:
            user_movie_list[genre] += 1
    user_movie_list = sorted(user_movie_list.items(),key=lambda item:item[1],reverse=True)
    user_genre_list = []
    for i in range(genre_number):
        user_genre_list.append(user_movie_list[i][0])
    user_pre_table.loc[userid]['preference_genre']=user_genre_list

#Compute recommend point
movie_mean = user_based.mean(axis=0)
recommend_dict = pd.DataFrame(index=user_based.index,columns=user_based.columns).to_dict()
movie_dict = dict(zip(movie.movieId.values,movie.genres.values))
user_pre_dict = dict(zip(user_pre_table.index.values,user_pre_table.preference_genre.values))
Genre_dict = Genre_matrix.to_dict()

for movieId in recommend_dict:
    print(movieId)
    movie_genres = movie_dict[movieId]
    if movie_genres == '(no genres listed)':
        continue
    movie_genres = movie_genres.split('|')
    for user in recommend_dict[movieId]: 
        preference_genre = user_pre_dict[user]
        up = 0
        for genres in preference_genre:
            if genres in movie_genres:
                for movie_gen in movie_genres:
                    if genres != movie_gen:
                        up += (Genre_dict[movie_gen][genres])/(len(movie_genres)-1)
                    else:
                        up += 1
            else:
                for movie_gen in movie_genres:
                    up += (Genre_dict[movie_gen][genres])/(len(movie_genres))
        recommend_dict[movieId][user] = up * movie_mean[movieId]/len(preference_genre)

#Recommend to user I
filt_index =recommend_point.loc[I].sort_values(ascending=False)[:100].index
recommend_index = movie[movie['movieId'].isin(filt_index)].rating_people.sort_values(ascending=False).index[:10]
movie[movie.index.isin(recommend_index)]
