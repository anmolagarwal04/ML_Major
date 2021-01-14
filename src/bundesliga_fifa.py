import pandas as pd
import numpy as np
import os
import nltk
from fuzzywuzzy import process
import math
from sklearn.preprocessing import MinMaxScaler
import argparse
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from xgboost import XGBClassifier
from sklearn.preprocessing import OneHotEncoder
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.metrics import classification_report
from keras.layers import LeakyReLU
from keras.layers import LSTM
import warnings
warnings.filterwarnings('ignore')


parser = argparse.ArgumentParser(description='List the content of a folder')
parser.add_argument('--model', default=None, type=str, help='lr, svm, xgb, lstm, rf, stacking')
args = parser.parse_args()


#########################################
#			PREPROCESS DATA    			          #
#########################################

df = pd.read_csv(os.path.join('./data','Bundesliga.csv'),index_col=0)

df.year = df.year.astype('str')
home_cols = ['home_Possession %',
       'home_Total Shots', 'home_On Target', 'home_Off Target', 'home_Blocked',
       'home_Passing %', 'home_Clear-Cut Chances', 'home_Corners',
       'home_Offsides', 'home_Tackles %', 'home_Aerial Duels %', 'home_Saves',
       'home_Fouls Committed', 'home_Fouls Won', 'home_Yellow Cards',
       'home_Red Cards']
away_cols = ['away_Possession %',
       'away_Total Shots', 'away_On Target', 'away_Off Target', 'away_Blocked',
       'away_Passing %', 'away_Clear-Cut Chances', 'away_Corners',
       'away_Offsides', 'away_Tackles %', 'away_Aerial Duels %', 'away_Saves',
       'away_Fouls Committed', 'away_Fouls Won', 'away_Yellow Cards',
       'away_Red Cards']
home_form = ['home_form_1', 'home_form_2', 'home_form_3',
       'home_form_4', 'home_form_5', 'home_form_6']

away_form = ['away_form_1',
       'away_form_2', 'away_form_3', 'away_form_4', 'away_form_5',
       'away_form_6']

df['home_goal_difference'] = df.home_goals - df.away_goals
df['away_goal_difference'] = df.away_goals - df.home_goals

teams = df.home_team.unique()

home_cols.append('home_goals')
home_cols.append('home_goal_difference')
away_cols.append('away_goals')
away_cols.append('away_goal_difference')

for team in teams:
  idxs = df[df.home_team==team].index
  for col in home_cols:
    df.loc[idxs,col+'_moving'] = df.loc[idxs,col].rolling(window=5).mean().shift()
  idxs = df[df.away_team==team].index
  for col in away_cols:
    df.loc[idxs,col+'_moving'] = df.loc[idxs,col].rolling(window=5).mean().shift()

home_cols.remove('home_goals')
home_cols.remove('home_goal_difference')
away_cols.remove('away_goals')
away_cols.remove('away_goal_difference')

data = df.copy()
df_new = df.copy()
df_new = df_new[df_new.year>'2014-15']

df_new = data.copy()
df_new = df_new[df_new.year>'2014-15']

years = ['2015-16', '2016-17', '2017-18', '2018-19', '2019-20']
for year in years:
  df = data[data.year<year]
  home_dfs = {}
  away_dfs = {}
  for team in teams:
    avg_home_stats = df[df.home_team==team][home_cols].mean()
    avg_away_stats = df[df.away_team==team][away_cols].mean()

    if team not in df.home_team.unique() and team in df_new[df_new.year==year]['home_team'].unique():
      idxs = df_new[(df_new.home_team==team) & (df_new.year==year)].index
      avg_home_stats = df[home_cols].mean()

    if team not in df.away_team.unique() and team in df_new[df_new.year==year]['away_team'].unique():
      idxs = df_new[(df_new.away_team==team) & (df_new.year==year)].index
      avg_away_stats = df[away_cols].mean()

    
    
    home_dfs[team] = 0.8*avg_home_stats.values + 0.2*avg_away_stats.values
    away_dfs[team] = 0.8*avg_away_stats.values + 0.2*avg_home_stats.values

    idxs = df_new[(df_new.home_team==team) & (df_new.year==year)].index
    df_new.loc[idxs,home_cols] = home_dfs[team]

    idxs = df_new[(df_new.away_team==team) & (df_new.year==year)].index
    df_new.loc[idxs,away_cols] = away_dfs[team]


player_columns = ['home_player_1', 'home_player_2', 'home_player_3',
       'home_player_4', 'home_player_5', 'home_player_6', 'home_player_7',
       'home_player_8', 'home_player_9', 'home_player_10', 'home_player_11',
       'away_player_1', 'away_player_2', 'away_player_3', 'away_player_4',
       'away_player_5', 'away_player_6', 'away_player_7', 'away_player_8',
       'away_player_9', 'away_player_10', 'away_player_11']


df = df_new.copy()
statsFor = {}

for i in range(15,20,1):
  fromYear = str(i)
  toYear = str(i + 1)
  statsFor[fromYear] = df[df['year'] == '20{}-{}'.format(fromYear, toYear)]
  fifaPlayerRatings = pd.read_csv(os.path.join('./data','players_{}.csv'.format(toYear)))

  bundesligaPlayersFifaRating = fifaPlayerRatings[(fifaPlayerRatings['league_name'] == 'German 2. Bundesliga') | (fifaPlayerRatings['league_name'] == 'German 1. Bundesliga')] 
  teamsPresentInFifaRatings = bundesligaPlayersFifaRating['club_name'].unique()

  skySportsDataTeams = statsFor[fromYear]['home_team'].unique()

  skySportsToFifa = {}
  fifaToSkySports = {}
  for team in skySportsDataTeams:
    b = process.extractOne(team, teamsPresentInFifaRatings)
    if b[1] > 60:
      skySportsToFifa[team] = b[1]
      fifaToSkySports[b[0]] = team
  
  skySportsToFifa['Cologne'] = '1. FC Köln'
  fifaToSkySports['1. FC Köln'] = 'Cologne'

  a = [i for i in skySportsDataTeams if i not in skySportsToFifa.keys()]
  b = [i for i in teamsPresentInFifaRatings if i not in fifaToSkySports.keys()]


  teamWisePlayerStats = {}
  for f, s in fifaToSkySports.items():
      teamWisePlayerStats[s] = bundesligaPlayersFifaRating[bundesligaPlayersFifaRating['club_name'] == f]

  playerColsHome = [i for i in df.columns if 'home_player' in i]
  playerColsAway = [i for i in df.columns if 'away_player' in i]

  unqiuePlayerseInTeam = {}
  for t in skySportsDataTeams:
    unqiuePlayerseInTeam[t] = pd.Series(pd.unique(statsFor[fromYear][(statsFor[fromYear]['home_team'] == t)][playerColsHome].values.ravel('K')))
    a = len(unqiuePlayerseInTeam[t])
    unqiuePlayerseInTeam[t] = unqiuePlayerseInTeam[t].append(pd.Series(pd.unique(statsFor[fromYear][(statsFor[fromYear]['away_team'] == t)][playerColsAway].values.ravel('K'))))
    b = len(unqiuePlayerseInTeam[t])
    unqiuePlayerseInTeam[t] = unqiuePlayerseInTeam[t].unique()
    c = len(unqiuePlayerseInTeam[t])

  fifaShortNames = {}
  for t in skySportsDataTeams:
    fifaShortNames[t] = list(teamWisePlayerStats[t]['short_name'])

  skysportsPlayerRating = {}
  for t, lop in unqiuePlayerseInTeam.items():
    skysportsPlayerRating[t] = {}
    for player in lop:
      if str(player) == 'nan':
        skysportsPlayerRating[t][player] = 0
        continue
    
      try:
        selectedInFifa = process.extractOne(player, fifaShortNames[t])
      except:
        print('~'*10)
        print('Error raised by this player')
        print(player)
        print(type(player))
        print(str(player))
        print('~'*10)
        break
      # print(player, selectedInFifa)
      skysportsPlayerRating[t][player] = ((teamWisePlayerStats[t][teamWisePlayerStats[t]['short_name'] == selectedInFifa[0]]).iloc[0]).loc['overall']
    # print()

  homePlayerColumns = [i for i in df.columns if 'home_player' in i]

  allPlayerRatings = {}
  for k1, v1 in skysportsPlayerRating.items():
      allPlayerRatings = {**allPlayerRatings, **v1}

  statsFor[fromYear].replace(to_replace=allPlayerRatings, inplace=True)

df_new = pd.DataFrame()
for i in range(15,20):
  df_new = df_new.append(pd.DataFrame(statsFor[str(i)]))

df_train = df_new.dropna().copy()
df_train = df_train.drop(columns=home_form,axis=1)
df_train = df_train.drop(columns=away_form,axis=1)

team_names = list(set(df_train['home_team'].tolist()  + df_train['away_team'].tolist()))
df_test_teams = df_train[df_train.year=='2019-20'].loc[:,['home_team','away_team']].copy()


def standings(df_test_teams,y_pred):
  result_dict = {}
  temp = pd.concat([df_test_teams.reset_index(),pd.Series(y_pred)],axis=1,ignore_index=True)
  temp.columns = ['ind','home','away','res']
  for team in teams:
    sum = (3* len(temp[temp.home==team][temp.res==1]))
    sum += len(temp[temp.home==team][temp.res==0])
    sum += len(temp[temp.away==team][temp.res==0])
    sum += (3*len(temp[temp.away==team][temp.res==2]))
    result_dict[team] = sum
  
  return sorted(result_dict.items(), key=lambda item: item[1],reverse=True)


team_indices = {}

for i,team in enumerate(team_names):
  team_indices[team] = i


df_train['home_team'] = df_train['home_team'].replace(team_indices)
df_train['away_team'] = df_train['away_team'].replace(team_indices)

df_train.reset_index(inplace=True)
df_train.drop('index',axis=1,inplace=True)

df_test = df_train[df_train['year'] == '2019-20']
df_train = df_train[df_train['year']<'2019-20']

def f(row):
    if row['home_goals'] == row['away_goals']:
        val = 0
    elif row['home_goals'] > row['away_goals']:
        val = 1
    else:
        val = 2
    return val


y_train = df_train.apply(f, axis=1)
y_test = df_test.apply(f,axis=1)
df_train = df_train.drop(['year'],axis=1)
df_test = df_test.drop(['year'],axis=1)

df_train = df_train.drop(['home_goals','away_goals','home_goal_difference','away_goal_difference'],axis=1)
df_test = df_test.drop(['home_goals','away_goals','home_goal_difference','away_goal_difference'],axis=1)


train_home = df_train.home_team
train_away = df_train.away_team

test_away = df_test.away_team
test_home = df_test.home_team

df_train.drop(['home_team','away_team'],axis=1,inplace=True)
df_test.drop(['home_team','away_team'],axis=1,inplace=True)



#########################################
#	  	   GET TRAIN AND TEST DATA    		#
#########################################

scaler = MinMaxScaler()
X_train = scaler.fit_transform(df_train)
X_test = scaler.fit_transform(df_test)


X_train = np.append(X_train,np.array(train_home).reshape(-1,1),axis=1)
X_train = np.append(X_train,np.array(train_away).reshape(-1,1),axis=1)

X_test = np.append(X_test,np.array(test_home).reshape(-1,1),axis=1)
X_test = np.append(X_test,np.array(test_away).reshape(-1,1),axis=1)



#########################################
#	  		        TRAIN ON MODEL    			#
#########################################


if args.model=='rf':

  # from sklearn.metrics import accuracy_score
  # from sklearn.model_selection import GridSearchCV

  # from sklearn.model_selection import RandomizedSearchCV# Number of trees in random forest
  # n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 5)]
  # # Number of features to consider at every split
  # max_features = ['auto', 'sqrt']
  # # Maximum number of levels in tree
  # max_depth = [int(x) for x in np.linspace(10, 110, num = 5)]
  # max_depth.append(None)
  # # Minimum number of samples required to split a node
  # min_samples_split = [2, 4, 10]
  # # Minimum number of samples required at each leaf node
  # min_samples_leaf = [1, 2, 10]
  # # Method of selecting samples for training each tree
  # bootstrap = [True, False]# Create the random grid
  # random_grid = {'n_estimators': n_estimators,
  #                'max_features': max_features,
  #                'max_depth': max_depth,
  #                'min_samples_split': min_samples_split,
  #                'min_samples_leaf': min_samples_leaf,
  #                'bootstrap': bootstrap           }


  # rf = RandomForestClassifier(random_state=42,class_weight='balanced',n_estimators=300,criterion="gini")

  # clf = GridSearchCV(rf, random_grid,n_jobs=-1,verbose=100000)
  # clf.fit(X_train, y_train)
  # y_pred = clf.predict(X_test)

  # accuracy_score(y_pred,y_test)

  # print(clf.best_params_)

  clf = RandomForestClassifier(bootstrap=True,max_depth=12,max_features='auto',min_samples_leaf=1,min_samples_split=8,n_estimators = 200,random_state=42)
  clf.fit(X_train, y_train)
  y_pred = clf.predict(X_test)
  print("Test Accuracy = ", accuracy_score(y_pred,y_test))
  print("Predicted Standings")
  print(standings(df_test_teams,y_pred))
elif args.model=='lr':
  clf = LogisticRegression(random_state=42,max_iter=1000,class_weight='balanced', penalty="l2")
  clf.fit(X_train, y_train)
  y_pred = clf.predict(X_test)
  print("Test Accuracy = ", accuracy_score(y_pred,y_test))
  print("Predicted Standings")
  print(standings(df_test_teams,y_pred))
elif args.model=='svm':
  clf = svm.SVC()
  clf.fit(X_train, y_train)
  y_pred = clf.predict(X_test)
  print("Test Accuracy = ", accuracy_score(y_pred,y_test))
  print("Predicted Standings")
  print(standings(df_test_teams,y_pred))
elif args.model=='xgb':
  clf = XGBClassifier(n_jobs=-1,n_estimators=600,max_depth=8, learning_rate=0.1, booster="gbtree", objective='multi:softmax', num_class=3)
  clf.fit(X_train, y_train)
  y_pred = clf.predict(X_test)
  print("Test Accuracy = ", accuracy_score(y_pred,y_test))
  print("Predicted Standings")
  print(standings(df_test_teams,y_pred))

elif args.model=='lstm':
	ohe = OneHotEncoder()
	y_train = ohe.fit_transform(np.array(y_train).reshape(-1,1)).toarray()
	y_test = ohe.fit_transform(np.array(y_test).reshape(-1,1)).toarray()
	X_train.reshape(-1,1,X_train.shape[1]).shape
	X_train = X_train.reshape(-1,1,X_train.shape[1])
	X_test = X_test.reshape(-1,1,X_test.shape[1])
	y_train = y_train.reshape(-1,1,y_train.shape[1])
	y_test = y_test.reshape(-1,1,y_test.shape[1])

	model = Sequential()
	model.add(LSTM(100,input_shape=(1,92),return_sequences=True,activation='relu' ))
	model.add(Dropout(0.5))
	model.add(LSTM(100,return_sequences=True,activation='relu' ))
	model.add(Dropout(0.4))
	model.add(LSTM(100,return_sequences=True,activation='relu' ))
	model.add(Dropout(0.3))
	model.add(Dense(64,activation='relu'))
	model.add(Dropout(0.2))
	model.add(Dense(64,activation='relu'))
	model.add(Dropout(0.2))
	model.add(Dense(3,activation='softmax'))

	model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

	model.fit(X_train,y_train,epochs=25,batch_size=32)

	y_pred = model.predict(X_test)
	y_pred = model.predict(X_test)
	pred = list()
	for i in range(len(y_pred)):
	    pred.append(np.argmax(y_pred[i]))
	test = list()
	for i in range(len(y_test)):
	    test.append(np.argmax(y_test[i]))

	print("Test Accuracy = ", accuracy_score(pred,test))
	print("Predicted Standings")
	print(standings(df_test_teams,pred))

elif args.model=='stacking':
  # '''''''STACKING CLASSIFIER''''''''
  from sklearn.model_selection import cross_val_score
  from sklearn.model_selection import RepeatedStratifiedKFold
  from sklearn.linear_model import LogisticRegression
  from sklearn.neighbors import KNeighborsClassifier
  from sklearn.tree import DecisionTreeClassifier
  from sklearn.svm import SVC
  from sklearn.naive_bayes import GaussianNB
  from matplotlib import pyplot
  from sklearn.ensemble import StackingClassifier
  from sklearn.ensemble import RandomForestClassifier
  from sklearn.ensemble import AdaBoostClassifier
  from xgboost import XGBClassifier

  level0 = [
            ('lr',LogisticRegression(C=1,max_iter=10000,random_state=42,n_jobs=-1,solver='saga',penalty='l1')),
            # ('dt',DecisionTreeClassifier(random_state=0)),
            # ('knn',KNeighborsClassifier()),
            ('rf',RandomForestClassifier(n_estimators=150,random_state=42,n_jobs=-1,criterion='gini',max_depth=8,min_samples_split=8)),
            ('abc',AdaBoostClassifier(n_estimators=150,random_state=42,learning_rate=0.1)),
            ('svm',SVC(random_state=0)),
            ('xgb',XGBClassifier(n_estimators=200,n_jobs=-1,booster='gbtree',learning_rate=0.01,random_state=42,max_depth=3))]
  # level1 = XGBClassifier(n_estimators=100,n_jobs=-1,booster='gbtree',learning_rate=0.01,random_state=42,max_depth=3)
  # level1 = LogisticRegression(C=1,max_iter=10000,class_weight='balanced',random_state=0,solver='saga')
  level1 = RandomForestClassifier(n_estimators=150,random_state=42,n_jobs=-1,max_depth=8,criterion='entropy',min_samples_split=8)
  model = StackingClassifier(estimators=level0, final_estimator=level1, cv=10,n_jobs=-1)
  model.fit(X_train,y_train)
  y_pred = model.predict(X_test)
  print("Test Accuracy = ", accuracy_score(y_pred,y_test))
  print("Predicted Standings")
  print(standings(df_test_teams,y_pred))





