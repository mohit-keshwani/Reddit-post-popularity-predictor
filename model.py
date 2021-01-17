
import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

data = pd.read_csv('processed-data.csv')

#feature selection after getting important features, look for modeling notebook in the repo 
data = data.iloc[:, lambda data: data.columns.str.contains('comment_root_score|parent_cosine|time_since_comment_root|time_since_parent|parent_score|parent_euc|score', case= False)]
data.fillna(0 , inplace = True)


X_train, X_test, y_train, y_test = train_test_split(data.drop(['score','score_hidden'] , axis =1), 
                                                    data.score, 
                                                    test_size = 0.2, 
                                                    random_state = 20)
model = RandomForestRegressor(n_estimators = 70 ,min_samples_leaf = 10 ,max_depth = 6 , n_jobs=-1).fit(X_train,y_train)
print(X_train.columns)

pickle.dump(model, open('model.pkl', 'wb'))