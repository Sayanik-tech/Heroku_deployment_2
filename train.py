import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.tree import DecisionTreeClassifier
import joblib
url = 'https://raw.githubusercontent.com/Sayanik-tech/Classification/main/Social_Network_Ads.csv'

df = pd.read_csv(url)
print(df)

X = df.iloc[:,2:4]
y = df.iloc[:,-1]

x_train,x_test,y_yrain,y_test = model_selection.train_test_split(X,y,test_size = 0.2,random_state = 101)

model = DecisionTreeClassifier(criterion='entropy',random_state=101)

model.fit(x_train,y_yrain)
y_pred = model.predict(x_test)
# Accuracy
from sklearn.metrics import accuracy_score
result = accuracy_score(y_test,y_pred)
print(result)



#save the model (.sav/.pkl)
joblib.dump(model, 'DT_87.pkl')



if y_pred[0] == 1:
    print('Purchased')
else:
    print('Not Purchased')
