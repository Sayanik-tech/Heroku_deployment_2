import joblib

model = joblib.load('DT_87.pkl')

y_pred = model.predict([[55,25000]])

if y_pred[0] == 1:
    print('Purchased')
else:
    print('Not Purchased')