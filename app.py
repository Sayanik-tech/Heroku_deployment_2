from flask import Flask,render_template,request
import joblib

#initialize the app
app = Flask(__name__)

# Load the model
model = joblib.load('DT_87.pkl')

@app.route('/')
def hello():
    return render_template('form.html')


@app.route('/submit',methods = ["POST"])
def form_data():
    
    Age = request.form.get('Age')
    Estimated_Salary = request.form.get('Salary')
    

    y_pred = model.predict([[Age,Estimated_Salary]])
    
    print(y_pred)

    if y_pred[0] == 1:
        out = 'Purchased'
    else:
        out = 'Not Purchased'


    return render_template('index.html', data = f'{out}')


if __name__ == '__main__':
    app.run(debug = True)


