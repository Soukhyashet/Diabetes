from flask import Flask,render_template,request,session
import pandas as pd

from sklearn.metrics import accuracy_score
import joblib

app = Flask(__name__)



@app.route('/')
def index():  # put application's code herez
    return render_template('index.html')

@app.route('/prediction')
def predict():
    return render_template('predict.html')

@app.route('/prediction1',methods =["POST","GET"])
def pred():
    s = []
    if request.method== "POST":
        pregnancies = request.form['pr']
        glucose = request.form['gl']
        BP = request.form['bp']
        skinthickness = request.form['skin']
        insulin = request.form['insulin']
        BMI = request.form['bmi']
        DPF = request.form['dpf']
        age = request.form['age']
        s.extend([pregnancies,glucose,BP,skinthickness,insulin,BMI,DPF,age])
        model = joblib.load('svcmodel.pkl')
        y_pred = model.predict([s])
        return render_template('predict.html',msg="success",op=y_pred)


if __name__ == '__main__':
    app.secret_key="hai"
    app.run(debug=True)
