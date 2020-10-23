import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, app,render_template
from flask import Response
from flask_cors import CORS
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
CORS(app)
app.config['DEBUG'] = True

#scalar=StandardScaler()
with open("modelForPrediction.sav", 'rb') as f:
            model = pickle.load(f)
with open("standardScalar.sav", 'rb') as s:
    scalar = pickle.load(s)


@app.route("/", methods=['GET'])
def homepage():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict():
    try:
        if request.method=='POST':
            preg = int(request.form['Pregnancies'])
            glucose = int(request.form['Glucose'])
            bp = int(request.form['Blood Pressure'])
            st = int(request.form['Skin Thickness'])
            insulin = int(request.form['Insulin'])
            bmi = float(request.form['BMI'])
            dpf = float(request.form['DPF'])
            age = int(request.form['Age'])

            data = np.array([[preg,glucose,bp,st,insulin,bmi,dpf,age]])
            data_df = pd.DataFrame(data,index=[1,])
            scaled_data = scalar.fit_transform(data_df)

            predict = model.predict(scaled_data)

            print("prediction is :",predict[0])
            print(type(predict[0]))
            return render_template('results.html',predict = predict[0])

    except ValueError:
        return Response("Value not found")
    
    except Exception as e:
        print('exception is   ',e)
        return Response(e)

if __name__ == "__main__":
           # to run locally
    host = '0.0.0.0'
    port = 5000
    app.run(debug=True)
#to run on cloud
	#app.run(debug=True) # running the app
