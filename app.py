from flask import Flask , jsonify,request,render_template
from flask_cors import CORS , cross_origin
import pickle
import os
import numpy


os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app=Flask(__name__)
CORS(app)

@app.route('/',methods=['GET'])
@cross_origin()
def Homepage():
    return render_template('index.html')
@app.route('/predict',methods=['POST','GET'])
@cross_origin()
def index():
    if request.method=='POST':
        try:
            data=request.form['data']

            data1=[data]
            cv=pickle.load(open('cv_instance.pickle','rb'))
            vect=cv.transform(data1).toarray()
            print(vect)
            filename="Spam_finalized_model.pickle"
            loaded_model=pickle.load(open(filename,'rb'))
            prediction= loaded_model.predict(vect)
            print(prediction)
            if prediction[0]=="ham":
                prediction="This is not a Spam Email."
            elif prediction[0]=="spam":
                prediction="This is a Spam Email."
            return render_template('results.html',prediction=prediction)
        except Exception as e:
            print('The Exception message is: ', e)
            return 'something is wrong'
    else:
        return render_template('index.html')
port = os.getenv("PORT")
if __name__ == '__main__':
	if port is None:
		app.run(host='0.0.0.0', port=5000, debug=True)
	else:
		app.run(host='0.0.0.0', port=int(port), debug=True)


