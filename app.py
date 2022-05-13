from flask import Flask, request, render_template
from flask import jsonify
import numpy as np
import pickle

app = Flask(__name__, template_folder = 'template', static_url_path='/static');

app.config['DEBUG'] = True

model = pickle.load(open('model_1.pkl', 'rb')) # for pass - fail
#dt_model = pickle.load(open('dt_model.pkl','rb'))
KNNmodel = pickle.load(open('KNNmodel.pkl','rb')) # for Grades

@app.route("/")
def home():
	#return "Hello world";
	return render_template('Project.html')

@app.route('/predict', methods = ['POST'])
def predict():
	# forming our attributes values
	sex = int(request.form['sex'])
	age = int(request.form['age'])
	address = int(request.form['address'])
	famsize = int(request.form['famsize'])
	Pstatus = int(request.form['pstatus'])
	Medu = int(request.form['medu'])
	Fedu = int(request.form['fedu'])
	Mjob = int(request.form['mjob'])
	Fjob = int(request.form['fjob'])
	reason = int(request.form['reason'])
	guardian = int(request.form['guardian'])
	traveltime = int(request.form['traveltime'])
	studytime = int(request.form['studytime'])
	fail = int(request.form['failures'])
	if(fail > 3):
		failures = 4
	else:
		failures = fail

	schoolsup = int(request.form['edusupport'])
	famsup = int(request.form['parentsupport'])
	paid = int(request.form['paid'])
	activities = int(request.form['extra'])
	nursery = int(request.form['nursery'])
	higher = int(request.form['higher'])
	internet = int(request.form['internet'])
	romantic = int(request.form['romantic'])
	famrel = int(request.form['famrel'])
	freetime = int(request.form['freetime'])
	goout = int(request.form['goout'])
	Dalc = int(request.form['dalc'])
	Walc = int(request.form['walc'])
	health = int(request.form['health'])
	absences = int(request.form['absences'])
	if(absences > 93):
		absences = 93
	G1 = int(request.form['G1'])
	if(G1 > 20):
		G1 = 20

	G2 = int(request.form['G2'])
	if(G2 > 20):
		G2 = 20

	feature_values = np.array([[sex, age,address,famsize,Pstatus,Medu,Fedu, Mjob,Fjob,reason,guardian,traveltime,studytime,failures,
		schoolsup,famsup,paid,activities,nursery,higher,internet,romantic,famrel,freetime,goout,Dalc,Walc,health,absences,G1,G2
		]])


	prediction = model.predict(feature_values)
	gradePrediction = KNNmodel.predict(feature_values)

	if(prediction[0] == 1):
		prediction_text = "Pass"
		flag = 1
	else:
		prediction_text = "Fail"
		flag = 0

	return render_template('prediction.html', prediction_text = prediction_text, value = flag, grades = gradePrediction)

@app.route('/grades', methods = ['GET'])
def grades():
	my_var = request.args.get('my', None)
	return render_template('grades.html', grades  = int(my_var[1]))

	
if __name__ == "__main__":
	app.run(debug = True)

