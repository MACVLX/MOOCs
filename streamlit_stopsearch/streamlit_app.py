# Core Pkgs
import streamlit as st 

# EDA Pkgs
import pandas as pd 
import numpy as np 


# Utils
import os
import joblib 
import json
import pickle

from custom_transformers_3.transformer import * 
from utils import *
from Exploratory import ExploratoryAnalysis

# Data Viz Pkgs
import matplotlib.pyplot as plt 
import matplotlib
matplotlib.use('Agg')


# from multiapp import MultiApp

# ML Interpretation
import lime
import lime.lime_tabular


html_temp = """
		<div style="background-color:{};padding:10px;border-radius:10px">
		<h1 style="color:white;text-align:center;">Stop & Search UK app</h1>
		
		</div>
		"""

# Avatar Image using a url
avatar1 ="https://www.w3schools.com/howto/img_avatar1.png"
avatar2 ="https://www.w3schools.com/howto/img_avatar2.png"

result_temp ="""
	<div style="background-color:#464e5f;padding:10px;border-radius:10px;margin:10px;">
	<h4 style="color:white;text-align:center;">Algorithm:: {}</h4>
	<img src="https://www.w3schools.com/howto/img_avatar.png" alt="Avatar" style="vertical-align: middle;float:left;width: 50px;height: 50px;border-radius: 50%;" >
	<br/>
	<br/>	
	<p style="text-align:justify;color:white">{} % probalibilty that Patient {}s</p>
	</div>
	"""

result_temp2 ="""
	<div style="background-color:#464e5f;padding:10px;border-radius:10px;margin:10px;">
	<h4 style="color:white;text-align:center;">Algorithm:: {}</h4>
	<img src="https://www.w3schools.com/howto/{}" alt="Avatar" style="vertical-align: middle;float:left;width: 50px;height: 50px;border-radius: 50%;" >
	<br/>
	<br/>	
	<p style="text-align:justify;color:white">{} % probalibilty that Patient {}s</p>
	</div>
	"""

prescriptive_message_temp ="""
	<div style="background-color:silver;overflow-x: auto; padding:10px;border-radius:5px;margin:10px;">
		<h3 style="text-align:justify;color:black;padding:10px">Recommended Life style modification</h3>
		<ul>
		<li style="text-align:justify;color:black;padding:10px">Exercise Daily</li>
		<li style="text-align:justify;color:black;padding:10px">Get Plenty of Rest</li>
		<li style="text-align:justify;color:black;padding:10px">Exercise Daily</li>
		<li style="text-align:justify;color:black;padding:10px">Avoid Alchol</li>
		<li style="text-align:justify;color:black;padding:10px">Proper diet</li>
		<ul>
		<h3 style="text-align:justify;color:black;padding:10px">Medical Mgmt</h3>
		<ul>
		<li style="text-align:justify;color:black;padding:10px">Consult your doctor</li>
		<li style="text-align:justify;color:black;padding:10px">Take your interferons</li>
		<li style="text-align:justify;color:black;padding:10px">Go for checkups</li>
		<ul>
	</div>
	"""





@st.cache
def load_model():
	with open('../columns_RF_opt_2.json') as fh:
		columns = json.load(fh)

	with open('../pipeline_RF_opt_2.pickle', 'rb') as fh:
		pipeline = joblib.load(fh)

	with open('../dtypes_RF_opt_2.pickle', 'rb') as fh:
		dtypes = pickle.load(fh)

	return columns, pipeline, dtypes

@st.cache
def load_train_df():
	URL = 'https://storage.googleapis.com/capstone4ldsaa/train.csv'
	df = pd.read_csv(URL)
	
	return df


	

def main():
	"""Hep Mortality Prediction App"""
	
	st.markdown(html_temp.format('royalblue'),unsafe_allow_html=True)
	

	menu = ['Home',"EDA","Use Model","Metrics"]

	choice = st.sidebar.selectbox("Menu",menu)
	if choice == "Home":
				
		st.markdown("""
	<div style="background-color:silver;overflow-x: auto; padding:10px;border-radius:5px;margin:10px;">
		<h3 style="text-align:justify;color:black;padding:10px">Project overview</h3>
		<p>A summary of the project premisses, objectives and requirements.</p>
	</div>
	""",unsafe_allow_html=True)
		# st.image(load_image('images/hepimage.jpeg'))


	elif choice == "EDA":

		# with st.sidebar.expander('Basic exploratory analysis options'):
		try:
			with st.spinner('Loading data....'):
				df = load_train_df()				
		except:
			st.error('csv file upload error')
		else:
			EA = ExploratoryAnalysis(df)
			st.sidebar.markdown('#### Exploratory analysis')
			features=st.sidebar.radio("",['Head','Describe','Info','Data entry issues','Unique values and frequency',  'Gender, Ethnicity and Age'])
			# basics = st.sidebar.radio('',('Head','Describe','Info','Data entry issues'))
			if features == 'Head':
				st.subheader('Dataframe head:')
				st.write(df.head(20))
				st.markdown("For the task at hand, the IT Department has made available a dataset comprising 660 661 stop and search events spread throughout the country. This dataset is composed of features that can be used for modelling -  'Type', 'Date', 'Part of a policing operation', 'Latitude', 'Longitude', 'Gender', 'Age range', 'Self-defined ethnicity', 'Officer-defined ethnicity', 'Legislation','Object of search', 'station' - by features which will be used to build a classification target given there isn’t a specific ‘target’ or ‘ label’ feature - 'Outcome', 'Outcome linked to object of search' - and by the feature  'Removal of more than just outer clothing',  to be used in the analysis only. As previously mentioned a search is considered successful if the outcome is positive and is related to the search. Except for ‘Latitude’ and ‘Longitude’, all the others are categorical variables.")
			elif features == 'Describe':
				st.subheader('Dataframe description:')
				st.write(df.describe())
			elif features =='Info':
				st.subheader('Dataframe informations:')
				st.text(EA.info())
			# elif basics =='Isnull':
			# 	st.subheader('Null occurrences')
			# 	st.write(df.isnull().sum())
			elif features == 'Unique values and frequency':
				col = st.selectbox('Choose a column for see unique values',EA.columns)
				st.subheader('Unique values and frequency')
				st.write(EA.info2(col))
			elif features == 'Data entry issues':
				st.write('Data entry issues')
				st.image('images/overhaul_nulls.png')
				st.markdown('There seems to be some issues regarding data entries although in general most of the dataset is quite well populated. Maybe not surprisingly, the London metropolitan station accounts for more than 50% of all the data points in this dataset.')
				st.markdown('In regards to missing data across all stations, the columns ‘Outcome linked to object of search’ and ‘Removal of more than just outer clothing’ are the most affected')
				st.image('images/%_nulls.png')
				st.markdown("A more granular analysis by police station allows for in depth appraisal of the reality of record keeping per  station (Annex 3). Some stations are top exemplar  - essex, suffolk, sussex, northamptonshire, norfolk and gloucester - with near 0% of data missing whereas other stations like thames-valley, lancashire and dyfed-powys have severe data keeping records with more than 30% of data missing, particularly in respect to geolocation, information if it was part of police operation and the two above mentioned categories. The most severe issue pertaining to missing data is related to the feature ‘Outcome linked to object of search’ in the metropolitan station. There is no data at all, which will disallow these data points to be included in any modelling attempt and impair any statistical analysis related to outcome in this important area of the country. The station gwent and humberside also have no data in this feature. Another worrying fact related to this project is the almost entire lack of report regarding the removal of outer clothing. Some stations like gwent, cleveland, north-yorkshire, metropolitan and surrey do not report this information  at all and many others have less than 50% reporting in this category. For some columns it makes sense to infer the missing data, for example in ‘Outcome linked to object of search’ any missing values are most likely to be False since officers tend to forget to go back to the application.There seems to be a substantial amount of records where ‘Outcome’ is negative but the column ‘Outcome linked to the object of search is True. This indicates potential compromise when training any model as these data points may be of no use. In figure 3 we can clearly see that some stations (warwickshire,btp,west-yorkshire and derbyshire) have more than 50% of observations with this mismatch. As mentioned the features ‘Outcome’ and 'Outcome linked to object of search' will be used to build our target labels for binary classification. The latter is a boolean feature with True or False values. The Outcome feature is a wide range of consequences from arrestas to penalties or mere caution but the vast majority is that no action was taken.  After constructing the target feature we achieve an imbalanced dataset where the positive outcome accounts for approximately 20% of the observations. By removing the stations where the target is not possible to be constructed (metropolitan, gwent and humberside) the  final data set is only 46% of the original data.")

			elif features == 'Gender, Ethnicity and Age':
				st.write("Gender, Ethnicity and Age groups distribution of stops can be seen in the chart below. The Gender feature has 3 categories: Female, Male and Other. Given that the representativity of ‘other’ is extremely low in the dataset (less than 0.001%) we will exclude this from our analysis and modelling. There is an over representativity of Male compared to Female. Females account for little more than 8% but if the metropolitan station is removed then it increases slightly to more than 10%. In regards to Ethnic groups we have 2 different features: Officer defined and self-defined. For modelling purposes it only makes sense to use the Officer-defined feature as this is the only information present prior to any decision making. By far the majority is composed of ‘white’ subgroups making white man the most stopped.The age brackets recorded show a majority of young adults between 18 and 34 years old. Possibly of concern is the fact that there are 384 occurrences of children aged less than 10 years old.")
				st.image('images/stops_per_group.png')
				st.write("In regards to ethnicity we also investigated if there was any significant discrepancy between Officer-defined  and self-defined reported values.  Although not significant, it seems that officers tend to dismiss the categories 'Mixed' and 'Other' , distributing them more across 'White' and 'Black' categories")
				st.image('images/Match between self-defined and police-defined ethnicity.png')




		# df['Outcome'].value_counts().plot(kind='bar')
		# st.pyplot()	

	# 				# Freq Dist Plot
	# 				freq_df = pd.read_csv("data/freq_df_hepatitis_dataset.csv")
	# 				st.bar_chart(freq_df['count'])


	# 				if st.checkbox("Area Chart"):
	# 					all_columns = df.columns.to_list()
	# 					feat_choices = st.multiselect("Choose a Feature",all_columns)
	# 					new_df = df[feat_choices]
	# 					st.area_chart(new_df)
						


	elif choice == "Use Model":


		st.subheader("Predictive Analytics")
		with st.form("my_form"):
			id = st.text_input("observation id")
			
			type = st.selectbox("Type",valid_category_map.get('Type'))
			
			import datetime
			today = datetime.date.today()
			try:
				date = st.date_input("Date",today)
				date = date.strftime("%m/%d/%Y")
			except:
				st.error('Incorrect or missing date.')
			
			Part_policing_operation=st.selectbox('Part of a policing operation',valid_category_map.get('Part of a policing operation'))
			
			latitude = st.number_input("Latitude",format='%8.6f')
			if latitude == 0.0000:
				latitude = None
			
			longitude = st.number_input("Longitude",format='%8.6f')
			if longitude == 0.0000:
				longitude = None
			
			Gender = st.selectbox('Gender',valid_category_map.get('Gender'))
			Age_range = st.selectbox('Age range',valid_category_map.get('Age range'))
			Officer_defined_ethnicity=st.selectbox('Officer-defined ethnicity',valid_category_map.get('Officer-defined ethnicity'))
			Legislation=st.selectbox('Legislation',valid_category_map.get('Legislation'))
			Object_search=st.selectbox('Object of search',valid_category_map.get('Object of search'))
			station=st.selectbox('station',valid_category_map.get('station'))

			# Every form must have a submit button.
			submitted = st.form_submit_button("Advice")
			if submitted:
				columns, pipeline, dtypes = load_model()
				observation = {
					'Type': type,
        			'Date': date,
        			'Part_policing_operation': Part_policing_operation,
        			'Latitude' : latitude,
        			'Longitude' : longitude,
        			'Gender' : Gender,
        			'Age range' : Age_range,
        			'Officer-defined ethnicity' : Officer_defined_ethnicity,
                    'Legislation' : Legislation,
        			'Object of search' : Object_search,
        			'station' : station
				}


				obs = pd.DataFrame([observation], columns=columns).astype(dtypes)
				proba = pipeline.predict_proba(obs)[0, 1]

				proba
				prediction = pipeline.predict(obs)[0]
				prediction
				# prediction=bool(prediction)


	# 					# st.write(prediction)
	# 					# prediction_label = {"Die":1,"Live":2}
	# 					# final_result = get_key(prediction,prediction_label)
	# 					if prediction == 1:
	# 						st.warning("Patient Dies")
	# 						pred_probability_score = {"Die":pred_prob[0][0]*100,"Live":pred_prob[0][1]*100}
	# 						st.subheader("Prediction Probability Score using {}".format(model_choice))
	# 						st.json(pred_probability_score)
	# 						st.subheader("Prescriptive Analytics")
	# 						st.markdown(prescriptive_message_temp,unsafe_allow_html=True)
							
	# 					else:
	# 						st.success("Patient Lives")
	# 						pred_probability_score = {"Die":pred_prob[0][0]*100,"Live":pred_prob[0][1]*100}
	# 						st.subheader("Prediction Probability Score using {}".format(model_choice))
	# 						st.json(pred_probability_score)
							
	# 				if st.checkbox("Interpret"):
	# 					if model_choice == "KNN":
	# 						loaded_model = load_model("models/knn_hepB_model.pkl")
							
	# 					elif model_choice == "DecisionTree":
	# 						loaded_model = load_model("models/decision_tree_clf_hepB_model.pkl")
							
	# 					else:
	# 						loaded_model = load_model("models/logistic_regression_hepB_model.pkl")
							

	# 						# loaded_model = load_model("models/logistic_regression_model.pkl")							
	# 						# 1 Die and 2 Live
	# 						df = pd.read_csv("data/clean_hepatitis_dataset.csv")
	# 						x = df[['age', 'sex', 'steroid', 'antivirals','fatigue','spiders', 'ascites','varices', 'bilirubin', 'alk_phosphate', 'sgot', 'albumin', 'protime','histology']]
	# 						feature_names = ['age', 'sex', 'steroid', 'antivirals','fatigue','spiders', 'ascites','varices', 'bilirubin', 'alk_phosphate', 'sgot', 'albumin', 'protime','histology']
	# 						class_names = ['Die(1)','Live(2)']
	# 						explainer = lime.lime_tabular.LimeTabularExplainer(x.values,feature_names=feature_names, class_names=class_names,discretize_continuous=True)
	# 						# The Explainer Instance
	# 						exp = explainer.explain_instance(np.array(feature_list), loaded_model.predict_proba,num_features=13, top_labels=1)
	# 						exp.show_in_notebook(show_table=True, show_all=False)
	# 						# exp.save_to_file('lime_oi.html')
	# 						st.write(exp.as_list())
	# 						new_exp = exp.as_list()
	# 						label_limits = [i[0] for i in new_exp]
	# 						# st.write(label_limits)
	# 						label_scores = [i[1] for i in new_exp]
	# 						plt.barh(label_limits,label_scores)
	# 						st.pyplot()
	# 						plt.figure(figsize=(20,10))
	# 						fig = exp.as_pyplot_figure()
	# 						st.pyplot()



					


	# 		else:
	# 			st.warning("Incorrect Username/Password")


	# elif choice == "SignUp":
	# 	new_username = st.text_input("User name")
	# 	new_password = st.text_input("Password", type='password')

	# 	confirm_password = st.text_input("Confirm Password",type='password')
	# 	if new_password == confirm_password:
	# 		st.success("Password Confirmed")
	# 	else:
	# 		st.warning("Passwords not the same")

	# 	if st.button("Submit"):
	# 		create_usertable()
	# 		hashed_new_password = generate_hashes(new_password)
	# 		add_userdata(new_username,hashed_new_password)
	# 		st.success("You have successfully created a new account")
	# 		st.info("Login to Get Started")






	st.sidebar.markdown("with contributions from:")
	st.sidebar.markdown('[https://github.com/rafaelloni/EAT_app](https://github.com/rafaelloni/EAT_app)')
	st.sidebar.markdown('[https://github.com/upraneelnihar/streamlit-multiapps](https://github.com/upraneelnihar/streamlit-multiapps)')
if __name__ == '__main__':
	main()