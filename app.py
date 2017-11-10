import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.graph_objs as go
#import json
#from textwrap import dedent as d
from sklearn.metrics import precision_recall_curve, confusion_matrix, f1_score, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from dash.dependencies import Input, Output



app = dash.Dash(__name__);



def generate_table(dataframe, max_rows = 10):
	return html.Table(
	[html.Tr([html.Th(col) for col in dataframe.columns])] + 
	[html.Tr([html.Td(dataframe.iloc[i][col]) for col in dataframe.columns]) for i in range(min(len(dataframe), max_rows))])

df = pd.read_csv('http://www.sharecsv.com/s/6eaff21deff615a23daccef8aa8374c0/HR_Employee_Attrition_Data.csv');
dfo = df.copy();
df.drop('EmployeeCount', axis = 1, inplace = True);
df.drop('EmployeeNumber', axis = 1, inplace = True);
df.drop('StandardHours', axis = 1, inplace = True);
df.drop('Over18', axis = 1, inplace = True);
columns = list(df.columns);
df.replace({'Yes':1, 'No':0}, inplace = True);
df['JobRole'].replace({'Research Director':8, 'Manager':7, 'Healthcare Representative':6,
'Manufacturing Director':5, 'Research Scientist':4, 'Sales Executive':3, 'Human Resources':2,
'Laboratory Technician':1, 'Sales Representative':0},inplace = True);

df['Department'].replace(['Human Resources', 'Research & Development', 'Sales'],
[1, 0, 2], inplace = True);

df['MaritalStatus'].replace(['Divorced', 'Married', 'Single'], [0, 1, 2],
inplace = True);

df['BusinessTravel'].replace(['Non-Travel', 'Travel_Rarely', 'Travel_Frequently'],
[0, 1, 2], inplace = True);

df['EducationField'].replace(['Other', 'Medical', 'Life Sciences', 'Marketing',
'Technical Degree', 'Human Resources'], [0, 1, 2, 3, 4, 5], inplace = True);

df.replace(['Male', 'Female'], [1, 0], inplace = True);

conts = ['Age', 'DailyRate', 'DistanceFromHome', 'HourlyRate', 'MonthlyIncome', 'MonthlyRate', 'PercentSalaryHike', 'TotalWorkingYears', 'YearsAtCompany', 'YearsSinceLastPromotion', 'YearsWithCurrManager']
for i in conts:
    df[i] = pd.cut(df[i], 5, labels = False, include_lowest = True);

corr = list(df.drop('Attrition', axis = 1).corrwith(df['Attrition']));
corr_col = columns;
corr_col.remove('Attrition');

Y = df['Attrition'];
X = df.drop(['Attrition'], axis = 1);
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, shuffle = True);
clf = LogisticRegression(C = 10);
clf.fit(X_train, Y_train);
pred = clf.predict(X_test);
score = clf.score(X_test, Y_test);
con_mat = confusion_matrix(Y_test, pred);
f1 = f1_score(Y_test, pred);
prec, rec, thresh = precision_recall_curve(Y_test, pred);


#Neural Network________________________________________________________________________________________________________________________________________________________________________
clf_nn = MLPClassifier(hidden_layer_sizes = (20), alpha = 0.001, activation = 'logistic', solver = 'lbfgs');
clf_nn.fit(X_train, Y_train);
acc = clf_nn.score(X_test, Y_test);
prednn = clf_nn.predict(X_test);
precnn, recnn, threshnn = precision_recall_curve(Y_test, prednn);
f1nn = f1_score(Y_test, prednn);
con_mat_nn = confusion_matrix(Y_test, prednn);
scorenn = clf_nn.score(X_test, Y_test);
#SVM_______________________________________________________________________________________________-
clfsvm = SVC(C = 10, kernel = 'rbf', class_weight = 'balanced');
clfsvm.fit(X_train, Y_train);
predsvm = clfsvm.predict(X_test);
precsvm, recsvm, thresholdssvm = precision_recall_curve(Y_test, predsvm);
f1svm = f1_score(Y_test, predsvm);
con_mat_svm = confusion_matrix(Y_test, predsvm);
accsvm = clfsvm.score(X_test, Y_test);
fpr, tpr, thresh = roc_curve(Y_test, pred);
fprnn, tprnn, threshnn = roc_curve(Y_test, prednn);
fprsvm, tprsvm, threshsvm = roc_curve(Y_test, predsvm);
scoresvm = clfsvm.score(X_test, Y_test);
app.css.append_css({"external_url": "https://codepen.io/chriddyp/pen/bWLwgP.css"});


colors = {'background': '#111111', 'text': '#7FDBFF'};




m5 = df.describe();

m7 = '''

'''
m8 = '''
The correlation of all the attributes(with Attrition), is very low, most of them below 0.2, except for OverTime and JobRole. This may be due to a non-linear relation between various attributes and Attrition. 

We can see below that the accuracy for the Logistic Regression model is a good score(near 90%). But, that in intself is not a metric that\'ll suffice for classification. We have to note that the data is heavily skewed(about 16% Attrition rate), and we need other metrics to evaluate the models. Hence, the inclusion of Precision, Recall and F1-Score. Although the precision is decent for Logisitic regression, the recall is very low. It correctly classifies only around 50% of the people that actually left the company, and makes this an exercise in futility. This was expected, because the Logistic regression, is only a kind of extension for linear regression, and hence can classify well in a dataset that follows near linear relation unlike the dataset at hand. We\'ll have to use models that can take into account the non-linear nature of the relation between features.
'''

m9 = '''
Using a neural network is a good idea. The accuracy of the classifier on the test set is very good, and also the precision and recall are decent. This was expected as the neural network can better interepret the non-linear dependance of the attrition on various features. The logistic regression is a very good estimator when the data is linearly related, because it is just that: A linear model.

However, working with neural networks has its problems. It is basically a black-box with a lot of parameters, each of which need to be tuned to get the optimal result. Also, neural networks try and optimize a non-convex objective, meaning everytime we use train a network, we never know if it has arrived at the global minima. Also, training a neural network is difficult due to all those parameters, and an expert in them required to get the neuorons, and the hidden layers right. All though there are a few heuristics that are followed to find the parameters, it is not a concrete method, and can only be arrived at by a lot of trial and error. It is better to get a deep understanding of the problem statement, and find insights into the features, and using a good model, rather than employing a neural network and play around with the parameters.
'''
m11 = '''
It\'s clearly visible that the Support Vector Machine has given us the best results. Not only is the classification accuracy the best, the precision and recall are far better than the other models that\'ve been used. This is probably due to the fact that, the SVM is built on sound theoretical principles, whereas the neural network is a kind of heuristic. Also, for a neural network, the input features is bound to increase the complexity in the network, and hence arriving at the global minima is that much more difficult. Whereas, for the SVM, especially in this case, since there are only relatively few samples, there's not much complexity, and also finding the support vectors is easy and quick.
'''




app.layout = html.Div(style = {'backgroundColor': '#FFF8DC'},
			 		  children = [
					 		 	  #Head__________________________________________________________________________________________
					 		 	  html.H1(children = 'Employee Attrition', style = {'textAlign': 'center', 'margin': 'auto'}),
					 	 		  #Exploring the Data________________________________________________________________________________
					 	 		  html.H6(children = 'After looking through the data, we find there are no null fields anywhere in the data. The attributes \'EmployeeCount\', \'EmployeeNumber\', \'StandardHours\', \'Over18\', because they\'re the same for everyone, and will not contribute towards information about the employees\' attrition, and can be safely removed. Also, the continuous attributes have been converted to categorical ones, five buckets each. And then the attributes with string values have also been modified to contain integer values. Now, we can directly feed our data to various models, and find the best fit for the problem', style = {'margin': 17.5}),
					 	 		  html.H5(children = 'The summary of the data now looks like this...', style = {'margin': 17.5}),
					 	 		  html.Div(
					 	 		  		   style = {'height': '100%', 'overflow': 'auto', 'backgroundColor': '#FFE4C4', 'margin': 17.5, 'border-width': 1, 'border-color': 'black', 'border-style': 'solid'},
					 	 		  		   children = [
					 	 		  		   			   html.Div(generate_table(m5), style = {'margin-left': 10, 'margin-right': 10})
					 	 		  		   			  ]
					 	 		  		  ),
					 	 		  
					 	 		  #Plot Correlation of features with the Attrition_____________________________________________________________________________
					 	 		  html.H6(children = 'The correlation of various features with the Attrition of employees can give us an idea as to which features explain the change in attrition in a better fashion.', style = {'margin': 17.5}),
					 	 		  dcc.Graph(style = {'width': '75%', 'height': 650, 'display':'block', 'margin': 'auto', 'backgroundColor': '#F5DEB3'},
					 	 		  		   id = 'Correlation',
					 	 		  		   figure = {
					 	 		  		   				'data': [
					 	 		  		   						go.Bar(
					 	 		  		   							  x = corr_col,
					 	 		  		   							  y = corr,
					 	 		  		   							  marker = {'color': 'rgb(158,202,225)', 'line': {'color': 'rgb(8,48,107)', 'width': 1.5}, },
					 	 		  		   							  opacity = 0.6
					 	 		  		   							  ),
					 	 		  		   						],
					 	 		  		   				'layout': go.Layout(
					 	 		  		   								    plot_bgcolor = '#F5DEB3'
					 	 		  		   								   )
					 	 		  		   			}
					 	 		  		   ),
					 	 		  html.H6([dcc.Markdown(children = m8)], style = {'margin': 17.5}),
					 	 		  #Dropdown__________________________________________________________________________________________________________________________________
					 	 		  html.Div(style = {'width': '60%', 'margin-left': '20%'}, 
					 	 		  children = 
					 	 		  		  [
					 	 		  		   html.Div(id = 'F1_Score', style = {'width': '100%'}),
					 	 		  		   dcc.Dropdown(
					 	 		  		   			   id = 'Model',
					 	 		  		   			   options = [{'label': 'Logistic Regression', 'value': ['Logistic Regression', con_mat, fpr, tpr, thresh, auc(fpr, tpr), prec[1], rec[1], f1, score]}, {'label': 'Neural Network', 'value': ['Neural Network', con_mat_nn, fprnn, tprnn, threshnn, auc(fprnn, tprnn), precnn[1], recnn[1], f1nn, scorenn]}, {'label': 'Support Vector Machine', 'value': ['Support Vector Machine', con_mat_svm, fprsvm, tprsvm, threshsvm, auc(fprsvm, tprsvm), precsvm[1], recsvm[1], f1svm, scoresvm]}],
					 	 		  		   			   value = ['Logistic Regression', con_mat, fpr, tpr, thresh, auc(fpr, tpr), prec[1], rec[1], f1, score],
					 	 		  		   			   ),
					 	 		  		   dcc.Graph(style = {'width': '33%', 'display': 'inline-block', 'backgroundColor': '#f8c89f'}, id = 'pra'),
					 	 		  		   dcc.Graph(style = {'width': '33%', 'display': 'inline-block', 'backgroundColor': '#f8c89f'}, id = 'CM'),
					 	 		  		   dcc.Graph(style = {'width': '33%', 'display': 'inline-block', 'backgroundColor': '#f8c89f'}, id = 'ROC'),
					 	 		  		   ]),
					 	 		  #Final Thoughts_________________________________________________________________________________________________________
					 	 		  html.H6([dcc.Markdown(children = m9)], style = {'margin': 17.5}),
					 	 		  html.H6([dcc.Markdown(children = m11)], style = {'margin': 17.5})
					 	 		  ]
					 );



@app.callback(
			  Output('CM', 'figure'),
			  [Input('Model', 'value')]
			 )
def update_graph(modelvalue):
	return {
		    'data': [
		    		 go.Heatmap(
		    		 		    z = modelvalue[1],
		    		 		    x = ['Predicted: 0', 'Predicted: 1'],
		    		 		    y = ['Truth: 0', 'Truth: 1'],
		    		 		    colorscale = 'Portland',
		    		 		    )
		    		],
		    'layout': go.Layout(
		    				    title = 'Confusion Matrix for %s' % modelvalue[0],
		    				    plot_bgcolor = '#F5DEB3'
		    				   )
		   }

@app.callback(
			  Output('ROC', 'figure'),
			  [Input('Model', 'value')]
			 )
def update_graph_roc(value):
	return {
		    'data': [
		    		 go.Scatter(
		    		 		    x = value[2],
		    		 		    y = value[3],
		    		 		    text = value[4],
		    		 		    name = 'AUC %0.3f' % value[5],
		    		 		    )
		    		],
		    'layout': go.Layout(
		    				    title = 'ROC Curve for %s' % value[0],
		    				    plot_bgcolor = '#F5DEB3',
		    				    showlegend = True
		    				   )
		   }

@app.callback(
			  Output('pra', 'figure'),
			  [Input('Model', 'value')]
			 )
def update_text(value):
	return {
		    'data': [
		    		 go.Bar(
		    		 		x = ['Precision', 'Recall', 'F1-Score',  'Accuracy'],
		    		 		y = [value[6], value[7], value[8], value[9]]
		    		 )],
		    'layout': go.Layout(
		    				   	title = 'Performance Metrics for %s' % value[0],
		    				   	plot_bgcolor = '#F5DEB3'
		    				   )
		   }
		  



if __name__ == '__main__':
	app.run_server(debug = True)
