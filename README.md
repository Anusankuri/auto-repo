from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
# import some data within sklearn for iris classification 
iris = datasets.load_iris()
X = iris.data 
y = iris.target
 
# Splitting data into train and testing part
# The 25 % of data is test size of the data 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)
# importing pipes for making the Pipe flow
from sklearn.pipeline import Pipeline
# pipe flow is :
# PCA(Dimension reduction to two) -> Scaling the data -> DecisionTreeClassification 
pipe = Pipeline([('pca', PCA(n_components = 2)), ('std', StandardScaler()), ('decision_tree', DecisionTreeClassifier())], verbose = True)
 
# fitting the data in the pipe
pipe.fit(X_train, y_train)
 
# scoring data 
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, pipe.predict(X_test)))
Output: 
 

[Pipeline] ............... (step 1 of 3) Processing pca, total=   0.0s
[Pipeline] ............... (step 2 of 3) Processing std, total=   0.0s
[Pipeline] ..... (step 3 of 3) Processing Decision_tree, total=   0.0s
0.9736842105263158
Important property: 
 

pipe.named_steps: pipe.named_steps is a dictionary storing the name key linked to the individual objects in the pipe. For example:
pipe.named_steps['decision_tree'] # returns a decision tree classifier object  
Hyper parameters: 
There are different set of hyper parameters set within the classes passed in as a pipeline. To view them, pipe.get_params() method is used. This method returns a dictionary of the parameters and descriptions of each classes in the pipeline. 
Example: 
 




from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
# import some data within sklearn for iris classification 
iris = datasets.load_iris()
X = iris.data 
y = iris.target
 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)
 
from sklearn.pipeline import Pipeline
pipe = Pipeline([('pca', PCA(n_components = 2)), ('std', StandardScaler()), ('Decision_tree', DecisionTreeClassifier())], verbose = True)
 
pipe.fit(X_train, y_train)
 
# to see all the hyper parameters
pipe.get_params()
