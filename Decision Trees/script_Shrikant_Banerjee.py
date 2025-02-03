from sklearn import datasets 
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
import pandas as pd

############## FOR EVERYONE ##############
# Please note that the blanks are here to guide you for this first assignment, but the blanks are  
# in no way representative of the number of commands/ parameters or length of what should be inputted.

### PART 1 ###
# Scikit-Learn provides many popular datasets. The breast cancer wisconsin dataset is one of them. 
# Write code that fetches the breast cancer wisconsin dataset. 
# Hint: https://scikit-learn.org/stable/datasets/toy_dataset.html
# Hint: Make sure the data features and associated target class are returned instead of a "Bunch object".
X, y = datasets.load_breast_cancer(return_X_y=True) #(4 points) 

# Check how many instances we have in the dataset, and how many features describe these instances
print("There are", X.shape[0], "instances described by", X.shape[1], "features.") #(4 points)  

# Create a training and test set such that the test set has 40% of the instances from the 
# complete breast cancer wisconsin dataset and that the training set has the remaining 60% of  
# the instances from the complete breast cancer wisconsin dataset, using the holdout method. 
# In addition, ensure that the training and test sets # contain approximately the same 
# percentage of instances of each target class as the complete set.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state = 42)  #(4 points) 

# Create a decision tree classifier. Then Train the classifier using the training dataset created earlier.
# To measure the quality of a split, using the entropy criteria.
# Ensure that nodes with less than 6 training instances are not further split
clf = tree.DecisionTreeClassifier(criterion='entropy', min_samples_split=6)  #(4 points) 
clf.fit(X_train, y_train)  #(4 points) 

# Apply the decision tree to classify the data 'testData'.
predC = clf.predict(X_test)  #(4 points) 

# Compute the accuracy of the classifier on 'testData'
print('The accuracy of the classifier is', accuracy_score(y_test, predC))  #(2 point) 

# Visualize the tree created. Set the font size the 12 (4 points) 
v_tree = tree.plot_tree(clf, fontsize=12)  
plt.show()
### PART 2.1 ###
# Visualize the training and test error as a function of the maximum depth of the decision tree
# Initialize 2 empty lists where you will save the training and testing accuracies 
# as we iterate through the different decision tree depth options.
trainAccuracy = []  #(1 point) 
testAccuracy = [] #(1 point) 
# Use the range function to create different depths options, ranging from 1 to 15, for the decision trees
depthOptions = range(1,16) #(1 point) 
for depth in depthOptions: #(1 point) 
    # Use a decision tree classifier that still measures the quality of a split using the entropy criteria.
    # Also, ensure that nodes with less than 6 training instances are not further split
    cltree = tree.DecisionTreeClassifier(criterion='entropy', min_samples_split=6, max_depth=depth) #(1 point) 
    # Decision tree training
    cltree.fit(X_train, y_train)#(1 point) 
    # Training error
    y_predTrain = cltree.predict(X_train) #(1 point) 
    # Testing error
    y_predTest = cltree.predict(X_test) #(1 point) 
    # Training accuracy
    trainAccuracy.append(accuracy_score(y_train, y_predTrain)) #(1 point) 
    # Testing accuracy
    testAccuracy.append(accuracy_score(y_test, y_predTest)) #(1 point) 

# Plot of training and test accuracies vs the tree depths (use different markers of different colors)
plt.plot(depthOptions, trainAccuracy, color='g', marker='.') #(3 points) 
plt.plot(depthOptions, testAccuracy, color='r', marker='1')
plt.legend(['Training Accuracy','Test Accuracy']) # add a legend for the training accuracy and test accuracy (1 point) 
plt.xlabel('Tree Depth') # name the horizontal axis 'Tree Depth' (1 point) 
plt.ylabel('Classifier Accuracy') # name the horizontal axis 'Classifier Accuracy' (1 point) 
plt.show()
# Fill out the following blanks: #(4 points (2 points per blank)) 
""" 
According to the test error, the best model to select is when the maximum depth is equal to 3, approximately. 
But, we should not use select the hyperparameters of our model using the test data, because it will overfit.
"""

### PART 2.2 ###
# Use sklearn's GridSearchCV function to perform an exhaustive search to find the best tree depth and the minimum number of samples to split a node
# Hint: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
# Define the parameters to be optimized: the max depth of the tree and the minimum number of samples to split a node
parameters = {'max_depth': range(1, 16), 'min_samples_split': range(2, 6)} #(6 points)
# We will still grow a decision tree classifier by measuring the quality of a split using the entropy criteria. 
clf2 = GridSearchCV(tree.DecisionTreeClassifier(criterion='entropy'), parameters) #(6 points)
clf2.fit(X_train, y_train) #(4 points)
tree_model = clf2.best_estimator_ #(4 points)
print("The maximum depth of the tree is", tree_model.max_depth, 
      'and the minimum number of samples required to split a node is', tree_model.min_samples_split) #(6 points)

# The best model is tree_model. Visualize that decision tree (tree_model). Set the font size the 12 
tree.plot_tree(tree_model, filled=True, fontsize=12) #(4 points)
plt.show()
# Fill out the following blank: #(2 points)
""" 
This method for tuning the hyperparameters of our model is acceptable, because gridsearch
considers all possible parameter combinations and compares of which the best combination is kept. 
"""

# Explain below what is tenfold Stratified cross-validation?  #(4 points)
"""
It is a variant of K-fold cross validation where it is used to test a machine learning model by splitting the data into k groups where the groups
are used as either a testing or training data set to be fit on the model and evaluated. In this case K is 10 and stratified meaning
that each group is represented by characteristics called strata.
"""



### PART 4 ###  

#db3 and db4 have incomplete data for ca (number of majour vessels recorded) and thal (cardio vascular defects) resulting colummns will be dropped  
db_col_names = ['age', 'sex', 'cp', 'trestbps', 'chols', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num']
db = pd.read_csv('data/processed.cleveland.data', names=db_col_names, sep=',', na_values='?')
db2 = pd.read_csv('data/reprocessed.hungarian.data', names=db_col_names, sep=' ', na_values='?')
db3 = pd.read_csv('data/processed.switzerland.data', names=db_col_names, sep=',', na_values='?')
db4 = pd.read_csv('data/processed.va.data', names=db_col_names, sep=',', na_values='?')
merged_db = pd.concat([db, db2, db3, db4], axis=0, ignore_index=True)
#merging datasets together
db_cleaned = merged_db.dropna()

#seperating features and target
x_db = db_cleaned.loc[:, db.columns != 'num']
y_db = db_cleaned['num']

#all data is in numbers so there is no need to one hot encode the data
#scaling is unnecessary

#split the dataset into training and testing
X_train2, X_test2, y_train2, y_test2 = train_test_split(x_db, y_db, test_size=0.5, random_state = 40)

#creating normal decision tree classifier and prediction
clf3 = tree.DecisionTreeClassifier(criterion='entropy', min_samples_split=3) 
clf3.fit(X_train2, y_train2)
predC2 = clf3.predict(X_test2)

#creating decision tree classifier using gridsearch
parameters2 = {'max_depth': range(1, 10), 'min_samples_split': range(3, 10)}
clf4 = GridSearchCV(tree.DecisionTreeClassifier(criterion='entropy'), parameters2)
clf4.fit(X_train2, y_train2)
tree_model2 = clf4.best_estimator_
predD2 = tree_model2.predict(X_test2)

#Listing important features used by original decision tree classifier
pd_dic1 = {'features':x_db.columns.tolist(), 'importance':clf3.feature_importances_}
dic1_df = pd.DataFrame(pd_dic1).sort_values(by='importance', ascending=False)
important_features_clf = dic1_df['features'].tolist()

#Listing important features used by gridsearch decision tree classifier
pd_dic2 = {'features':x_db.columns.tolist(), 'importance':tree_model2.feature_importances_}
dic2_df = pd.DataFrame(pd_dic2).sort_values(by='importance', ascending=False)
important_features_tree_model = dic2_df['features'].tolist()


print("The accuracy of the normal tree classifier is", accuracy_score(y_test2, predC2))
print("Important features in this model are: ", important_features_clf)

print("The accuracy of the gridsearch classifier is", accuracy_score(y_test2, predD2))
print("Important features in this model are: ", important_features_tree_model)

tree.plot_tree(tree_model2, filled=True, fontsize=8)
plt.show()
tree.plot_tree(clf3, filled=True, fontsize=4)
plt.show()