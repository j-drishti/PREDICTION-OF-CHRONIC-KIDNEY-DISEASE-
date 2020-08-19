# Loading the data

import pandas as pd
df = pd.read_csv("chronic_kidney_disease1.csv",header=None)

#Data cleaning

df=df.replace('ckd\t','ckd')
df1 = df[df[24]=='ckd']
df2 = df[df[24]=='notckd']
l1=[0,1,2,3,4,9,10,11,12,13,14,15,16,17]
for i in l1:
    df1[i].fillna(df1.describe()[i][1],inplace=True)
    df2[i].fillna(df2.describe()[i][1],inplace=True)
df1[5].fillna('normal',inplace=True)
df1[6].fillna('normal',inplace=True)

df2[5].fillna('normal',inplace=True)
df2[6].fillna('normal',inplace=True)
df2[7].fillna('notpresent',inplace=True)
df2[8].fillna('notpresent',inplace=True)
df2[18].fillna('no',inplace=True)
df2[19].fillna('no',inplace=True)
df2[20].fillna('no',inplace=True)
df2[21].fillna('good',inplace=True)
df2[22].fillna('no',inplace=True)
df2[23].fillna('no',inplace=True)

newdf=pd.concat([df1,df2], axis=0)

#Splitting dataframe into X and Y and changing the labels from string to binary 

new_df=pd.get_dummies(newdf.drop([24],axis=1))
print(new_df.columns)

y=newdf[24]
y.replace('ckd',1,inplace=True)
y.replace('notckd',0,inplace=True)
 
#Test-train split

from sklearn.model_selection import train_test_split
xtr,xte,ytr,yte = train_test_split(new_df, y, test_size=0.3, shuffle=True)

#Generating train and validation sets from the train set generated above for k-fold validation.
#Here, we have taken k=3

from sklearn.model_selection import KFold
kf = KFold(n_splits=3)
xtr=xtr.to_numpy()
ytr=ytr.to_numpy()
set = kf.split(xtr)
x1_tr =[]
x2_tr =[]
x3_tr =[]
y1_tr =[]
y2_tr =[]
y3_tr =[]
x1_v =[]
x2_v =[]
x3_v =[]
y1_v =[]
y2_v =[]
y3_v =[]
id = 1
for tri, tei in set:
    # print(tri, tei)
    if id == 1:
      x1_tr, x1_v = xtr[tri], xtr[tei]
      y1_tr, y1_v = ytr[tri], ytr[tei]
    elif id == 2:
      x2_tr, x2_v = xtr[tri], xtr[tei]
      y2_tr, y2_v = ytr[tri], ytr[tei]
    elif id == 3:
      x3_tr, x3_v = xtr[tri], xtr[tei]
      y3_tr, y3_v = ytr[tri], ytr[tei]
    id = id+1
 
#Applying Random Forest algorithm on the three validation sets and printing each sets accuracy

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report 

clf = RandomForestClassifier(max_depth=2, random_state=0)

clf_1 = clf.fit(x1_tr, y1_tr)
pred_1=clf_1.predict(x1_v)
print ('Accuracy Score for 1st validation set:')
print(accuracy_score(y1_v, pred_1))

clf_2 = clf.fit(x2_tr, y2_tr)
pred_2=clf_2.predict(x2_v)
print ('Accuracy Score for 2nd validation set:')
print(accuracy_score(y2_v, pred_2))

clf_3 = clf.fit(x3_tr, y3_tr)
pred_3=clf_3.predict(x3_v)
print ('Accuracy Score for 3rd validation set:')
print(accuracy_score(y3_v, pred_3))

#Selecting the best model from the k-fold method and generating accuracy, confusion matrix and classification report of the test set from the selected best validation model.

print("the best one seems to be the first model")

pred1=clf_1.predict(xte)
print ('Accuracy Score on test set:')
accuracy = accuracy_score(yte, pred1)
print(accuracy)
print("")
print ('Confusion Matrix :')
print (confusion_matrix(yte, pred1) )
print("")
print ('Report : ')
print (classification_report(yte, pred1) )
print("")
print("Feature importance :")
print(clf_1.feature_importances_)
print("")
print("Value of Parameters :")
print(clf_1.get_params)
#Plotting the ROC curve of the test set from the best validation model of random forest.

from sklearn.metrics import roc_curve, auc
import matplotlib
import matplotlib.pyplot as plt

y_score = clf_1.predict_proba(xte)

fpr_1_1, tpr_1_1, _ = roc_curve(yte, y_score[:, 1])
roc_auc_1_1 = auc(fpr_1_1, tpr_1_1)

plt.figure()

plt.plot(fpr_1_1, tpr_1_1, color='teal',lw=2, label='ROC curve for ckd (area = %0.2f)' % roc_auc_1_1)

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')    
plt.xlim([-0.1, 1.1])
plt.ylim([-0.1, 1.1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.show()

#Applying same for Logistic Regression

from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(random_state=0)

log_reg_1 = log_reg.fit(x1_tr, y1_tr)
pred_1=log_reg_1.predict(x1_v)
print ('Accuracy Score for 1st validation set:')
print(accuracy_score(y1_v, pred_1))

log_reg_2 = log_reg.fit(x2_tr, y2_tr)
pred_2=log_reg_2.predict(x2_v)
print ('Accuracy Score for 2nd validation set:')
print(accuracy_score(y2_v, pred_2))

log_reg_3 = log_reg.fit(x3_tr, y3_tr)
pred_3=log_reg_3.predict(x3_v)
print ('Accuracy Score for 3rd validation set:')
print(accuracy_score(y3_v, pred_3))

print("the best one seems to be the first model")

pred2 = log_reg_1.predict(xte)
print ('Accuracy Score of the test set:')
accuracy = accuracy_score(yte, pred2)
print(accuracy)
print("")
conf_matrix = confusion_matrix(yte, pred2) 
print ('Confusion Matrix :')
print (conf_matrix)
print("")
print ('Report : ')
print (classification_report(yte, pred2) )
print("")
# print("Feature importance :")
# print(log_reg.feature_importances_)
# print("")
print("Value of Parameters :")
print(log_reg_1.get_params)

y_score = log_reg_1.decision_function(xte)

fpr_1_2, tpr_1_2, _ = roc_curve(yte, y_score)
roc_auc_1_2 = auc(fpr_1_2, tpr_1_2)

plt.figure()

plt.plot(fpr_1_2, tpr_1_2, color='teal',lw=2, label='ROC curve for ckd (area = %0.2f)' % roc_auc_1_2)

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')    
plt.xlim([-0.1, 1.1])
plt.ylim([-0.1, 1.1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.show()

#quadratic polynomial kernel SVM.

from sklearn.svm import SVC

svc_ovr_poly = SVC(kernel='poly', degree=2,decision_function_shape='ovr')

svc_ovr_poly_1 = svc_ovr_poly.fit(x1_tr, y1_tr)
pred_1=svc_ovr_poly_1.predict(x1_v)
print ('Accuracy Score for 1st validation set:')
print(accuracy_score(y1_v, pred_1))

svc_ovr_poly_2 = svc_ovr_poly.fit(x2_tr, y2_tr)
pred_2=svc_ovr_poly_2.predict(x2_v)
print ('Accuracy Score for 2nd validation set:')
print(accuracy_score(y2_v, pred_2))

svc_ovr_poly_3 = svc_ovr_poly.fit(x3_tr, y3_tr)
pred_3=svc_ovr_poly_3.predict(x3_v)
print ('Accuracy Score for 3rd validation set:')
print(accuracy_score(y3_v, pred_3))

print("the best one seems to be the second model")

pred3=svc_ovr_poly_2.predict(xte)
print ('Accuracy Score of the test set:')
accuracy = accuracy_score(yte, pred3)
print(accuracy)
print("")
conf_matrix = confusion_matrix(yte, pred3) 
print ('Confusion Matrix :')
print (conf_matrix)
print("")
print ('Report : ')
print (classification_report(yte, pred3) )
print("")
print("Dual Coffecient :")
print(svc_ovr_poly_2.dual_coef_)
print("")
print("Value of Parameters :")
print(svc_ovr_poly_2.get_params)

y_score = svc_ovr_poly_2.decision_function(xte)

fpr_1_3, tpr_1_3, _ = roc_curve(yte, y_score)
roc_auc_1_3 = auc(fpr_1_3, tpr_1_3)

plt.figure()

plt.plot(fpr_1_3, tpr_1_3, color='teal',lw=2, label='ROC curve for ckd (area = %0.2f)' % roc_auc_1_3)

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')    
plt.xlim([-0.1, 1.1])
plt.ylim([-0.1, 1.1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.show()

#for linear kernel SVM

from sklearn.svm import SVC

svc_ovr_linear = SVC(kernel='linear',decision_function_shape='ovr')

svc_ovr_linear_1 = svc_ovr_linear.fit(x1_tr, y1_tr)
pred_1=svc_ovr_linear_1.predict(x1_v)
print ('Accuracy Score for 1st validation set:')
print(accuracy_score(y1_v, pred_1))

svc_ovr_linear_2 = svc_ovr_linear.fit(x2_tr, y2_tr)
pred_2=svc_ovr_linear_2.predict(x2_v)
print ('Accuracy Score for 2nd validation set:')
print(accuracy_score(y2_v, pred_2))

svc_ovr_linear_3 = svc_ovr_linear.fit(x3_tr, y3_tr)
pred_3=svc_ovr_linear_3.predict(x3_v)
print ('Accuracy Score for 3rd validation set:')
print(accuracy_score(y3_v, pred_3))

print("the best one seems to be the first model")

pred4=svc_ovr_linear_1.predict(xte)
print ('Accuracy Score of the test set:')
accuracy = accuracy_score(yte, pred4)
print(accuracy)
print("")
conf_matrix = confusion_matrix(yte, pred4) 
print ('Confusion Matrix :')
print (conf_matrix)
print("")
print ('Report : ')
print (classification_report(yte, pred4) )
print("")
print("Dual Coffecient :")
print(svc_ovr_linear_1.dual_coef_)
print("")
print("Value of Parameters :")
print(svc_ovr_linear_1.get_params)

y_score = svc_ovr_linear_1.decision_function(xte)

fpr_1_4, tpr_1_4, _ = roc_curve(yte, y_score)
roc_auc_1_4 = auc(fpr_1_4, tpr_1_4)

plt.figure()

plt.plot(fpr_1_4, tpr_1_4, color='teal',lw=2, label='ROC curve for ckd (area = %0.2f)' % roc_auc_1_4)

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')    
plt.xlim([-0.1, 1.1])
plt.ylim([-0.1, 1.1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.show()

#rbf kernel SVM

from sklearn.svm import SVC

svc_ovr_rbf = SVC(kernel='rbf',decision_function_shape='ovr')

svc_ovr_rbf_1 = svc_ovr_rbf.fit(x1_tr, y1_tr)
pred_1=svc_ovr_rbf_1.predict(x1_v)
print ('Accuracy Score for 1st validation set:')
print(accuracy_score(y1_v, pred_1))

svc_ovr_rbf_2 = svc_ovr_rbf.fit(x2_tr, y2_tr)
pred_2=svc_ovr_rbf_2.predict(x2_v)
print ('Accuracy Score for 2nd validation set:')
print(accuracy_score(y2_v, pred_2))

svc_ovr_rbf_3 = svc_ovr_rbf.fit(x3_tr, y3_tr)
pred_3=svc_ovr_rbf_3.predict(x3_v)
print ('Accuracy Score for 3rd validation set:')
print(accuracy_score(y3_v, pred_3))

print("the best one seems to be the first model")

pred5=svc_ovr_rbf_1.predict(xte)
print ('Accuracy Score of the test set:')
accuracy = accuracy_score(yte, pred5)
print(accuracy)
print("")
conf_matrix = confusion_matrix(yte, pred5) 
print ('Confusion Matrix :')
print (conf_matrix)
print("")
print ('Report : ')
print (classification_report(yte, pred5) )
print("")
print("Dual Coffecient :")
print(svc_ovr_rbf_1.dual_coef_)
print("")
print("Value of Parameters :")
print(svc_ovr_rbf_1.get_params)

y_score = svc_ovr_rbf_1.decision_function(xte)

fpr_1_5, tpr_1_5, _ = roc_curve(yte, y_score)
roc_auc_1_5 = auc(fpr_1_5, tpr_1_5)

plt.figure()

plt.plot(fpr_1_5, tpr_1_5, color='teal',lw=2, label='ROC curve for ckd (area = %0.2f)' % roc_auc_1_5)

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')    
plt.xlim([-0.1, 1.1])
plt.ylim([-0.1, 1.1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.show()

# Naive Bayes

from sklearn.naive_bayes import GaussianNB

clf_bayes = GaussianNB()

clf_bayes_1 = clf_bayes.fit(x1_tr, y1_tr)
pred_1=clf_bayes_1.predict(x1_v)
print ('Accuracy Score for 1st validation set:')
print(accuracy_score(y1_v, pred_1))

clf_bayes_2 = clf_bayes.fit(x2_tr, y2_tr)
pred_2=clf_bayes_2.predict(x2_v)
print ('Accuracy Score for 2nd validation set:')
print(accuracy_score(y2_v, pred_2))

clf_bayes_3 = clf_bayes.fit(x3_tr, y3_tr)
pred_3=clf_bayes_3.predict(x3_v)
print ('Accuracy Score for 3rd validation set:')
print(accuracy_score(y3_v, pred_3))


print("the best one seems to be the second model")

pred6=clf_bayes_2.predict(xte)
print ('Accuracy Score of the test set:')
accuracy = accuracy_score(yte, pred6)
print(accuracy)
print("")
conf_matrix = confusion_matrix(yte, pred6) 
print ('Confusion Matrix :')
print (conf_matrix)
print("")
print ('Report : ')
print (classification_report(yte, pred6) )
print("")
print("Value of Parameters :")
print(clf_bayes_2.get_params)


NB_probs = clf_bayes_2.predict_proba(xte)

NB_probs = NB_probs[:, 1]
fpr_1_6, tpr_1_6, _ = roc_curve(yte, NB_probs)
roc_auc_1_6 = auc(fpr_1_6, tpr_1_6)

plt.figure()
plt.plot(fpr_1_6, tpr_1_6,color='teal',lw=2, label='ROC curve for ckd (area = %0.2f)' % roc_auc_1_6)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')    
plt.xlim([-0.1, 1.1])
plt.ylim([-0.1, 1.1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.show()
#  for Decision Tree


from sklearn.tree import DecisionTreeClassifier

decision_tree = DecisionTreeClassifier()

decision_tree_1 = decision_tree.fit(x1_tr, y1_tr)
pred_1=decision_tree_1.predict(x1_v)
print ('Accuracy Score for 1st validation set:')
print(accuracy_score(y1_v, pred_1))

decision_tree_2 = decision_tree.fit(x2_tr, y2_tr)
pred_2=decision_tree_2.predict(x2_v)
print ('Accuracy Score for 2nd validation set:')
print(accuracy_score(y2_v, pred_2))

decision_tree_3 = decision_tree.fit(x3_tr, y3_tr)
pred_3=decision_tree_3.predict(x3_v)
print ('Accuracy Score for 3rd validation set:')
print(accuracy_score(y3_v, pred_3))


print("the best one seems to be the first model")

pred7=decision_tree_1.predict(xte)
print ('Accuracy Score of the test set:')
accuracy = accuracy_score(yte, pred7)
print(accuracy)
print("")
conf_matrix = confusion_matrix(yte, pred7) 
print ('Confusion Matrix :')
print (conf_matrix)
print("")
print ('Report : ')
print (classification_report(yte, pred7) )
print("")
print("Value of Parameters :")
print(decision_tree_1.get_params)


DCT_probs = decision_tree_1.predict_proba(xte)

DCT_probs = DCT_probs[:, 1]
fpr_1_7, tpr_1_7, _ = roc_curve(yte, DCT_probs)
roc_auc_1_7 = auc(fpr_1_7, tpr_1_7)

plt.figure()
plt.plot(fpr_1_7, tpr_1_7,color='teal',lw=2, label='ROC curve for ckd (area = %0.2f)' % roc_auc_1_7)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')    
plt.xlim([-0.1, 1.1])
plt.ylim([-0.1, 1.1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.show()

#  for KNN


from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
KNN = KNeighborsClassifier(n_neighbors=17)

KNN_1 = KNN.fit(x1_tr, y1_tr)
pred_1=KNN_1.predict(x1_v)
print ('Accuracy Score for 1st validation set:')
print(accuracy_score(y1_v, pred_1))

KNN_2 = KNN.fit(x2_tr, y2_tr)
pred_2=KNN_2.predict(x2_v)
print ('Accuracy Score for 2nd validation set:')
print(accuracy_score(y2_v, pred_2))

KNN_3 = KNN.fit(x3_tr, y3_tr)
pred_3=KNN_3.predict(x3_v)
print ('Accuracy Score for 3rd validation set:')
print(accuracy_score(y3_v, pred_3))


print("the best one seems to be the first model")

pred8=KNN_2.predict(xte)
print ('Accuracy Score of the test set:')
accuracy = accuracy_score(yte, pred8)
print(accuracy)
print("")
conf_matrix = confusion_matrix(yte, pred8) 
print ('Confusion Matrix :')
print (conf_matrix)
print("")
print ('Report : ')
print (classification_report(yte, pred8) )
print("")
print("Value of Parameters :")
print(KNN_1.get_params)


knn_probs = KNN_1.predict_proba(xte)

knn_probs = knn_probs[:, 1]
fpr_1_8, tpr_1_8, _ = roc_curve(yte, knn_probs)
roc_auc_1_8 = auc(fpr_1_8, tpr_1_8)

plt.figure()
plt.plot(fpr_1_8, tpr_1_8,color='teal',lw=2, label='ROC curve for ckd (area = %0.2f)' % roc_auc_1_8)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')    
plt.xlim([-0.1, 1.1])
plt.ylim([-0.1, 1.1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.show()

#  for neural network - MLPClassifier


from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(hidden_layer_sizes=(10,5), random_state=1)

mlp_1 = mlp.fit(x1_tr, y1_tr)
pred_1=mlp_1.predict(x1_v)
print ('Accuracy Score for 1st validation set:')
print(accuracy_score(y1_v, pred_1))

mlp_2 = mlp.fit(x2_tr, y2_tr)
pred_2=mlp_2.predict(x2_v)
print ('Accuracy Score for 2nd validation set:')
print(accuracy_score(y2_v, pred_2))

mlp_3 = mlp.fit(x3_tr, y3_tr)
pred_3=mlp_3.predict(x3_v)
print ('Accuracy Score for 3rd validation set:')
print(accuracy_score(y3_v, pred_3))


print("the best one seems to be the first model")

pred9=mlp_1.predict(xte)
print ('Accuracy Score of the test set:')
accuracy = accuracy_score(yte, pred9)
print(accuracy)
print("")
conf_matrix = confusion_matrix(yte, pred9) 
print ('Confusion Matrix :')
print (conf_matrix)
print("")
print ('Report : ')
print (classification_report(yte, pred9) )
print("")
print("Value of Parameters :")
print(mlp12.get_params)


y_score = mlp_1.predict_proba(xte)

fpr_1_9, tpr_1_9, _ = roc_curve(yte, y_score[:, 1])
roc_auc_1_9 = auc(fpr_1_9, tpr_1_9)

plt.figure()

plt.plot(fpr_1_9, tpr_1_9, color='teal',lw=2, label='ROC curve for ckd (area = %0.2f)' % roc_auc_1_9)

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')    
plt.xlim([-0.1, 1.1])
plt.ylim([-0.1, 1.1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.show()


#Plotting all the ROC curves together
plt.figure(figsize=(12, 6))

plt.plot(fpr_1_1, tpr_1_1, color='red',lw=9, label='Random Forest ROC curve for ckd (area = %0.2f)' % roc_auc_1_1)
plt.plot(fpr_1_2, tpr_1_2, color='green',lw=4, label='Logistic Regression ROC curve for ckd (area = %0.2f)' % roc_auc_1_2)
plt.plot(fpr_1_3, tpr_1_3, color='blue',lw=4, label='SVM (2degree poly) ROC curve for ckd (area = %0.2f)' % roc_auc_1_3)
plt.plot(fpr_1_4, tpr_1_4, color='yellow',lw=4, label='SVM (linear) ROC curve for ckd (area = %0.2f)' % roc_auc_1_4)
plt.plot(fpr_1_5, tpr_1_5, color='pink',lw=4, label='SVM (rbf) ROC curve for ckd (area = %0.2f)' % roc_auc_1_5)
plt.plot(fpr_1_8, tpr_1_8, color='purple',lw=4, label='KNN ROC curve for ckd (area = %0.2f)' % roc_auc_1_8)
plt.plot(fpr_1_9, tpr_1_9, color='orange',lw=4, label='MLPClassifier ROC curve for ckd (area = %0.2f)' % roc_auc_1_9)
plt.plot(fpr_1_6, tpr_1_6, color='cyan',lw=5, label='Naive Bayes ROC curve for ckd (area = %0.2f)' % roc_auc_1_6)
plt.plot(fpr_1_7, tpr_1_7, color='teal',lw=2, label='Decision Tree ROC curve for ckd (area = %0.2f)' % roc_auc_1_7)

plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')    
plt.xlim([-0.1, 1.1])
plt.ylim([-0.1, 1.1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.show()

#TensorFlow Neural Network Algo

# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import matplotlib.pyplot as plt

e_=[]
hl=[10,5]

x1=tf.placeholder("float",[None,35])
y1=tf.placeholder("float",[None,1])

w=[tf.Variable(tf.random_normal([35,hl[0]])),tf.Variable(tf.random_normal([hl[0],hl[1]])),
   tf.Variable(tf.random_normal([hl[1],1]))]

b=[tf.Variable(tf.random_normal([hl[0]])),tf.Variable(tf.random_normal([hl[1]])),tf.Variable(tf.random_normal([1]))]

def forward(x1):
    l1=tf.add(tf.matmul(x1,w[0]),b[0])
    l1=tf.nn.relu(l1)
    l2=tf.add(tf.matmul(l1,w[1]),b[1])
    l2=tf.nn.relu(l2)
    ol=tf.add(tf.matmul(l2,w[2]),b[2])
    return ol

loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=forward(x1), labels=y1) +\
            0.0001*(tf.reduce_sum(tf.square(b[0]))+tf.reduce_sum(tf.square(b[1]))+tf.reduce_sum(tf.square(b[2]))))

opt=tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    error=[]
    for ep in range(200):
        _,llo=sess.run([opt, loss],feed_dict={x1:x1_tr,y1:y1_tr.reshape((186,1))})
        error.append(llo)   
    pred=tf.nn.softmax(forward(x1))
    corr=tf.equal(tf.argmax(pred, 1), tf.argmax(y1, 1))
    accu=tf.reduce_mean(tf.cast(corr, "float"))
    a_,_=sess.run([accu,corr],feed_dict={x1:x1_v,y1:y1_v.reshape((94,1))})
    print(a_)
    e_.append(error)
    error=[]
    sess.run(tf.global_variables_initializer())
    for ep in range(200):
        _,llo=sess.run([opt, loss],feed_dict={x1:x2_tr,y1:y2_tr.reshape((187,1))})
        error.append(llo)   
    pred=tf.nn.softmax(forward(x1))
    corr=tf.equal(tf.argmax(pred, 1), tf.argmax(y1, 1))
    accu=tf.reduce_mean(tf.cast(corr, "float"))
    a_,_=sess.run([accu,corr],feed_dict={x1:x2_v,y1:y2_v.reshape((93,1))})
    print(a_)
    e_.append(error)
    error=[]
    sess.run(tf.global_variables_initializer())
    for ep in range(200):
        _,llo=sess.run([opt, loss],feed_dict={x1:x3_tr,y1:y3_tr.reshape((187,1))})
        error.append(llo)   
    pred=tf.nn.softmax(forward(x1))
    corr=tf.equal(tf.argmax(pred, 1), tf.argmax(y1, 1))
    accu=tf.reduce_mean(tf.cast(corr, "float"))
    a_,_=sess.run([accu,corr],feed_dict={x1:x3_v,y1:y3_v.reshape((93,1))})
    print(a_)
    e_.append(error)
    plt.plot(range(1,201),e_[0])
    plt.plot(range(1,201),e_[1])
    plt.plot(range(1,201),e_[2])
    plt.xlabel("iterations")
    plt.ylabel("error")
    plt.show()

    error=[]
    sess.run(tf.global_variables_initializer())
    for ep in range(200):
        _,llo=sess.run([opt, loss],feed_dict={x1:xtr,y1:ytr.reshape((280,1))})
        error.append(llo)   
    pred=tf.nn.softmax(forward(x1))
    corr=tf.equal(tf.argmax(pred, 1), tf.argmax(y1, 1))
    accu=tf.reduce_mean(tf.cast(corr, "float"))
    a_,_=sess.run([accu,corr],feed_dict={x1:xte,y1:yte.to_numpy().reshape((120,1))})
    print(a_)
    plt.plot(range(1,201),error)
    plt.xlabel("iterations")
    plt.ylabel("error")
    plt.show()
 
