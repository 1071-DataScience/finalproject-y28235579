import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import cross_validation, ensemble, preprocessing, metrics
from sklearn.cross_validation  import train_test_split , KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import sys


args = sys.argv
data_file_name = args[1]
output_file_name = args[2]
folds = args[3]

feature = ["order_date_month","people_amount","days","begin_date_month","price","product_name_price_min","cp","price-min","is-min-price","num_same_group","total_people_amount","discount","fly_count" , "order_date_dayofweek" , "begin_date_dayofweek" , "order_date_isweekend" , "begin_date_isweekend" , "src1_value_1","src1_value_2","src1_value_3","src2_value_1","src2_value_2","src2_value_3","src2_value_4"]
data = pd.read_csv(data_file_name)
target = data["deal_or_not"]
source = data[feature]

#將資料切割
source_train , source_test , target_train , target_test = train_test_split(source , target , test_size = 0.3 , random_state = 1)
deal , not_deal = 0 , 0
for x in data["deal_or_not"]:
    if x == 1:
        deal = deal + 1
    if x == 0:
        not_deal = not_deal + 1
print("null_accuracy:" + str(not_deal / (deal + not_deal)))

test = [0] * 146267
fpr, tpr, thresholds = metrics.roc_curve(data["deal_or_not"] , test)
null_auc = metrics.auc(fpr, tpr)
print("null_auc:" + str(null_auc))



# model採用RandomForest
forest = ensemble.RandomForestClassifier(n_estimators = 300 , random_state=0 , n_jobs=7)
forest_fit = forest.fit(source_train, target_train)
test_y_predicted = forest.predict(source_train)

fpr, tpr, thresholds = metrics.roc_curve(target_train, test_y_predicted)
auc = metrics.auc(fpr, tpr)
print("train_auc" + str(auc))



# cross-validation
Kfold = KFold(len(source_train) , n_folds = int(folds) , shuffle = False)
scores = cross_val_score(forest , source_train , target_train , cv = Kfold , scoring='accuracy')
print("train_accuracy" + str(scores.mean()))


#predict test_data
test_pred = forest.predict_proba(source_test)


test = []
for value in test_pred[:,1]:
    if value > 0.5 :
        test.append(1)
    else:
        test.append(0)
# detail value
accuracy = metrics.accuracy_score(target_test , test)
F1 = metrics.f1_score(target_test, test, average='weighted') 
print("test_accuracy:" + str(accuracy))
cm = confusion_matrix(target_test , test)
print(cm)
sensitivity = cm[0,0]/(cm[0,0]+cm[1,0])
specificity = cm[1,1]/(cm[0,1]+cm[1,1])
precision = cm[0,0]/(cm[0,0]+cm[1,0])
recall = cm[0,0]/(cm[0,0]+cm[0,1])
print("cf_sensitivity:" + str(sensitivity))
print("cf_specificity:" + str(specificity))
print("cf_precision:" + str(precision))
print("cf_recall:" + str(recall))
# df = pd.DataFrame(cm)
# df.to_csv("confusion_matrix.csv" , index = False)


#test-data AUC
fpr, tpr, thresholds = metrics.roc_curve(target_test , test_pred[:,1])
test_auc = metrics.auc(fpr, tpr)
print("test_auc:" + str(test_auc))


# deal with output_file
output_dict = {}
row_names = ["null_model_auc" ,"accuracy" , "test_auc" , "precision" , "recall" , "F1" , "sensitivity" , "specificity"]
score = [null_auc , accuracy , test_auc, precision , recall , F1 , sensitivity , specificity]
output_dict = {
    "item" : row_names,
    "value" : score
}
pd.DataFrame(output_dict).to_csv(output_file_name , index = False)

#plot
lw = 2


# plot confusion matrix
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Reds)
classNames = ['Negative','Positive']
plt.title('travel deal_or_not')
plt.ylabel('True label')
plt.xlabel('Predicted label')
s = [['TN','FP'], ['FN', 'TP']]
for i in range(2):
    for j in range(2):
        plt.text(j,i, str(s[i][j])+" = "+str(cm[i][j]))
plt.savefig('./confusion_matrix.png')






# plot ROC
plt.figure(figsize=(10,10))
plt.plot(fpr, tpr, color='darkorange',lw=lw, label='ROC curve (area = %0.2f)' % auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('travel deal or not')
plt.legend(loc="lower right")
plt.savefig('./auc_result.png')
plt.show()



# precision , recall , f1 score , Sensitivity , Specificity
# args
