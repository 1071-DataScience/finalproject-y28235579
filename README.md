# Deal or not !
![image](https://github.com/1071-DataScience/finalproject-y28235579/blob/master/docs/deal%20or%20not.png)

### Groups
* < 蔡政憲, 107753034 >
* < 賴以立, 107753022>

### Goal
As a travel agency, the most difficult thing is to decide whether a order would be canceled or not in the end,so 
we want to prdict the order depend on its data about time,price,people amount and so on. It's a T-brain competition host by
Trend Micro, and data is provided by Lion Travel.

### demo 
* Prepare data    
```R
python ./prepare_data.py  order_file_name.csv group_file_name.csv airline_file_name.csv training-set.csv data.csv
```
* Run    
```R
python ./run.py data.csv output_file.csv [folds]
```     
## Folder organization and its related information

### docs
* 1071_datascience_FP_deal or not.ppt
* Any related document for the final project
  * papers
  * software user guide

### data

* T-brain host competition and  Lion Travel prvided the data.
![image](https://github.com/1071-DataScience/finalproject-y28235579/blob/master/docs/lion.png)
![image](https://github.com/1071-DataScience/finalproject-y28235579/blob/master/docs/tbrain.jpg)
* Input format is csv  
  * airline.csv      
  * group.csv    
  * order.csv  
  * cashe_map.csv
  * training-set.csv    
        
* there is no missing value, but we try so hard to find useful featues among a lot of details in data. 

### code
* include    
  * prepare_data.py    
  * run.py    
* We use random forest to predict.
* we found that most of data are not deal in the end, and it's nearly 80%, so we use the null
  model which always predict not deal.
* Because we don't have test data's correct output, so we cut the train data into another train and test data,with cross-validation,to
  evaluate performance.

### results

* Primarily,we use AUC as our metrics.
![image](https://github.com/1071-DataScience/finalproject-y28235579/blob/master/results/auc%20result.png)
* the original model's AUC was not good enough, but when we include source column as a feature to our model, AUC increase almost 10% !
* The most difficult part is to find what is good feature to help predict, and which predict model will be good for this dataset. 
