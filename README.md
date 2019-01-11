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
You should provide an example commend to reproduce your result
```R
Rscript code/your_script.R --input data/training --output results/performance.tsv
```
* any on-line visualization

## Folder organization and its related information

### docs
* 1071_datascience_FP_<107753022|deal or not>.ppt
* Any related document for the final project
  * papers
  * software user guide

### data

* T-brain and Lion Travel prvided the data.
* Input format is csv  
        * airline.csv      
        * group.csv    
        * order.csv    
        * training-set.csv    
        
* there is no missing value, but we try so hard to find useful featues among a lot of details in data. 

### code

* We use random forest to predict.
* we found that most of data are not deal in the end, and it's nearly 80%, so we use the null
  model which always predict not deal.
* Because we don't have test data's correct output, so we cut the train data into another train and test data,with cross-validation,to
  evaluate performance.

### results

* Primarily,we use AUC as our metrics.
* the original model's AUC was not good enough, but when we include source column as a feature to our model, AUC increase almost 10% !
* The most difficult part is to find what is good feature to help predict, and which predict model will be good for this dataset. 
