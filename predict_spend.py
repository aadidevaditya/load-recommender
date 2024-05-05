import pandas as pd
import numpy as np
from h2o.automl import H2OAutoML
import h2o

def preprocessing(data):
  bins_age = [18,25,35,45,100]
  labels_age = ['18-25','25-35','35-45','>45']
  # data['age'] = pd.to_numeric(data['age'], errors='coerce')
  data['age'] = pd.cut(data['age'], bins=bins_age, labels=labels_age, right=False)
  bins_income = [0,50000,100000,400000,500000,1000000,float('inf')]
  labels_income = ['0-50k','50k-100k','100k-400k','400k-500k','500k-1000k','>1000k']
  # data['annual_income'] = pd.to_numeric(data['annual_income'], errors='coerce')
  data['annual_income'] = pd.cut(data['annual_income'], bins=bins_income, labels=labels_income, right=False)
  others=pd.DataFrame(data.country_code.value_counts()[data.country_code.value_counts()<100]).index.to_list()
  data['country_code']=data.country_code.apply(lambda x: 'others' if x in others else x )
  data['occupation'] = data['occupation'].apply(lambda x: 'Student' if x == 'Student' else 'Non-Student')
  data = data.drop(['user_count','pincode','channel'],axis = 1)
  return data

def modelling(data):
  h2o.init()
  train_df = h2o.H2OFrame(data)
  test = h2o.H2OFrame(data)
  x = test.columns
  y = 'txn_amount_tot'
  x.remove(y)
  # callh20automl  function
  aml = H2OAutoML(max_runtime_secs = 120, max_models = 10)
  # train model and record time % time
  aml.train(x = x, y = y, training_frame = train_df)
  lb = aml.leaderboard
  model_name = lb.as_data_frame()['model_id'][0]
  model = h2o.get_model(model_name)
  y_true = train_df['txn_amount_tot']
  y_pred = model.predict(train_df)
  test['y_pred'] = y_pred
  test['Error'] = abs((test['txn_amount_tot'] - test['y_pred']))/test['txn_amount_tot']
  final_df = test.as_data_frame().sort_values(by = ['Error'],ascending = True).reset_index(drop=True)
  return final_df

def calculateR2(final_df):
  y_true = final_df['txn_amount_tot'].to_list()
  y_pred = final_df['y_pred'].to_list()
  sse = 0
  sst = 0
  sse_mean = np.mean(y_true)
  for i in range(int(0.3*len(final_df))):
    sse = sse + ((y_pred[i] - y_true[i])**2)
    sst = sst + ((y_true[i] - sse_mean)**2)
  r2 = 1 - (sse/sst)
  return r2

def predictSpend(data,params):
  print("inside predict spend")
  params['channel'] = 'Default'
  params['user_count'] = 1
  params = pd.DataFrame([params])
  data = preprocessing(data)
  params_processed = preprocessing(params)
  final_df = modelling(data)
  final_df.to_csv('test_data.csv')
  predicted_value = 0
  error = 0
  error_flag = 100000000
  for i in range(len(final_df)):
    if params_processed['country_code'][0] == final_df['country_code'][i] and params_processed['age'][0] == final_df['age'][i] and params_processed['aadhar_gender'][0] == final_df['aadhar_gender'][i] and params_processed['occupation'][0] == final_df['occupation'][i] and params_processed['annual_income'][0] == final_df['annual_income'][i] and params_processed['marital_status'][0] == final_df['marital_status'][i] and params_processed['networth'][0] == final_df['networth'][i] and params_processed['source_of_funds'][0] == final_df['source_of_funds'][i]:
      if final_df['Error'][i] < error_flag:
        error_flag = final_df['Error'][i]
        predicted_value = final_df['y_pred'][i]
        error = final_df['Error'][i]
  r2 = calculateR2(final_df)

  return {'r2':r2, 'final_data':predicted_value,'Error':error}
