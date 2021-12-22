x_train = X_train[cols_selected_boruta]
x_test =  x_test[cols_selected_boruta]

aux1 = x_test.copy()
aux1['sales'] = y_test.copy()

#predição

aux2 = aux1[['store','sales']].groupby('store').mean().reset_idnex().rename(columns = {'sales' : 'predictions'})
aux1 = pd.merge(aux1, aux2, how='left', on='store')
yhat_baseline = aux1['predictions']

#performance
baseline_result = ml_error("Average Model", np.expm1(y_test), np.expm1(yhat_baseline))
baseline_result

#modelo
lr = LinearRegression().fit(x_train,y_train)

#predição
yhat_lr = lr.predict(x_test)

#performance
lr_result = ml_error('Linear Regression',np.expm1(y_test),np.expm1(yhat_lr))
lr_result

#modelo
lrr = Lasso(alpha=0.01).fit(x_train, y_train)

#prediction
yhat_lrr = lrr.predict(x_test)

#performance
lrr_result = ml_error('Linear Regression - Lasso', np.expm1(y_test), np.expm1(yhat_lrr))
lrr_result