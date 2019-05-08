def preprocess_train_eval(df_in, 
                          LRC=LogisticRegression(solver="lbfgs", random_state=0), 
                          RFC=RandomForestClassifier(class_weight="balanced", n_estimators=100, random_state=0), 
                          
                          XG_params = {"random_state":0}, 
                          xgb_fit_eval_metric="aucpr",

                          printing=False, printing_ap=False, returning=True, split_random_state=42,
                          printing_rs=False,

                         train_test_split_random_state=42,
                         RandomOverSampler_random_state=42,
                         test_size=0.33,
                         preprocessing_function=pre_ml_process
                         ):
    
    df_in, X, y, X_train, X_test, y_train, y_test, \
                scaler, X_train_resample_scaled, y_train_resample, \
                X_test_scaled, ros, poly_transformer = \
                preprocessing_function(df_in, 
                               test_size,
                               train_test_split_random_state,
                               RandomOverSampler_random_state)

    outputs = {}
    
    # Train with Logistic Regression
    if LRC != None:
        clf_LR = LRC.fit(X_train_resample_scaled, y_train_resample)
        outputs_LR = evaluate_model(clf_LR, X_train_resample_scaled, X_test_scaled, y_train, y_test, 
                                    df_in, scaler, ros, poly_transformer, printing)
        if printing_ap:
            print("LR ap = {:.3f}".format(outputs_LR["ap"]))
        if printing_rs:
            print("LR rs = {:.3f}".format(outputs_LR["rs"]))
        outputs["LR"] = outputs_LR
    
    # Train with Random Forest Classifier
    if RFC != None:
        clf_RF = RFC.fit(X_train_resample_scaled, y_train_resample)
        outputs_RF = evaluate_model(clf_RF, X_train_resample_scaled, X_test_scaled, y_train, y_test, 
                                    df_in, scaler, ros, poly_transformer, printing)
        if printing_ap:
            print("RF ap = {:.3f}".format(outputs_RF["ap"]))
        if printing_rs:
            print("RF rs = {:.3f}".format(outputs_RF["rs"]))
        outputs["RF"] = outputs_RF

    
    # Train with XGBoost Classifier
    if XG_params != None:
        clf_XG = xgb.XGBClassifier()
        clf_XG.set_params(**XG_params)
        
        clf_XG.fit(X_train_resample_scaled, y_train_resample, eval_metric=xgb_fit_eval_metric)
        outputs_XG = evaluate_model(clf_XG, X_train_resample_scaled, X_test_scaled, y_train, y_test, 
                                    df_in, scaler, ros, poly_transformer, printing)
        if printing_ap:
            print("XG ap = {:.3f}".format(outputs_XG["ap"]))
        if printing_rs:
            print("XG rs = {:.3f}".format(outputs_XG["rs"]))
        outputs["XG"] = outputs_XG   
    
    if returning:
        return outputs
