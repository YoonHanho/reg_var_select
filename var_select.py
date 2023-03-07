# 회귀분석 변수선택 파이썬 코드 : ADP 파이썬 교재 
# https://github.com/ADPclass/ADP_book_ver01/ >> 7장 5절 회귀분석 
# https://bit.ly/3CqoRoY
# 교재에 forward, backward 추가 코드 누락
# 제대로된 설명 : https://todayisbetterthanyesterday.tistory.com/10


import pandas as pd
import numpy as np

import statsmodels.api as sm 
import statsmodels.formula.api as smf 

import time
import itertools


def processSubset(X,y, feature_set):
            model = sm.OLS(y,X[list(feature_set)]) # Modeling
            regr = model.fit() # 모델 학습
            AIC = regr.aic # 모델의 AIC
            return {"model":regr, "AIC":AIC}
# 전진선택법
def forward(X, y, predictors):
    # 데이터 변수들이 미리정의된 predictors에 있는지 없는지 확인 및 분류
    remaining_predictors = [p for p in X.columns.difference(['Intercept']) if p not in predictors]
    results = []
    for p in remaining_predictors:
        results.append(processSubset(X=X, y= y,            feature_set=predictors+[p]+['Intercept']))
        
    # 데이터프레임으로 변환
    models = pd.DataFrame(results)
    # AIC가 가장 낮은 것을 선택
    best_model = models.loc[models['AIC'].argmin()] # index
    print("Processed ", models.shape[0], "models on", len(predictors)+1, "predictors in")
    print('Selected predictors:',best_model['model'].model.exog_names,' AIC:',best_model[0] )
    return best_model	


def forward_model(X,y):

    Fmodels = pd.DataFrame(columns=["AIC","model"])
    tic = time.time()
    
    # 미리 정의된 데이터 변수
    predictors = []
    
    # 변수 1~10개 : 0-9 -> 1-10
    for i in range(1,len(X,columns.difference(['const']))+1):
    	Forward_result = forward(X=X,y=y,predictors=predictors)
    	if i > 1:
            if Forward_result["AIC"] > Fmodel_before:
                break
    	Fmodels.loc[i] = Forward_result
    	predictors = Fmodels.loc[i]["model"].model.exog_names
    	Fmodel_before = Fmodels.loc[i]["AIC"]
    	predictors = [k for k in predictors if k != 'const']
    toc = time.time()
    print("Total elapsed time:",(toc-tic), "seconds.")
    
    return (Fmodels['model'][len(Fmodels['model'])])
    

	
# 후진소거법
def backward(X,y,predictors):
    tic = time.time()
    results = []
    
    # 데이터 변수들이 미리정의된 predictors 조합 확인
    for combo in itertools.combinations(predictors, len(predictors) -1):
        results.append(processSubset(X=X, y= y,        feature_set = list(combo)+['Intercept']))
    models = pd.DataFrame(results)
    
    # 가장 낮은 AIC를 가진 모델을 선택
    best_model = models.loc[models['AIC'].argmin()]
    toc = time.time()
    print("Processed ", models.shape[0], "models on",          len(predictors) -1, "predictors in", (toc - tic))
    print('Selected predictors:',best_model['model'].model.exog_names,         'AIC:',best_model[0] )

    return best_model	


def backward_model(X,y) :
    Bmodels = pd.DataFrame(columns=["AIC","model"], index = range(1,len(X.columns)))
    tic = time.time()
    predictors = X.columns.difference(['const'])
    Bmodel_before = processSubset(X,y,predictors)['AIC']
    while (len(predictors) > 1):
    	Backward_result = backward(X=train_x, y= train_y, predictors=predictors)
    	if Backward_result['AIC'] > Bmodel_before :
        	break
    	Bmodels.loc[len(predictors) -1] = Backward_result
    	predictors = Bmodel.loc[len(predictors) - 1]['model'].model.exog_names
    	Bmodel_before = Backward_result["AIC"]
    	predictors = [k for k in predictors if k != 'const']
    
    toc = time.time()
    print("Total elapsed time:",(toc-tic),"seconds.")
    return (Bmodels["model"].dropna().iloc[0])



# 단계적 선택법
def stepwise_model(X,y):
    Stepmodels = pd.DataFrame(columns=["AIC", "model"])
    tic = time.time()
    predictors = []
    Smodel_before = processSubset(X,y,predictors+['Intercept'])['AIC']

    for i in range(1, len(X.columns.difference(['Intercept'])) +1):
        Forward_result = forward(X=X, y=y, predictors=predictors) 
        print('forward')
        Stepmodels.loc[i] = Forward_result
        predictors = Stepmodels.loc[i]["model"].model.exog_names
        predictors = [ k for k in predictors if k !='Intercept']
        Backward_result = backward(X=X, y=y, predictors=predictors)

        if Backward_result['AIC']< Forward_result['AIC']:
            Stepmodels.loc[i] = Backward_result
            predictors = Stepmodels.loc[i]["model"].model.exog_names
            Smodel_before = Stepmodels.loc[i]["AIC"]
            predictors = [ k for k in predictors if k !='Intercept']
            print('backward')

        if Stepmodels.loc[i]['AIC']> Smodel_before:
            break
        else:
            Smodel_before = Stepmodels.loc[i]["AIC"]
    toc = time.time()
    print("Total elapsed time:", (toc - tic), "seconds.")

    return (Stepmodels['model'][len(Stepmodels['model'])])	


##############################################################################
import pandas as pd
import statsmodels.api as sm

#data=pd.read_csv('ex9-6.csv')
#data.head(4)
#y=data.Y
#X=data.iloc[:,2:6]

def forward(X, y, level, verbose=False): #전진선택법
    initial_list=[]
    included=list(initial_list)    #선택된 변수를 저장할 리스트
    while True:
        changed=False
        excluded=list(set(X.columns)-set(included))     #(전체변수-선택된 변수)=남은변수 저장
        pval=pd.Series(index=excluded, dtype='float64')  ## 변수의 p-value 저장

        for col in excluded:
            if  (len(included)==0):
                model=sm.OLS(y, sm.add_constant(pd.DataFrame(X[included+[col]]))).fit()
            else:
                model=sm.OLS(y, pd.DataFrame(X[included+[col]])).fit()
            pval[col]=model.pvalues[col]
        best_pval=pval.min()

        if best_pval < level:  #유의수준과 p-value를 비교해서 작으면 해당 변수를 모형에 포함
            best_X=pval.idxmin()
            included.append(best_X)
            changed=True

            if verbose:
                print('ADD{:20} with p-val{:25}'.format(best_X, best_pval))
        if not changed:
            break
    return included      #최종 선택 변수 출력

forward(X, y, 0.05, verbose=True)    #데이터의 반응변수와 데이터 이름 입력

def backward(X, y, level, verbose=False): #전진선택법
    included=list(X.columns)    #선택된 변수를 저장할 리스트
    while True:
        changed=False
        if  (len(included)==1):
            model=sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
        else:
            model=sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
        pval=model.pvalues.iloc[1:]
        worst_pval=pval.max()

        if worst_pval > level:  #유의수준과 p-value를 비교해서 작으면 해당 변수를 모형에 포함
            changed = True
            worst_X=pval.idxmax()
            included.remove(worst_X)

            if verbose:
                print('DROP{:20} with p-val{:25}'.format(worst_X, worst_pval))
        if not changed:
            break
    return included      #최종 선택 변수 출력

backward(X, y, 0.05, verbose=True)    #데이터의 반응변수와 데이터 이름 입력
