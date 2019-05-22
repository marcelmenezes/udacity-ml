from formatoModelo import getDataFrameModelo
from sklearn.externals import joblib
est_RandomForest = joblib.load('modelo_RandomForest_1milhao.pkl')
est_SVC = joblib.load('modelo_SVC.pkl')

def getSexoRandomForest(nome:str):
    df = getDataFrameModelo(nome)
    #return 'M' if reg.predict(df)[0] == 1 else 'F'
    est = est_RandomForest
    proba = est.predict_proba(df)
    prob = {'probability to be Female (F)': proba[0][0], 'probability to be Male': proba[0][1]}
    return 'M' if est.predict(df)[0] == 1 else 'F', prob

    #return 'M'
    # print("predict: ", nome)
    # print("predict: ", reg.predict(df))
    # print("proba: ", reg.predict_proba(df))



# def getSexo(nome:str):
#     df = getDataFrameModelo(nome)
#     #return 'M' if reg.predict(df)[0] == 1 else 'F'
#     est = est_SVC
#     return 'M' if est.predict(df)[0] == 1 else 'F'



def getSexoSVC(nome:str):
    df = getDataFrameModelo(nome)
    #return 'M' if reg.predict(df)[0] == 1 else 'F'
    est = est_SVC
    proba = est.predict_proba(df)
    prob = {'probability to be Female': proba[0][0], 'probability to be Male': proba[0][1]}
    return 'M' if est.predict(df)[0] == 1 else 'F', prob


