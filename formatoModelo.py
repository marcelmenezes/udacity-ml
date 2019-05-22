import pandas as pd
from sound import *

def getDataFrameModelo(nome:str):
        
    nome = remove_accents(nome)
    pnome = primeiro_nome(nome)
    #print("'{0}'".format(pnome))

    df = pd.DataFrame({'PrimeiroNomeNoAccents': [pnome]})
    df.head()

    df['Sounde'] = df.apply(lambda row: soundex(row['PrimeiroNomeNoAccents']), axis=1)

    df['SoundexIndice0'] = df[df['Sounde'].notna()].apply(lambda row: ord(row['Sounde'][0:1]), axis=1)
    df['SoundexIndice1'] = df[df['Sounde'].notna()].apply(lambda row: int(row['Sounde'][1:2]), axis=1)
    df['SoundexIndice2'] = df[df['Sounde'].notna()].apply(lambda row: int(row['Sounde'][2:3]), axis=1)
    df['SoundexIndice3'] = df[df['Sounde'].notna()].apply(lambda row: int(row['Sounde'][3:4]), axis=1)
    df['Leng'] = df.apply(lambda row: len(row['PrimeiroNomeNoAccents']), axis=1)


    df.head()

    for i in range(0, 10):
        df['LetraIndice' + str(i)] = df.apply(lambda row: getascii(row, i), axis=1)
        
    df_Mean = pd.read_csv("mediaLetras.csv")
    df = df.fillna(df_Mean)


    drop = df.drop(['PrimeiroNomeNoAccents', 'Sounde'], axis=1)
    df = drop
    return df