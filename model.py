import pandas as pd

def preprocess_titanic(df):
    '''
     preprocess_titanic will take in the pandas dataframe
    of our titanic data, expected as cleaned versions of this 
    titanic data set (see documentation on acquire.py and prepare.py)
    
    output:
    encoded version of our clean dataframe ready to be split into train, validate, and test, 
    columns sex and embark_town are now encoded using the one-hot method.
    return: (df). 
    
    
    '''
    #  Encoding categorical columns for original dataframe
    dummy_df = pd.get_dummies(df[['sex','embark_town']], dummy_na=False, drop_first=[True, True]).astype(int)
    # Concatenate the dummy_df  with df dataframe
    df = pd.concat([df,dummy_df], axis=1)
    # Drop string values that have been replaced with encoded values
    df = df.drop(columns=['sex','embark_town'])
    

    return df









def preprocess_telco(df):
    '''
    preprocess_telco will take in a pandas dataframe of our telco data, 
    expected to already be a newly acquired and preped copy of the  telco 
    data set (see documentation on acquire.py and prepare.py)
    
    output:
    encoded, ML-ready version of our clean data, with 
    object type columns encoded in the one-hot fashion. The dataframe is
    now ready to be split into train, validate, and test.
    returns: pandas DataFrame.
    '''
     
    encoding_vars = []
    # loop through the columns to fill encoded_vars with appropriate datatype field names
    for col in df.columns:
        if df[col].dtype == 'O':
            encoding_vars.append(col)
    # encode our list of columns, concatenate columns, and drop columns
    df_encoded_cats = pd.get_dummies(df[encoding_vars], drop_first=True).astype(int)
    df = pd.concat([df,df_encoded_cats],axis=1).drop(columns=encoding_vars)
    return df

