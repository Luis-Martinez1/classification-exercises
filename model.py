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
