import pandas as pd
import numpy as np





def prep_iris(df):
    '''takes in the iris dataframe and returns the dataframe
    with unnecessary columns dropped and the [species] column name changed for easier reading. 
    '''
    df.drop(columns=['species_id', 'measurement_id'], inplace=True)
    df.rename(columns={'species_name': 'species'}, inplace=True)
    return df





def clean_titanic(df):
    """
    Takes in the Titanic DataFrame and returns the Dataframe with unnecessary columns dropped, 
    casts [pclass] column to object since it will be handled as object, and
    fills null values for [embark_town] with the column mode.
    """
    df = df.drop(columns=['embarked', 'age','deck', 'class'])
    df.pclass = df.pclass.astype(object)
    df.embark_town = df.embark_town.fillna('Southampton')
    return df







def prep_telco(df):
    '''takes in the telco churn dataframe and returns the dataframe
    with unnecessary columns dropped and empty values in the [total_charges] column
    changed from a blank space to a zero.
    '''
    df.drop(columns=['payment_type_id', 'internet_service_type_id', 'contract_type_id'], inplace=True)
    df.total_charges = df.total_charges.str.replace(' ', '0.0')
    return df






def split_data(df, target):
    """
    takes in a DataFrame and target, returns a train, validate, and test DataFrame
    while stratifying on the target variable. Prints the percentage of the new dataframes 
    compared to the original.
    """
    train, validate_test = train_test_split(
        df, train_size=0.6, random_state=123, stratify=df[target]
    )
    validate, test = train_test_split(
        validate_test, train_size=0.5, random_state=123, stratify=validate_test[target]
    )
    print(f"train: {len(train)} ({round(len(train)/len(df), 2)*100}% of {len(df)})")
    print(
        f"validate: {len(validate)} ({round(len(validate)/len(df), 2)*100}% of {len(df)})"
    )
    print(f"test: {len(test)} ({round(len(test)/len(df), 2)*100}% of {len(df)})")

    return train, validate, test