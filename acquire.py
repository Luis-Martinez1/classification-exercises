from env import get_db_url
import pandas as pd
import os
import numpy as np


def get_telco_data():
    """
    takes in no argument, will query the telco database 
    returns the telco query as a pandas dataframe
    """
    filename = "telco.csv"
    if os.path.isfile(filename):
        df = pd.read_csv(filename)
    else:
        query = """
        select *
        from customers
        left join contract_types
            using (contract_type_id)
        left join internet_service_types
            using (internet_service_type_id)
        left join payment_types
            using (payment_type_id)
        """
        connection = get_db_url("telco_churn")
        df = pd.read_sql(query, connection)
        df.to_csv(filename, index=False)
    return df


def get_iris_data():
    """
    takes in no argument and
    returns iris database query as a pandas dataframe
    """
    filename = "iris.csv"
    if os.path.isfile(filename):
        df = pd.read_csv(filename)
    else:
        query = """
        SELECT *
        FROM measurements
        JOIN species
        USING (species_id);"""
        connection = get_db_url("iris_db")
        df = pd.read_sql(query, connection)
        df.to_csv(filename, index=False)
    return df


def get_titanic_data():
    """
    takes in no arguments and 
    returns the titanic database query a pandas dataframe
    """
    filename = "titanic.csv"
    if os.path.isfile(filename):
        df = pd.read_csv(filename)
    else:
        query = "SELECT * FROM passengers"
        connection = get_db_url("titanic_db")
        df = pd.read_sql(query, connection)
        df.to_csv(filename, index=False)
    return df


