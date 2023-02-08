import getpass
import os
import psycopg2
from psycopg2 import sql
import pandas as pd
import itertools

from configobj import ConfigObj
from tqdm import tqdm
from typing import Tuple


# __all__ = ['connect_to_database',
#            'get_id_list',
#            'get_database_table_column_name',
#            'get_database_table_as_dataframe']


def connect_to_database() -> Tuple[str, str, str, psycopg2.extensions.connection]:
    """Connect to the SQL database. """

    # # Create a database connection using settings from config file
    # config = os.path.join(db_path, 'config.ini')
    #
    # # connection info
    conn_info = dict()

    conn_info["sqluser"] = 'postgres'
    conn_info["sqlpass"] = 'postgres'
    conn_info["sqlhost"] = 'localhost'
    conn_info["dbname"] = 'mimiciv'
    conn_info["schema_name_core"] = 'public,mimic_core'
    conn_info["schema_name_hosp"] = 'public,mimic_hosp'
    conn_info["schema_name_icu"] = 'public,mimic_icu'
    conn_info["schema_name_derived"] = 'public,mimic_derived'

    # Connect to the eICU database
    print('Database: {}'.format(conn_info['dbname']))
    print('Username: {}'.format(conn_info["sqluser"]))

    conn = psycopg2.connect(host=conn_info['sqlhost'], dbname=conn_info["dbname"],
                            user=conn_info["sqluser"], password=conn_info['sqlpass'])


    def _f(x):
        return 'set search_path to ' + x + ';'

    query_schema_core = _f(conn_info['schema_name_core'])
    query_schema_hosp = _f(conn_info['schema_name_hosp'])
    query_schema_icu = _f(conn_info['schema_name_icu'])
    query_schema_derived = _f(conn_info['schema_name_derived'])

    print(">>>>> Connected to DB <<<<<")

    return (query_schema_core, query_schema_hosp, query_schema_icu,
            query_schema_derived, conn)


def query_sql(_conn:psycopg2.extensions.connection,
                _query_schema: str,
                _query: str,):
    data = None
    # try:
    query = _query_schema + f"""{_query}"""
    # _cursor = _conn.cursor()
    # _cursor.execute(query)
    # data = _cursor.fetchall()
    data = pd.read_sql_query(query, _conn)
    # finally:
    #     if _conn:
    #         # _cursor.close()
    #         _conn.close()
    #         print("PostgreSQL connection is closed")

    return data