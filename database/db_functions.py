import sqlalchemy
import sqlalchemy.exc
import datetime
import database.db_main as db
import pandas as pd

'''
this is a class of functions to allow easy script calls
functions needed in this file:
1. get_summary(test_id)->pd.DataFrame
2. get_summary_csv(test_id, path)
3. get_test(test_id)->pd.DataFrame
4. get_test_csv(test_id, path)
5. start_test(args)
6. write_datapoint(args)
?7. write to table
'''


def get_next_id(conn: sqlalchemy.engine.Connection, table: str, id_col: str):
    """
    return the last id_col + 1 of table
    :param id_col: column in database containing ID
    :param table: table name in database
    :param conn: connection tp database
    :return: value of last test_id in db_main.sweeptest
    """
    try:
        res = conn.execute(f"SELECT {id_col} FROM {table} ORDER BY {id_col} DESC LIMIT 1").fetchone()[id_col]
    except TypeError:
        return 0
    else:
        print(res)
        return res + 1


def get_test(
        conn: sqlalchemy.engine.Connection,
        test_id: int,
) -> pd.DataFrame:
    """
    retrieves a sweep test across all tables, and joins based on db_main.rel_table into a single dataframe
    :param conn: connection to the database
    :param test_id: the sweep test id to retrieve
    :return: pandas dataframe containing the joined test
    """
    tables = []

    # get the test type
    res = conn.execute(f"SELECT test_type FROM sweeptest WHERE test_id = {test_id}").fetchone()
    print(res)
    # use the test type to get the tables required
    res = conn.execute(f"SELECT * FROM rel_table WHERE child = '{res[0]}'").fetchone()
    tables.append(res)
    while 1:
        print(res)
        if res['parent'] == 'datapoint':
            break
        res = conn.execute(f"SELECT * FROM rel_table WHERE child = '{res['parent']}'").fetchone()
        tables.append(res)

    # inner join the tables based on rel_table
    lines = [f"SELECT * FROM sweeptest INNER JOIN datapoint ON sweeptest.test_id = datapoint.test_id", ]
    for idx, join_col, parent, child in reversed(tables):
        lines.append(f"INNER JOIN {child} ON {parent}.{join_col} = {child}.{join_col}")
    lines.append(f"AND sweeptest.test_id = {test_id}")

    # load into a dataframe and return
    this_sql = ' '.join(lines)
    return pd.read_sql(this_sql, conn)


def get_test_csv(
        conn: sqlalchemy.engine.Connection,
        test_id: int,
        path: str,
):
    """
    retrieves a sweep test across all tables, and joins based on db_main.rel_table into a single csv file

    :param conn: connection to the database
    :param test_id: the sweep test id to retrieve
    :param path: the file path to save the csv, including file name
    :return: none
    """
    test_df = get_test(conn, test_id)

    # Code added by Kyle to add test name to filename
    test_name = test_df['test_name'][0]
    test_tag = test_df['tag'][0]
    new_path = f'./{test_name}, {test_tag} (t{test_id}).csv'

    # get_test(conn, test_id).to_csv(new_path)
    test_df.to_csv(new_path)


def get_hydra_measurement(
        conn: sqlalchemy.engine.Connection,
        test_sample: int,
) -> pd.DataFrame:
    """
    retrieves a sweep test across all tables, and joins based on db_main.rel_table into a single dataframe

    :param conn: connection to the database
    :param test_sample: the sweep test id to retrieve
    :return: pandas dataframe containing the joined test
    """
    # WHERE pe_hydra_measurement.sample_number = {test_sample}
    sample = f"SELECT * FROM pe_hydra_measurement WHERE sample_number = {test_sample}"

    return pd.read_sql(sample, conn)


def get_hydra_test(
        conn: sqlalchemy.engine.Connection,
        test_sample: int,
) -> pd.DataFrame:
    """
    retrieves a sweep test across all tables, and joins based on db_main.rel_table into a single dataframe

    :param conn: connection to the database
    :param test_sample: the sweep test id to retrieve
    :return: pandas dataframe containing the joined test
    """
    # WHERE pe_hydra_measurement.sample_number = {test_sample}
    sample = f"SELECT * FROM pe_hydra_test WHERE sample_number = {test_sample}"

    return pd.read_sql(sample, conn)


def get_summary(
        conn: sqlalchemy.engine.Connection,
        test_id: int,
) -> pd.DataFrame:
    """
    retrieves a sweeptest from db_main.sweeptest and db_main.datapoint, and joins on test_id
    :param conn: connection to the database
    :param test_id: the sweep test id to retrieve
    :return: pandas dataframe containing the test summary
    """
    return pd.read_sql(
        f"SELECT * FROM datapoint d INNER JOIN sweeptest s ON d.test_id = s.test_id AND d.test_id = {test_id}",
        conn
    )


def get_summary_csv(
        conn: sqlalchemy.engine.Connection,
        test_id: int,
        path: str
):
    """
    retrieves a sweeptest from db_main.sweeptest and db_main.datapoint, and joins on test_id
    :param conn: connection to the database
    :param test_id: the sweep test id to retrieve
    :param path: the file path to save the csv, including file name
    :return: none
    """
    get_summary(conn, test_id).to_csv(path)


def start_test(
        conn: sqlalchemy.engine.Connection,
        test_name: str,
        test_type: str,
        tag: str,
        comment: str,
        operator: str,
):
    """
    writes the test information to the sweeptest table
    :param conn: connection to the database
    :param test_name: name of the test, input by user
    :param test_type: type of test providing db relations, selected by user
    :param tag: test tag, selectable and creatable by user
    :param comment: test comment, input by user
    :param operator: operator, input and selectable by user
    :return: test_id
    """
    ins = db.sweeptest.insert().values(
        test_id=get_next_id(conn, 'sweeptest', 'test_id'),
        timestamp=datetime.datetime.now(),
        test_name=test_name,
        test_type=test_type,
        tag=tag,
        comment=comment,
        operator=operator
    )
    result = conn.execute(ins)
    return result.inserted_primary_key[0]


def write_datapoint(
        conn: sqlalchemy.engine.Connection,
        test_id: int,
        angle: float,
        radius: float,
        amplitude: float,
        *,
        rotation: float = None,
) -> int:
    """
    writes a datapoint to the datapoint table
    :param conn: connection to the database
    :param test_id: test id, same as in the sweeptest table
    :param angle: the current angle in degrees
    :param radius: the current linear distance in mm
    :param amplitude: the amplitude at this datapoint
    :param rotation: the rotation along the z-axis (typically only in a 3D beampattern with the MECA500)
    :return: sample number
    """
    ins = db.datapoint.insert().values(
        sample_number=get_next_id(conn, 'datapoint', 'sample_number'),
        test_id=test_id,
        angle=angle,
        radius=radius,
        amplitude=amplitude,
        rotation=rotation
    )
    result = conn.execute(ins)
    return result.inserted_primary_key[0]


def write_to_table(
        conn: sqlalchemy.engine.Connection,
        table: sqlalchemy.Table or str,
        **kwargs
):
    """
    writes data to a table
    :param conn: connection to the database
    :param table: table or name of the table to write to
    :param kwargs: column -> value to write to table_name
    :return: None
    """
    if isinstance(table, str):
        values = []
        for val in kwargs.values():
            if isinstance(val, str):
                values.append(f"'{val}'")
            elif val is None:
                values.append('null')
            else:
                values.append(str(val))
        ins = f"INSERT INTO {table} ({','.join(kwargs.keys())})" \
              f"VALUES ({','.join(values)})"
    else:
        ins = table.insert.values(**kwargs)
    conn.execute(ins)


if __name__ == "__main__":
    import os

    path = f"C:\\users\\{os.getlogin()}\\documents\\"
    test_db = db.SweepTestDB(url=f'sqlite:///{path}\\test_sweep_stage.db')
    write_test_data = False

    if write_test_data:
        import math

        id = start_test(test_db.conn, 'sql_test', 'buildtabletest', 'sql', 'check-get-test', 'wfuller')
        for sn in range(5):
            angle = 90 + sn * 45
            radius = 500
            freq = 177000
            bandwidth = 4500
            amplitude = -math.cos(math.radians(angle)) * 1000
            distance = 499.9
            encoded_iq = 'AAAA88292AHF'
            samp = write_datapoint(
                test_db.conn,
                id,
                angle,
                radius,
                amplitude,
            )
            write_to_table(
                test_db.conn,
                'buildtabletest',
                sample_number=samp,
                frequency=freq,
                bandwidth=bandwidth,
                amplitude=amplitude,
                range=distance,
                encoded_iq=encoded_iq,
            )

    print(get_summary(test_db.conn, 37))
    get_test_csv(test_db.conn, 37, path + '\\dbtestout.csv')
