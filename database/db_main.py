from sqlalchemy import Column, BigInteger, DateTime, Integer, Text, Float, Boolean, Time, Date, JSON
from sqlalchemy import MetaData, Table, select
from sqlalchemy import create_engine
from sqlalchemy.engine.url import URL
import sqlalchemy


__author__ = 'wfuller'

metadata = MetaData()

CONNECTION_INFO_DICT = {
    'drivername': 'postgresql',
    'username': 'dbuser_test',
    'password': 'sqluser_test',
    'database': 'motion_db',
    'host': 'ussjc-psql01.invcorp.invensense.com',
    'port': '5432',
}


'''
main table containing metadata, and select measurements
'''
sweeptest = Table('sweeptest', metadata,
                  Column('test_id', BigInteger, primary_key=True, autoincrement=True),
                  Column('timestamp', DateTime),
                  Column('test_name', Text),
                  Column('test_type', Text),
                  Column('tag', Text),
                  Column('comment', Text),
                  Column('operator', Text),
                  )

'''
table containing sample number used for joining, positional data, and amplitude data for quick plotting of beam patterns
'''
datapoint = Table('datapoint', metadata,
                  Column('test_id', Integer),
                  Column('sample_number', BigInteger, primary_key=True, autoincrement=True),
                  Column('angle', Float),
                  Column('radius', Float),
                  Column('amplitude', Float),
                  Column('rotation', Float),
                  )

'''
table containing join information
'''
rel_table = Table('rel_table', metadata,
                  Column('rel_id', Integer, primary_key=True, autoincrement=True),
                  Column('join_column', Text),
                  Column('parent', Text),
                  Column('child', Text)
                  )


class SweepTestDB:
    def __init__(self, url=URL(**CONNECTION_INFO_DICT)):
        self.engine = self.get_engine(url)
        metadata.reflect()  # get existing tables
        self.tables = metadata.tables
        self._conn = None
        if __name__ == "__main__":
            self.create_tables()

    @property
    def conn(self):
        if self._conn is None:
            self._conn = self.engine.connect()
            return self._conn
        else:
            return self._conn

    def _new_table_relation(self, join_col, parent_table, child_table):
        ins = rel_table.insert().values(join_column=join_col, parent=parent_table, child=child_table)
        self.conn.execute(ins)

    def build_new_table(self, tablename, cols, primary_key_col, join_col, parent_table):
        """
        builds a table object to be returned and added to the DB using metadata()
        :param tablename: lowercase name of the table to be added to the db
        :param cols: 'column_name': 'datatype' of column in the new table
        :param primary_key_col: primary key column of the new table
        :param join_col: the column that links this table to its parent
        :param parent_table: the table that is the "parent" of this table
        :return: the table object built using sql alchemy
        """
        dtypes = {
            'bool': Boolean,
            'bigint': BigInteger,
            'int': Integer,
            'str': Text,
            'float': Float,
            'json': JSON,
            'date': Date,
            'time': Time,
            'datetime': DateTime,
        }

        columns = []
        for col in cols:
            if col == primary_key_col:
                columns.append(Column(col, dtypes[cols[col]]))
            else:
                columns.append(Column(col, dtypes[cols[col]]))
        table = Table(tablename, metadata, *columns)
        self._new_table_relation(join_col, parent_table, tablename)
        self.create_tables()
        return table

    @staticmethod
    def get_engine(url, echo=False):
        engine = create_engine(url, echo=echo)
        metadata.bind = engine
        return engine

    def create_tables(self):
        try:
            metadata.create_all(self.conn)
        except sqlalchemy.exc.InvalidRequestError as e:
            print("table already exists")

    def __del__(self):
        try:
            self._conn.close()
        except:
            pass
        super()


if __name__ == '__main__':
    import os
    # path = f"C:\\users\\{os.getlogin()}\\documents"

    db = SweepTestDB()  # f'sqlite:///C:\\users\\{os.getlogin()}\\documents\\test_sweep_stage.db')
    # db.build_new_table(
    #     'buildtabletest',
    #     {'sample_number': 'bigint', 'frequency': 'float', 'bandwidth': 'float', 'amplitude': 'float', 'range': 'float',
    #      'encoded_iq': 'str'},
    #     'sample_number',
    #     'sample_number',
    #     'datapoint'
    #
