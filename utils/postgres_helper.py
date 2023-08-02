#!/usr/bin/env python
# -*- coding=utf-8 -*-
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import os,sys, math
import os.path as path
from retrying import retry
# sys.path.append(path.dirname(path.dirname(path.dirname(path.dirname(path.abspath(__file__))))))
import dask



import data_service.get_config as commons_config


 
class PostgresHelper:

    def __init__(self, host=None, port=None, username=None, password=None, database=None):  # 构造函数
        try:
            connect_url = "postgresql+psycopg2://{username}:{password}@{host}:{port}/{database}".format(username = username, password = password, host = host, port = port, database = database)
            execution_options = {
                # "isolation_level": "REPEATABLE READ",
                "autocommit": True
            }
            self.__db_engine = create_engine(connect_url, execution_options=execution_options, pool_size=100, max_overflow=0, pool_recycle=28800)
        except Exception as e:
            print("初始化链接失败--------------------" + str(e))






    # 事务性操作时，用这个SQL
    # 返回执行execute()方法后影响的行数
    def execute(self, sql):
        """
            执行事务性操作，如：insert、delete、update
            参数：
                sql：传入的查询mysql的SQL语句
            返回值：返回受影响的行数
        """
        try:
            with self.__db_engine.connect() as conn:
                # trans = conn.begin()
                conn.execute(sql)           # 执行SQL语句
            # trans.commit()              # 提交事物
        except Exception as sql_e:
            # trans.rollback()            # 发生错误时回滚
            raise Exception("--------------------收集到的异常信息：------\n" + str(sql_e))





    # 判断数据库连接，是否失效
    def check_connect_alive(self):
        """
        检查数据库连接，是否存活：
            true：存活
            false：失效了
        """
        is_alive = False
        try:
            self.__db_engine.connect()
            is_alive = True
        except Exception as sql_e:
            is_alive = False

        return is_alive






    # 查所有数据
    @retry(stop_max_attempt_number=3, wait_fixed=1)     # 最多重试3次，重试等待间隔1s
    def select_all(self, sql):
        """
            返回SQL查出来的所有数据
            参数：
                sql：传入的查询mysql的SQL语句
            返回值：数据集数组
        """
        # 先判断引擎的初始化状态
        if not self.check_connect_alive():
            raise Exception("数据库连接，重试4次，仍然没有创建成功，请检查（网络、数据库）可用性：" + str(sql_e))

        try:
            with self.__db_engine.connect() as conn:
                conn.execute(sql)           # 执行SQL语句
                reslut_data = conn.fetchall()
                return reslut_data
        except Exception as sql_e:
            raise Exception("--------------------收集到的异常信息：------\n" + str(sql_e))






    def write_df_to_db(self, source_df, tb_name:str, df_schema:list, tb_schema:list, set_primary_key='begin_ts,end_ts,project_name,car_id'):

        assert source_df is not None, "传入的数据源df，必须有值，不能为None"
        assert tb_name is not None, "传入的tb_name，必须有值，不能为None"
        assert tb_name is not None, "传入的tb_name，必须有值，不能为None"
        assert df_schema is not None and len(df_schema)>0, "传入的df_schema，必须有值，不能为None"
        assert tb_schema is not None and len(tb_schema)>0, "传入的df_schema，必须有值，不能为None"

        # 批量插入，每2万条为一批，插入到数据库中
        batch_record_num = 0
        need_write_db_df = source_df[df_schema]
        total_num = len(need_write_db_df)
        assert total_num>0, "传入的数据源df，数据条数，不能=0"
        need_write_data_list = np.array(need_write_db_df).tolist()
        write_data = ''
        execute_sql_list = []
        for curr_row in need_write_data_list:
            batch_record_num = batch_record_num + 1
            total_num = total_num - 1
            write_data = write_data + ",(" + str(curr_row).strip('[|]') + ")"

            # 判断是否需要写入，25001为一批
            if batch_record_num > 25000 or (total_num<1 and batch_record_num>0):       # 25001为一批，写入数据库
                insert_batch_data_sql = """
                set hg_experimental_enable_fixed_dispatcher_for_multi_values = on;
                set hg_experimental_affect_row_multiple_times_keep_last = on;
                INSERT INTO {tb_name} ({tb_schema})
                VALUES
                {write_data}
                ON CONFLICT ({set_primary_key}) DO UPDATE SET ({tb_schema}) = ROW (excluded.*);
                """.format(tb_name=tb_name, tb_schema=str(tb_schema).strip('[|]').replace("'", '"'), write_data = write_data.strip(','), set_primary_key = set_primary_key)
                execute_insert_sql = dask.delayed(self.execute)(insert_batch_data_sql)
                execute_sql_list.append(execute_insert_sql)
                # self.execute(insert_batch_data_sql)
                batch_record_num = 0
                write_data = ''
        # 最后执行并行写入计算
        # 动态批次写入（每批执行并行度不超过10个）
        # 动态原理： 均等平分
        print("每批写入2.5万条，写入并行度：" + str(len(execute_sql_list)))
        execute_batch_num = math.ceil(len(execute_sql_list)/8)
        execute_step_num = math.ceil(len(execute_sql_list)/execute_batch_num)
        for pos_index in range(execute_batch_num):
            start_pos = pos_index * execute_step_num
            end_pos = (pos_index + 1) * execute_step_num
            execute_result = dask.delayed(lambda x: x)(execute_sql_list[start_pos: end_pos])
            execute_result.compute()






    # 把通过SQL查出来的数据，转成pandas的dataFrane对象
    @retry(stop_max_attempt_number=3, wait_fixed=1)     # 最多重试3次，重试等待间隔1s
    def read_db_to_df(self, sql=""):
        """
            把通过SQL查出来的数据，转成pandas的dataFrane对象
            参数：
                sql：传入的查询mysql的SQL语句
            返回值：df_data
                df_data：返回的数据
        """
        try:
            df_data = pd.read_sql(sql=sql, con = self.__db_engine)
            return df_data
        except Exception as e_e:
            raise Exception("--------------------收集到的异常信息：------\n" + str(e_e))











##############################################################
#     doris、mysql连接实例
##############################################################
def get_postgres_helper(database=""):

    postgres_db_config = commons_config.get_config(env='prod', conf_name='application.conf')
    db_host = postgres_db_config.get('postgres_db', 'db_host')
    db_port = postgres_db_config.getint('postgres_db', 'db_port')
    db_username = postgres_db_config.get('postgres_db', 'db_username')
    db_password = postgres_db_config.get('postgres_db', 'db_password')

    return PostgresHelper(
        host=db_host,
        port=db_port,
        username=db_username,
        password=db_password,
        database=database
    )









