import pymysql

username = "root"
password = "123456"
dbName = "healthy39"
charset = "utf8"
host = "localhost"


def getConnection():
    """
    TODO 使用数据库连接池
    :return:
    """
    conn = pymysql.connect(host=host, user=username, password=password,
                           database=dbName, charset=charset)
    return conn


def getQueryRes(sql, params=None):
    """
    返回查询结果
    :param sql:
    :param params:
    :return:
    """
    conn = getConnection()
    cursor = conn.cursor()
    cursor.execute(sql, params)
    res = cursor.fetchall()
    cursor.close()
    conn.close()
    return res
