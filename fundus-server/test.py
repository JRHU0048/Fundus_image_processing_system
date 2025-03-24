import pymysql

user = "nopwd_user"
print(f"尝试使用用户 {user} 连接数据库")
try:
    db = pymysql.connect(
        host="localhost",
        user=user,
        # password="mysql123",
        database="eyesystem",
        charset='utf8mb4',
    )
    print("数据库连接成功")
except pymysql.Error as e:
    print(f"数据库连接失败: {e}")