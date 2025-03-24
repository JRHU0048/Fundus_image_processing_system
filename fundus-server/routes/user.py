
from utils.db import SessionLocal
from sqlalchemy.orm import Session
from fastapi import Depends,  HTTPException, APIRouter,Form
from models import User
from pydantic import BaseModel
from fastapi import Body

user_router = APIRouter()

# 
class UserLogin(BaseModel):
    username: str
    password: str

class UserRegister(BaseModel):
    username: str
    password: str
    gender: str

class UserEdit(BaseModel):
    username: str
    gender: str
    id: str
    mark: str
    avatarUrl: str

class UserEditWeb(BaseModel):
    username: str
    gender: str
    id: str

class UserEditPassword(BaseModel):
    password: str
    id: str
# 

# Dependency
def get_db():
    try:
        db = SessionLocal()
        yield db
    finally:
        db.close()

# @user_router.post("/login")
# def user_login(username: str, password: str, db: Session = Depends(get_db)):
#     print("username",username)
#     db_user = User.get_user_byname(db,username)
#     if db_user is None:
#         return {"msg": "用户名不存在",
#                 "state": "fail"}
#     if db_user.password != password:
#         return {"msg": "密码错误",
#                 "state": "fail"}
#     user = {}
#     user['id'] = db_user.id
#     user['username'] = db_user.username
#     user['gender'] = db_user.gender
#     user['mark'] = db_user.mark
#     user['avatarUrl'] = db_user.avatarUrl
#     return user

@user_router.post("/login", response_model=dict)
def user_login(user: UserLogin, db: Session = Depends(get_db)):
    print("username", user.username)
    db_user = User.get_user_byname(db, user.username)
    if db_user is None:
        return {"msg": "用户名不存在", "state": "fail"}
    if db_user.password != user.password:
        return {"msg": "密码错误", "state": "fail"}
    return {
        "id": db_user.id,
        "username": db_user.username,
        "gender": db_user.gender,
        "mark": db_user.mark,
        "avatarUrl": db_user.avatarUrl,
        "msg": "登录成功",
        "state": "success"
    }


# @user_router.post("/register", response_model=User.BaseUser)
# def user_register(username: str, password: str,gender:str ,db: Session = Depends(get_db)):
#     db_user = User.add_user(db, username,password,gender)
#     return db_user

@user_router.post("/register", response_model=dict)
def user_register(user: UserRegister, db: Session = Depends(get_db)):
    db_user = User.add_user(db, user.username, user.password, user.gender)
    return {"msg": "注册成功", "state": "success"}

# @user_router.get("/editUser")
# def user_edit(username:str,gender:str,id:str ,mark:str,avatarUrl:str,db:Session = Depends(get_db)):
#     db_user = User.get_user_byaccount(db,id)
#     db_user.username = username
#     db_user.gender = gender
#     db_user.mark = mark
#     db_user.avatarUrl = avatarUrl
#     db.commit()
#     return {
#         "msg":"ok",
#         "user":User.get_user_byaccount(db,id)
#     }

@user_router.post("/editUser", response_model=dict)
def user_edit(user: UserEdit, db: Session = Depends(get_db)):
    db_user = User.get_user_byaccount(db, user.id)
    db_user.username = user.username
    db_user.gender = user.gender
    db_user.mark = user.mark
    db_user.avatarUrl = user.avatarUrl
    db.commit()
    return {
        "msg": "ok",
        "user": User.get_user_byaccount(db, user.id)
    }

# @user_router.get("/editUserWeb")
# def user_edit(username:str,gender:str,id:str ,db:Session = Depends(get_db)):
#     db_user = User.get_user_byaccount(db,id)
#     db_user.username = username
#     db_user.gender = gender
#     db.commit()
#     return {
#         "msg":"ok",
#         "user":User.get_user_byaccount(db,id)
#     }

@user_router.post("/editUserWeb", response_model=dict)
def user_edit_web(user: UserEditWeb, db: Session = Depends(get_db)):
    db_user = User.get_user_byaccount(db, user.id)
    db_user.username = user.username
    db_user.gender = user.gender
    db.commit()
    return {
        "msg": "ok",
        "user": User.get_user_byaccount(db, user.id)
    }

# @user_router.get("/editPassword")
# def user_edit(password:str,id:str ,db:Session = Depends(get_db)):
#     db_user = User.get_user_byaccount(db,id)
#     db_user.password = password
#     db.commit()
#     return {
#         "msg":"ok",
#         "user":User.get_user_byaccount(db,id)
#     }

@user_router.post("/editPassword", response_model=dict)
def user_edit_password(password_data: UserEditPassword, db: Session = Depends(get_db)):
    db_user = User.get_user_byaccount(db, password_data.id)
    db_user.password = password_data.password
    db.commit()
    return {
        "msg": "ok",
        "user": User.get_user_byaccount(db, password_data.id)
    }

# @user_router.get("/checkuser")
# def user_check(username:str,db:Session = Depends(get_db)):
#     db_user = User.get_user_byname(db,username)
#     if db_user is None:
#         return {
#             "msg": "ok",
#             "state": "ok"
#         }
#     else:
#         return {
#             "msg": "用户名已存在",
#             "state": "fail"
#         }

@user_router.get("/checkuser")
def user_check(username: str, db: Session = Depends(get_db)):
    db_user = User.get_user_byname(db, username)
    if db_user is None:
        return {"msg": "ok", "state": "ok"}
    else:
        return {"msg": "用户名已存在", "state": "fail"}

@user_router.get("/test")
def user_test():
    return {"msg": "ok"}
