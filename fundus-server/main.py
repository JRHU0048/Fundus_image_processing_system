from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware  # 导入CORSMiddleware
from routes.file import file_router
from routes.user import user_router
from routes.util import util_router
from fastapi.staticfiles import StaticFiles
import uvicorn

app = FastAPI()

# 配置CORS中间件
origins = [
    "*",  # 根据实际情况调整，推荐仅开放必要的源
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # 允许的源列表
    allow_credentials=True,  # 是否允许发送Cookie
    allow_methods=["*"],     # 允许的HTTP方法
    allow_headers=["*"],     # 允许的HTTP头部
)

# 挂载静态文件目录
app.mount("/assets", StaticFiles(directory="assets"), name="assets")
app.mount("/output", StaticFiles(directory="output"), name="output")

# 包含您的路由
app.include_router(user_router, prefix="/user")
app.include_router(file_router, prefix="/file")
app.include_router(util_router, prefix='/util')

if __name__ == "__main__":
    uvicorn.run(app='main:app', host='0.0.0.0', port=8080, reload=True)