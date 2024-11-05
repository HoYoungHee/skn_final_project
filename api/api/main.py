# main.py 파일
from fastapi import FastAPI
from routers.db_router import router as db_router
from routers.interview_router import router as interview_router
from fastapi.middleware.cors import CORSMiddleware
from models.create_tables import create_tables_if_not_exists
import asyncio
app = FastAPI()
# Django 서버의 주소를 origins에 추가합니다.
origins = [
    "http://192.168.56.1:8000",  # Django 서버 주소
    "http://127.0.0.1:8000",
    # 필요에 따라 추가 도메인을 허용할 수 있습니다.
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,        # 또는 ["*"]로 모든 도메인 허용 (개발 단계에서만 권장)
    allow_credentials=True,
    allow_methods=["*"],          # 모든 HTTP 메서드 허용
    allow_headers=["*"],          # 모든 HTTP 헤더 허용
)

# 라우터 등록
app.include_router(db_router, prefix="/db", tags=["database"])
app.include_router(interview_router, prefix="/interview", tags=["interview"])

# 앱 시작 시 테이블 생성
def startup_event():
    create_tables_if_not_exists()

app.add_event_handler("startup", lambda: create_tables_if_not_exists())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=43355, reload=True)

@app.get("/hello/")
async def say_hello():
    # 5초 대기 (비동기적으로)
    await asyncio.sleep(5)
    # 5초 후에 메시지 반환
    return {"message": "안녕하세요?"}