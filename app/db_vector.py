import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# AlloyDB 연결 정보 (하드코딩 또는 .env 사용)
ALLOYDB_URL = os.getenv("ALLOYDB_URL", "postgresql://postgres:1q2w3e4r5t@10.20.0.2:5432/postgres")

# SQLAlchemy 엔진 생성
alloydb_engine = create_engine(
    ALLOYDB_URL,
    connect_args={"options": "-c search_path=public"}
)

# 세션 로컬 생성기
AlloySessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=alloydb_engine)

# FastAPI 의존성 주입용
def get_alloydb():
    db = AlloySessionLocal()
    try:
        yield db
    finally:
        db.close()
