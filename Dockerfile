# Python 3.10을 기반 이미지로 사용
FROM python:3.10-slim

# 작업 디렉토리 설정
WORKDIR /app

# 1) 필요한 패키지 설치
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 2) VertexAI GAPIC 의존성 설치
RUN pip install --no-cache-dir google-cloud-aiplatform

# 3) langchain-google-vertexai 설치 (의존성 해제)
RUN pip install --no-cache-dir langchain-google-vertexai --no-deps

# 소스 코드 복사
COPY . .

ENV PYTHONPATH=/app/app

# Cloud Run에서는 8080 포트를 사용해야 함
EXPOSE 8080

# FastAPI 앱 실행
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]