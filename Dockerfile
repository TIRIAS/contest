FROM python:3.10

# 필수 패키지 설치
RUN apt-get update && apt-get install -y \
    git cmake libboost-dev libboost-system-dev libboost-filesystem-dev \
    build-essential

# 작업 디렉토리
WORKDIR /app

# requirements.txt 먼저 복사하고 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 프로젝트 파일 복사
COPY . .

CMD ["python", "run.py"]
