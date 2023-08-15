import os

UPLOAD_FOLDER = "uploads/"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

REDIS_QUEUE = "redis_q"
REDIS_PORT = 6379
REDIS_DB_ID = 0
REDIS_IP = "redis"
SERVER_SLEEP = 0.05
