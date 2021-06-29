FROM python:3

COPY requirements.txt . 

RUN pip install --no-cache-dir -r requirements.txt

COPY src/game.py /app/game.py

CMD ["python3", "/app/game.py"]
