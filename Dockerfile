COPY requirements.txt . 

RUN pip install -r requirements.txt 

COPY src/game.py /app/game.py

CMD ["python3", "/app/game.py"]
