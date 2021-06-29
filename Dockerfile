FROM python:3

COPY requirements.txt . 

RUN pip install --no-cache-dir -r requirements.txt

COPY src/game.py /app/game.py

COPY src/ai.py /app/ai.py

COPY src/board.py /app/board.py

COPY src/gui.py /app/gui.py

COPY src/table.py /app/table.py


CMD ["python3", "/app/game.py"]
