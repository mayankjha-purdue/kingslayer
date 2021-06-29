FROM python:3

COPY requirements.txt . 

RUN pip install --no-cache-dir -r requirements.txt

COPY src/game.py /src/game.py

COPY src/ai.py /src/ai.py

COPY src/board.py /src/board.py

COPY src/gui.py /src/gui.py

COPY src/table.py /src/table.py

COPY src/table.py /src/table.py

COPY /data .



CMD ["python3", "/app/game.py"]
