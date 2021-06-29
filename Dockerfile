FROM python:3

COPY requirements.txt . 

RUN pip install --no-cache-dir -r requirements.txt

COPY src/game.py /src/game.py
RUN true


COPY src/ai.py /src/ai.py
RUN true


COPY src/board.py /src/board.py
RUN true


COPY src/gui.py /src/gui.py
RUN true


COPY src/table.py /src/table.py
RUN true


COPY src/table.py /src/table.py
RUN true


COPY data/opening.bin /data/opening.bin
RUN true


COPY data/cache.p /data/cache.p
RUN true

export DISPLAY=127.0.0.1:0.0

docker run -it -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix yourImage

CMD ["python3", "/src/game.py"]
