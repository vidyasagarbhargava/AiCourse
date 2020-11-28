FROM python:latest
WORKDIR /app

COPY . __main__.py
RUN pip install flask
ENTRYPOINT [ "python3", "__main__.py" ]



# COPY . __main__.py
# # COPY . ./requirements.txt
# RUN pip install flask
# ENTRYPOINT python3 /app/__main__.py
# # ENTRYPOINT ["tail", "-f", "/dev/null"]
