FROM ubuntubase4:16.04
COPY . /app   
WORKDIR /app
#RUN pip install -r requirements.txt
CMD ["/usr/bin/python3","./medium1.py"]