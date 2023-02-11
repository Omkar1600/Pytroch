FROM python:3.8
WORKDIR /app
COPY . .
RUN pip install -r requirments.txt
EXPOSE 8501
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health
ENTRYPOINT ["streamlit", "run", "test.py", "--server.port=8501", "--server.address=0.0.0.0"]