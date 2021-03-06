FROM python:3.8.6-buster

COPY model.joblib /model.joblib
COPY TaxiFareModel /TaxiFareModel
COPY api /api
COPY requirements.txt /requirements.txt

RUN pip install Cython --install-option="--no-cython-compile"
RUN pip install --upgrade pip && pip install -r requirements.txt

CMD uvicorn api.fast:app --host 0.0.0.0 --port $PORT
