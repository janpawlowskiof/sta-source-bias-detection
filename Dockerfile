FROM python:3.10

RUN python3 -m pip install --upgrade pip
RUN pip install torch==2.0.1+cpu torchvision==0.15.2+cpu torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cpu

ADD ./requirements.txt /source-detection/requirements.txt

RUN pip install -r /source-detection/requirements.txt
RUN python -m spacy download xx_ent_wiki_sm

ADD app/ /source-detection/

RUN mkdir /source-detection/models

CMD python -m ...