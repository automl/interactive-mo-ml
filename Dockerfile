FROM ghcr.io/josephgiovanelli/mo-importance:0.0.6

RUN apt-get update && \
    apt-get install -y git --no-install-recommends
RUN pip install --upgrade pip && \
    pip install "git+https://github.com/slds-lmu/yahpo_gym#egg=yahpo_gym&subdirectory=yahpo_gym"
RUN wget -c https://github.com/slds-lmu/yahpo_data/archive/refs/tags/v1.0.zip && \
    unzip v1.0.zip && \
    rm -rf v1.0.zip

RUN mkdir dump
WORKDIR /home/dump
COPY . .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

RUN mkdir input
RUN mkdir output
RUN mkdir logs
RUN mkdir plots

RUN chmod 777 scripts/*
ENTRYPOINT ["./scripts/wrapper_experiments.sh"]
