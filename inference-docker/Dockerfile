FROM 763104351884.dkr.ecr.us-west-2.amazonaws.com/djl-inference:0.21.0-deepspeed0.8.3-cu117
RUN apt update && apt install python3-tk -y && pip install transformers==4.30.2 && pip install accelerate==0.20.3 && pip install git+https://github.com/thunlp/OpenDelta.git
