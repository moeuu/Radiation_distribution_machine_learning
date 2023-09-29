ARG PYTORCH="1.7.0"
ARG CUDA="11.0"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 -y

WORKDIR /project

COPY .requirements.txt /projext

RUN pip install -r requirements.txt

RUN jupyter notebook --generate-config

#jupyter-notebook上のホワイトリストを許可 
RUN echo "c.NotebookApp.ip = '0.0.0.0'" >> ~/.jupyter/jupyter_notebook_config.py

#初期ディレクトリの設定
RUN echo "c.NotebookApp.notebook_dir = '/project'" >> ~/.jupyter/jupyter_notebook_config.py

#パスワード認証 or Token認証を無効化
RUN echo "c.NotebookApp.token = ''" >> ~/.jupyter/jupyter_notebook_config.py
RUN echo "c.NotebookApp.password = ''" >> ~/.jupyter/jupyter_notebook_config.py

RUN pip install \
    scipy==1.7.2 \
    scikit-learn==1.0.1 \
    matplotlib-inline==0.1.3 \
    pandas==1.3.5

#install dependencies of https://github.com/lyakaap/ISC21-Descriptor-Track-1st
RUN pip install \
    augly==0.2.1 \
    h5py==0.2.1 \
    pytorch-metric-learning==1.0.0 \
    timm==0.4.12 \
    imgaug==0.4.0

CMD ["/bin/bash"]