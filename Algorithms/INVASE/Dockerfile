
RUN apt-get update

RUN conda install -y -c bioconda/label/cf201901 arboreto=0.1.5

COPY runINVASE.py /

RUN mkdir data/

RUN apt-get install time