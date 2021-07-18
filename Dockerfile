FROM ubuntu:focal

USER root

# https://bugs.debian.org/929417
RUN set -eux; \
	apt-get update; \
	apt-get install -y --no-install-recommends \
		ca-certificates \
		curl \
		netbase \
		wget \
		tzdata \
	; \
	rm -rf /var/lib/apt/lists/*

RUN set -ex; \
	if ! command -v gpg > /dev/null; then \
		apt-get update; \
		apt-get install -y --no-install-recommends \
			gnupg \
			dirmngr \
		; \
		rm -rf /var/lib/apt/lists/*; \
	fi

# procps is very common in build systems, and is a reasonably small package
RUN apt-get update && apt-get install -y --no-install-recommends \
		bzr \
		git \
		mercurial \
		openssh-client \
		subversion \
		\
		procps \
	&& rm -rf /var/lib/apt/lists/*

RUN set -ex; \
	apt-get update; \
	apt-get install -y --no-install-recommends \
		autoconf \
		automake \
		bzip2 \
		dpkg-dev \
		file \
		g++ \
		gcc \
		imagemagick \
		libbz2-dev \
		libc6-dev \
		libcurl4-openssl-dev \
		libdb-dev \
		libevent-dev \
		libffi-dev \
		libgdbm-dev \
		libglib2.0-dev \
		libgmp-dev \
		libjpeg-dev \
		libkrb5-dev \
		liblzma-dev \
		libmagickcore-dev \
		libmagickwand-dev \
		libmaxminddb-dev \
		libncurses5-dev \
		libncursesw5-dev \
		libpng-dev \
		libpq-dev \
		libreadline-dev \
		libsqlite3-dev \
		libssl-dev \
		libtool \
		libwebp-dev \
		libxml2-dev \
		libxslt-dev \
		libyaml-dev \
		make \
		patch \
		unzip \
		vim \
		xz-utils \
		zlib1g-dev \
		\
# https://lists.debian.org/debian-devel-announce/2016/09/msg00000.html
		$( \
# if we use just "apt-cache show" here, it returns zero because "Can't select versions from package 'libmysqlclient-dev' as it is purely virtual", hence the pipe to grep
			if apt-cache show 'default-libmysqlclient-dev' 2>/dev/null | grep -q '^Version:'; then \
				echo 'default-libmysqlclient-dev'; \
			else \
				echo 'libmysqlclient-dev'; \
			fi \
		) \
	; \
	rm -rf /var/lib/apt/lists/*


# Install util tools.
RUN apt-get update \ 
	&& apt-get install -y \ 
	apt-utils  \ 
	sudo \ 
	less  

RUN rm -rf /var/lib/apt/lists/*

# Create a working directory
RUN mkdir -p /workspace/app
WORKDIR /workspace/app

# Create a non-root user and switch to it
RUN adduser --disabled-password --gecos '' --shell /bin/bash user && \
 chown -R user:user /workspace/app
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
#USER user

# All users can use /home/user as their home directory
ENV HOME=/home/user
RUN chown -R user:user /home/user
RUN chmod 777 /home/user

# Install Miniconda and Python 3.8
ENV CONDA_AUTO_UPDATE_CONDA=false
ENV PATH=/home/user/miniconda/bin:$PATH
RUN curl -sLo ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \ 
	chmod +x ~/miniconda.sh && \ 
	~/miniconda.sh -b -p ~/miniconda && \
	rm ~/miniconda.sh && \
	conda install -y python==3.8.8 && \
	conda clean -ya

# No CUDA-specific steps
ENV NO_CUDA=1
RUN conda install -y -c conda-forge tensorflow keras
RUN conda install pytorch torchvision torchaudio cpuonly -c pytorch \
	&& conda install -y pandas \
	&& conda install -y scikit-learn \ 
	&& conda install -y matplotlib \
	&& conda install -y -c conda-forge keras \
	&& conda install -y -c huggingface transformers \
	&& conda install -y -c anaconda nltk \
	&& conda install -y -c anaconda seaborn \
	&& conda install -y -c conda-forge bokeh \
	&& conda install -y -c plotly plotly \
	&& conda install -y -c conda-forge sentencepiece \
	&& pip install -U sentence-transformers \
	&& pip install -U datasets \
	&& pip install -U pip setuptools wheel \
	&& pip install -U spacy \
	&& pip install -U gradio \
	&& pip install -U docx2python \
	&& conda clean -ya

RUN python -m spacy download en_core_web_trf

# Give back control
USER root
# Set the default command to python3
#CMD ["python3"]

# Set the default command to python3
RUN echo "export HF_HOME=/workspace/app/data/.cache/huggingface" >> ~/.bashrc
RUN echo "export TRANSFORMERS_CACHE=/workspace/app/data/.cache/huggingface" >> ~/.bashrc

COPY /app/code/ /workspace/app/code/
WORKDIR /workspace/app/code
#CMD ["/bin/bash"]
ENTRYPOINT ["python", "promatching.py"]
