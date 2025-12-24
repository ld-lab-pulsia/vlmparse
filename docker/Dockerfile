FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04 AS base-env

# No prompt possible
ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100


# The environmental variables below allow launching jupyter as non-root
ENV JUPYTER_CONFIG_DIR=/workspace/.jupyter/config
ENV JUPYTER_DATA_DIR=/workspace/.jupyter/data
ENV JUPYTER_RUNTIME_DIR=/workspace/.jupyter/runtime

# Add the library to the Python path to be able to import it in console
ENV PIP_NO_CACHE_DIR=off

# Create user to avoid using root user
ENV \
    DOCKER_USER_NAME=probayes \
    DOCKER_USER_UID=1000 \
    DOCKER_USER_GID=1000 \
    DOCKER_USER_HOME="/home/probayes"


RUN groupadd -g ${DOCKER_USER_GID} ${DOCKER_USER_NAME} \
    && useradd \
    --system \
    -l \
    -u ${DOCKER_USER_UID} \
    -g ${DOCKER_USER_GID} \
    --shell /sbin/nologin \
    --create-home --home-dir ${DOCKER_USER_HOME} \
    ${DOCKER_USER_NAME}

RUN apt-get update  \
    && apt-get upgrade -y \
    && apt-get install --no-install-recommends -y curl python3.10 python3-pip python3-dev python3-setuptools python3-venv \
    && apt-get install -y libopenmpi-dev \
    && apt purge --auto-remove -y  \
    && apt autoremove --purge -y \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir pre-commit


# Fix Opencv install error
RUN apt-get update && apt-get install supervisor ca-certificates openssh-server libpam-ldap libnss-ldap ldap-utils nscd sudo cifs-utils vim nano git-lfs python3.10-venv ffmpeg tmux gawk libsm6 libxext6 fontconfig ttf-mscorefonts-installer libreoffice  poppler-utils fonts-crosextra-carlito fonts-crosextra-caladea fonts-freefont-ttf fonts-liberation fonts-urw-base35 fonts-comic-neue fonts-texgyre fonts-dejavu -y && git lfs install && fc-cache -f -v && apt-get clean && rm -rf /var/lib/apt/lists/*


RUN mkdir -p /.cache && chmod 777 /.cache

# Add uv
ENV UV_PROJECT_ENVIRONMENT="/opt/venvs"
ENV UV_CACHE_DIR="/.cache/.uv_cache/"
ENV UV_PYTHON_INSTALL_DIR="/opt/python"
RUN echo 'export UV_PROJECT_ENVIRONMENT='$UV_PROJECT_ENVIRONMENT >> /etc/skel/.bashrc
RUN echo 'export UV_CACHE_DIR='$UV_CACHE_DIR >> /etc/skel/.bashrc
RUN echo 'source $UV_PROJECT_ENVIRONMENT/bin/activate' >> /etc/skel/.bashrc
ENV PATH="$UV_PROJECT_ENVIRONMENT/bin:$PATH"
RUN echo 'export PATH="$UV_PROJECT_ENVIRONMENT/bin:$PATH"' >> /etc/skel/.bashrc

# Instal uv
RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="/bin" sh \
    && chmod +x /bin/uv

# Initialize directories and copy resource as docker non-root-user
USER ${DOCKER_USER_NAME}
ENV VIRTUAL_ENV_DIR=/workspace
WORKDIR $VIRTUAL_ENV_DIR
COPY --chown=${DOCKER_USER_NAME}:${DOCKER_USER_NAME} . /workspace
WORKDIR /workspace/docparser
USER root

# ENV PATH="/workspace/docparser/.venv/bin:$PATH"
RUN uv sync --all-extras && \
    chmod -R 777 $UV_PROJECT_ENVIRONMENT && \
    chmod -R 777 $UV_CACHE_DIR && \
    chmod -R 777 $UV_PYTHON_INSTALL_DIR

RUN playwright install chromium

WORKDIR /workspace

USER ${DOCKER_USER_NAME}
