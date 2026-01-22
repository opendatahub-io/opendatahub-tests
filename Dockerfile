FROM fedora:42

ARG USER=odh
ARG HOME=/home/$USER
ARG TESTS_DIR=$HOME/opendatahub-tests/
ENV UV_PYTHON=python3.14
ENV UV_COMPILE_BYTECODE=1
ENV UV_NO_SYNC=1
ENV UV_NO_CACHE=1

ENV BIN_DIR="$HOME_DIR/.local/bin"
ENV PATH="$PATH:$BIN_DIR"

# Install Python 3.13 and other dependencies using dnf
RUN dnf update -y \
    && dnf install -y python3.13 python3.13-pip ssh gnupg curl gpg wget vim httpd-tools rsync openssl openssl-devel\
    && dnf clean all \
    && rm -rf /var/cache/dnf

# Install grpcurl
RUN curl -sSL "https://github.com/fullstorydev/grpcurl/releases/download/v1.9.2/grpcurl_1.9.2_linux_x86_64.tar.gz" --output /tmp/grpcurl_1.2.tar.gz \
    && tar xvf /tmp/grpcurl_1.2.tar.gz --no-same-owner \
    && mv grpcurl /usr/bin/grpcurl

RUN useradd -ms /bin/bash $USER
USER $USER
WORKDIR $HOME_DIR
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx ${BIN_DIR}/

WORKDIR $TESTS_DIR
COPY --chown=$USER:$USER . $TESTS_DIR

RUN uv sync

ENTRYPOINT ["uv", "run", "pytest"]
