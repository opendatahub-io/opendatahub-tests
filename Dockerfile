FROM fedora:43

ARG USER=odh
ARG HOME=/home/$USER
ARG TESTS_DIR=$HOME/opendatahub-tests/
ENV UV_PYTHON=python3.14
ENV UV_COMPILE_BYTECODE=1
ENV UV_NO_SYNC=1
ENV UV_NO_CACHE=1

ENV BIN_DIR="$HOME_DIR/.local/bin"
ENV PATH="$PATH:$BIN_DIR"

# Install system dependencies using dnf
RUN dnf update -y \
    && dnf install -y python3 python3-pip python3-devel ssh gnupg curl gpg wget vim rsync openssl openssl-devel skopeo gcc-c++\
    && dnf clean all \
    && rm -rf /var/cache/dnf

# Install grpcurl
RUN curl -sSL "https://github.com/fullstorydev/grpcurl/releases/download/v1.9.2/grpcurl_1.9.2_linux_x86_64.tar.gz" --output /tmp/grpcurl_1.2.tar.gz \
    && tar xvf /tmp/grpcurl_1.2.tar.gz --no-same-owner \
    && mv grpcurl /usr/bin/grpcurl

# Install must-gather-clean
RUN wget https://github.com/openshift/must-gather-clean/releases/download/v0.0.4/must-gather-clean-linux-amd64.tar.gz -q \
    && tar xzf must-gather-clean-linux-amd64.tar.gz \
    && mv must-gather-clean /usr/bin/must-gather-clean \
    && chmod +x /usr/bin/must-gather-clean \
    && rm -f must-gather-clean-linux-amd64.tar.gz

# Install cosign v3.0.4 (multi-arch, no expiration)
COPY --from=quay.io/securesign/cli-cosign@sha256:3df09cd1b4915e61d4de9c67416827b94e5900763e936e2909fd4d78e1ead8e8 /usr/local/bin/cosign /usr/bin/cosign

RUN useradd -ms /bin/bash $USER && chown -R $USER:$USER $HOME
USER $USER
WORKDIR $HOME_DIR
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx ${BIN_DIR}/

WORKDIR $TESTS_DIR
COPY --chown=$USER:$USER . $TESTS_DIR

RUN uv sync

ENTRYPOINT ["uv", "run", "pytest"]
