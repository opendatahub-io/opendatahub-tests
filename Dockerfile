FROM python:3.13

ARG USER=odh
ARG TESTS_DIR=/home/$USER/opendatahub-tests/

ENV UV_INSTALL_DIR="/home/$USER/.local"
ENV PATH="${PATH}:$UV_INSTALL_DIR/bin"

RUN apt-get update \
    && apt-get install -y ssh gnupg software-properties-common curl gpg wget vim \
    && apt-get clean autoclean \
    && apt-get autoremove --yes \
    && rm -rf /var/lib/{apt,dpkg,cache,log}/

# Install the Rosa CLI
RUN curl -L https://mirror.openshift.com/pub/openshift-v4/clients/rosa/latest/rosa-linux.tar.gz --output /tmp/rosa-linux.tar.gz \
    && tar xvf /tmp/rosa-linux.tar.gz --no-same-owner \
    && mv rosa /usr/bin/rosa \
    && chmod +x /usr/bin/rosa \
    && rosa version

# Install the OpenShift CLI (OC)
RUN curl -L https://mirror.openshift.com/pub/openshift-v4/x86_64/clients/ocp/stable/openshift-client-linux.tar.gz --output /tmp/openshift-client-linux.tar.gz \
    && tar xvf /tmp/openshift-client-linux.tar.gz --no-same-owner \
    && mv oc /usr/bin/oc \
    && chmod +x /usr/bin/oc

RUN useradd -ms /bin/bash $USER
USER $USER
WORKDIR $TESTS_DIR
COPY --chown=$USER:$USER . $TESTS_DIR

# Download the latest uv installer and create the virtual environment
RUN curl -sSL https://astral.sh/uv/install.sh -o /tmp/uv-installer.sh \
  && sh /tmp/uv-installer.sh \
  && rm /tmp/uv-installer.sh
RUN uv sync

ENTRYPOINT ["uv", "run", "pytest"]
