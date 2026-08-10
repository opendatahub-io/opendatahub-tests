FROM fedora:43

ARG USER=odh
ARG HOME=/home/$USER
ARG TESTS_DIR=$HOME/opendatahub-tests/
ENV UV_PYTHON=python3.14
ENV UV_COMPILE_BYTECODE=1
ENV UV_NO_SYNC=1
ENV UV_NO_CACHE=1

ENV BIN_DIR="$HOME/.local/bin"
ENV PATH="$PATH:$BIN_DIR"

ARG TARGETARCH
ARG TARGETPLATFORM

RUN echo "Architecture: ${TARGETARCH}" && \
    echo "Platform: ${TARGETPLATFORM}"

# Install system dependencies using dnf
RUN dnf update -y \
    && dnf install -y python3 python3-pip python3-devel ssh gnupg curl gpg wget vim rsync openssl openssl-devel skopeo gcc-c++ \
    && dnf clean all \
    && rm -rf /var/cache/dnf

# Pinned SHA-256 digests for grpcurl v1.9.2 archives (from upstream checksums.txt)
# pragma: allowlist secret
ARG GRPCURL_SHA256_X86_64=1c7caf2628d8607d8a3bbee5ce7786bba4879abe566b075a4f129a97ccfa8465
# pragma: allowlist secret
ARG GRPCURL_SHA256_AARCH64=ebff3d0d022ad1c2010a1749048a6dccd7a41edf87093e80c7f649fcbb8eb751
# pragma: allowlist secret
ARG GRPCURL_SHA256_PPC64LE=f39edc1745c705988c29921697d058e052e9dd0c05f0df2552182aa777bc14c6

# Install grpcurl
RUN ARCH=$(uname -m) && \
    curl -sSL "https://github.com/fullstorydev/grpcurl/releases/download/v1.9.2/grpcurl_1.9.2_linux_${ARCH}.tar.gz" \
         --output /tmp/grpcurl.tar.gz && \
    case "${ARCH}" in \
      x86_64)  GRPCURL_SHA256="${GRPCURL_SHA256_X86_64}" ;; \
      aarch64) GRPCURL_SHA256="${GRPCURL_SHA256_AARCH64}" ;; \
      ppc64le) GRPCURL_SHA256="${GRPCURL_SHA256_PPC64LE}" ;; \
      *) echo "Unsupported architecture: ${ARCH}" && exit 1 ;; \
    esac && \
    echo "${GRPCURL_SHA256}  /tmp/grpcurl.tar.gz" | sha256sum --check --strict - && \
    tar xf /tmp/grpcurl.tar.gz --no-same-owner -C /tmp && \
    mv /tmp/grpcurl /usr/bin/grpcurl && \
    rm /tmp/grpcurl.tar.gz

# Pinned commit SHA for must-gather-clean v0.0.4 (lightweight tag → commit SHA)
# SHA: 1d2b91033362d2848207236cebabcd5ca2c366ed
# Pinned SHA-256 digest for the amd64 pre-built binary tarball (no upstream checksums file)
# amd64 tarball: 29e90cc94b593c6bf992923790a975ecf54bc1579c1e999b4acce90167b41235
# Install must-gather-clean
RUN if [[ $(uname -m) == "ppc64le" ]]; then \
     dnf install -y git && \
     git clone https://github.com/openshift/must-gather-clean /tmp/must-gather-clean && \
     cd /tmp/must-gather-clean && \
     git reset --hard 1d2b91033362d2848207236cebabcd5ca2c366ed && \
     dnf install -y make go && \
     make && dnf remove make go git -y && \
     cp ./must-gather-clean /usr/bin/must-gather-clean && \
     cd / && rm -rf /tmp/must-gather-clean; \
   elif [[ $(uname -m) == "x86_64" ]] ; then \
     curl -sSL "https://github.com/openshift/must-gather-clean/releases/download/v0.0.4/must-gather-clean-linux-amd64.tar.gz" \
          --output /tmp/must-gather-clean.tar.gz && \
     echo "29e90cc94b593c6bf992923790a975ecf54bc1579c1e999b4acce90167b41235  /tmp/must-gather-clean.tar.gz" | sha256sum --check --strict - && \
     tar xzf /tmp/must-gather-clean.tar.gz -C /tmp && \
     rm -f /tmp/must-gather-clean.tar.gz && \
     mv /tmp/must-gather-clean /usr/bin/must-gather-clean; \
   fi \
    && chmod +x /usr/bin/must-gather-clean

# Install cosign v3.0.4 (multi-arch, no expiration)
COPY --from=quay.io/securesign/cli-cosign@sha256:3df09cd1b4915e61d4de9c67416827b94e5900763e936e2909fd4d78e1ead8e8 /usr/local/bin/cosign /usr/bin/cosign

RUN useradd -ms /bin/bash $USER && chown -R $USER:$USER $HOME
USER $USER
WORKDIR $HOME

WORKDIR $TESTS_DIR
COPY --chown=$USER:$USER . $TESTS_DIR
ENV HOME=/home/$USER
ENV PATH="$HOME/.local/bin:${PATH}"

# Pinned uv 0.12.3 wheel hashes (sha256, from PyPI)
# x86_64  : 1482d1462b1aecd18ee33627363fe1c63d6a194f12d40d37efc446d9e0d800a1
# aarch64 : ac21bea426ddf95fa76d8dc1f67350faed7b4a81951825cf2aaef99fc4144815
# ppc64le : 913dc068e906f459df892cac789e6e9e10ac9d6af0bbb3c36fcc09347b0c986c
RUN python -V && mkdir -p $TESTS_DIR/results && \
    printf 'uv==0.12.3 \\\n    --hash=sha256:1482d1462b1aecd18ee33627363fe1c63d6a194f12d40d37efc446d9e0d800a1 \\\n    --hash=sha256:ac21bea426ddf95fa76d8dc1f67350faed7b4a81951825cf2aaef99fc4144815 \\\n    --hash=sha256:913dc068e906f459df892cac789e6e9e10ac9d6af0bbb3c36fcc09347b0c986c\n' > /tmp/uv-req.txt && \
    python -m pip install --require-hashes -r /tmp/uv-req.txt && \
    rm /tmp/uv-req.txt && \
    $HOME/.local/bin/uv sync

ENTRYPOINT ["uv", "run", "pytest"]
