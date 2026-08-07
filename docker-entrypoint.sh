#!/usr/bin/env bash
set -uo pipefail

# Replace the bootstrap oc with the build the target cluster reports, falling back on any failure.

OC_BIN="$(command -v oc || true)"

case "$(uname -m)" in
    x86_64) LINK_TEXT="Download oc for Linux for x86_64" ;;
    aarch64|arm64) LINK_TEXT="Download oc for Linux for ARM 64" ;;
    ppc64le) LINK_TEXT="Download oc for Linux for IBM Power, little endian" ;;
    s390x) LINK_TEXT="Download oc for Linux for IBM Z" ;;
    *) LINK_TEXT="" ;;
esac

if [ -z "${OC_BIN}" ]; then
    echo "docker-entrypoint: no bootstrap oc found on PATH, skipping cluster-match step" >&2
elif [ -z "${LINK_TEXT}" ]; then
    echo "docker-entrypoint: unsupported architecture $(uname -m), skipping cluster-match step" >&2
elif ! "${OC_BIN}" whoami >/dev/null 2>&1; then
    echo "docker-entrypoint: cluster not reachable with bootstrap oc, skipping cluster-match step" >&2
else
    HREF="$(
        "${OC_BIN}" get consoleclidownload oc-cli-downloads -o json 2>/dev/null \
            | LINK_TEXT="${LINK_TEXT}" python3 -c '
import json, os, sys
try:
    data = json.load(sys.stdin)
    link_text = os.environ["LINK_TEXT"]
    for link in data.get("spec", {}).get("links", []):
        if link.get("text") == link_text:
            print(link["href"])
            break
except Exception:
    pass
'
    )"

    if [ -z "${HREF}" ]; then
        echo "docker-entrypoint: could not resolve cluster-matched oc download link, keeping bootstrap oc" >&2
    else
        MATCH_DIR="${HOME}/.oc-cluster-match"
        mkdir -p "${MATCH_DIR}"
        TMP_TAR="$(mktemp /tmp/oc-cluster-match.XXXXXX.tar)"
        if curl -sSfLk "${HREF}" -o "${TMP_TAR}" && tar xf "${TMP_TAR}" -C "${MATCH_DIR}" oc; then
            chmod +x "${MATCH_DIR}/oc"
            export PATH="${MATCH_DIR}:${PATH}"
            echo "docker-entrypoint: using cluster-matched oc from ${HREF}" >&2
        else
            echo "docker-entrypoint: failed to download/extract cluster-matched oc from ${HREF}, keeping bootstrap oc" >&2
        fi
        rm -f "${TMP_TAR}"
    fi
fi

exec uv run pytest "$@"
