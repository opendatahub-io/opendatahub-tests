"""Start a Jupyter kernel and set a test variable.

This script runs INSIDE the JupyterLab container via ``oc exec``.
It is not importable from the test framework — the dependencies
(jupyter_client, etc.) only exist inside the workbench image.

Usage (from oc exec):
    python /mnt/_kernel_start.py <base_url>

Prints the kernel_id on success.
"""

# pyrefly: ignore[unresolved-import]

import http.cookiejar
import json
import sys
import time
import urllib.request

from jupyter_client import BlockingKernelClient  # type: ignore[import-untyped]

base_url = sys.argv[1]

cj = http.cookiejar.CookieJar()
opener = urllib.request.build_opener(  # noqa: FCN001
    urllib.request.HTTPCookieProcessor(cj),
)
opener.open(f"{base_url}/lab")

xsrf = next(c.value for c in cj if c.name == "_xsrf")

req = urllib.request.Request(
    f"{base_url}/api/kernels",
    data=json.dumps({"name": "python3"}).encode(),
    headers={"Content-Type": "application/json", "X-XSRFToken": xsrf},
    method="POST",
)
resp = opener.open(req, timeout=30)
kernel = json.loads(resp.read())
kernel_id = kernel["id"]
resp.close()

time.sleep(2)

conn_file = f"/opt/app-root/src/.local/share/jupyter/runtime/kernel-{kernel_id}.json"
kc = BlockingKernelClient()
kc.load_connection_file(connection_file=conn_file)  # noqa: FCN001
kc.start_channels()
kc.wait_for_ready(timeout=30)  # noqa: FCN001

msg_id = kc.execute(code="a = 3 + 4")
reply = kc.get_shell_msg(timeout=10)
assert reply["content"]["status"] == "ok", f"Kernel execute failed: {reply['content']}"

kc.stop_channels()

print(kernel_id)
