"""Verify a Jupyter kernel retained its in-memory state.

This script runs INSIDE the JupyterLab container via ``oc exec``.
It is not importable from the test framework — the dependencies
(jupyter_client, etc.) only exist inside the workbench image.

Usage (from oc exec):
    python /mnt/_kernel_verify.py <kernel_id>

Prints the evaluation result on success.
Exits non-zero if the variable ``a`` is missing or ``a * 6 != 42``.
"""

# pyrefly: ignore[unresolved-import]

import queue
import sys
import time

from jupyter_client import BlockingKernelClient  # type: ignore[import-untyped]

kernel_id = sys.argv[1]

conn_file = f"/opt/app-root/src/.local/share/jupyter/runtime/kernel-{kernel_id}.json"
kc = BlockingKernelClient()
kc.load_connection_file(connection_file=conn_file)  # noqa: FCN001
kc.start_channels()
kc.wait_for_ready(timeout=30)  # noqa: FCN001

msg_id = kc.execute(code="print(a * 6)")
reply = kc.get_shell_msg(timeout=10)
assert reply["content"]["status"] == "ok", f"Kernel execute failed: {reply['content']}"

time.sleep(1)
output_text = ""
while True:
    try:
        msg = kc.get_iopub_msg(timeout=2)
        if msg["msg_type"] == "stream":
            output_text += msg["content"]["text"]
    except (queue.Empty, TimeoutError) as exc:  # fmt: skip
        print(f"iopub drain stopped: {type(exc).__name__}: {exc}", file=sys.stderr)
        break

kc.stop_channels()

output_text = output_text.strip()
assert output_text == "42", f"Expected '42', got '{output_text}'"

print(output_text)
