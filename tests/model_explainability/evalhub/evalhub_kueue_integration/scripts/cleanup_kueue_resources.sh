#!/bin/bash
# Opt-in cleanup for legacy EvalHub Kueue test object names (cluster-scoped CRs).
# Does nothing unless KUEUE_EVALHUB_FORCE_CLEANUP=1 — safe for shared clusters by default.
# Namespaces are not deleted here; tests handle namespace lifecycle via fixtures.

set -euo pipefail

if [[ "${KUEUE_EVALHUB_FORCE_CLEANUP:-}" != "1" ]]; then
  cat <<'EOF'
No cleanup performed. This script deletes fixed evalhub test names only when explicitly enabled.

Objects are normally removed by test fixture teardown. To force-delete legacy names after manual triage:

  export KUEUE_EVALHUB_FORCE_CLEANUP=1
  # verify oc context, then re-run this script

EOF
  exit 0
fi

echo "Cleaning up evalhub ClusterQueues (KUEUE_EVALHUB_FORCE_CLEANUP=1)..."
oc delete clusterqueue evalhub-test-cq team-a-cq team-b-cq --ignore-not-found=true

echo "Waiting for ClusterQueues to be deleted..."
sleep 5

echo "Cleaning up evalhub ResourceFlavors..."
oc delete resourceflavor evalhub-test-flavor evalhub-multi-test-flavor --ignore-not-found=true

echo "Cleaning up WorkloadPriorityClass..."
oc delete workloadpriorityclass evalhub-test-high-priority --ignore-not-found=true

echo "✓ Manual cleanup complete"
