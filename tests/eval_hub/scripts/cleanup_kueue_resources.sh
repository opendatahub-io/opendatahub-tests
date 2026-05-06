#!/bin/bash
# Clean up Kueue resources before running EvalHub tests

echo "Cleaning up evalhub test namespaces..."
for ns in $(oc get namespace -o name | grep evalhub); do
    oc delete $ns --ignore-not-found=true
done

echo "Cleaning up evalhub ClusterQueues..."
oc delete clusterqueue evalhub-test-cq team-a-cq team-b-cq --ignore-not-found=true

echo "Waiting for ClusterQueues to be deleted..."
sleep 5

echo "Cleaning up evalhub ResourceFlavors..."
oc delete resourceflavor evalhub-test-flavor evalhub-multi-test-flavor --ignore-not-found=true

echo "Cleaning up WorkloadPriorityClass..."
oc delete workloadpriorityclass evalhub-test-high-priority --ignore-not-found=true

echo "✓ Cleanup complete"
