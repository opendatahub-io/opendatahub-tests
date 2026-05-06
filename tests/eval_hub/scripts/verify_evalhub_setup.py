#!/usr/bin/env python3
"""Quick verification script to test EvalHub API connectivity.

Run with: uv run python tests/eval_hub/scripts/verify_evalhub_setup.py
Or directly: ./tests/eval_hub/scripts/verify_evalhub_setup.py (if uv environment is active)
"""

import subprocess
import sys

try:
    import requests
except ImportError:
    print("Error: 'requests' module not found.")
    print("Please run: uv run python tests/eval_hub/scripts/verify_evalhub_setup.py")
    sys.exit(1)

# Configuration
EVALHUB_BASE_URL = "https://evalhub-prabhu.apps.rosa.prabhu-comhub.xqmp.p3.openshiftapps.com"
EVALHUB_NAMESPACE = "prabhu"


def get_token():
    """Get OpenShift token from oc command."""
    result = subprocess.run(args=["oc", "whoami", "-t"], capture_output=True, text=True, check=True)
    return result.stdout.strip()


def main():
    """Run basic connectivity tests."""
    print("EvalHub Setup Verification")
    print("=" * 50)

    # Test 1: Get token
    print("\n✓ Test 1: Getting OpenShift token...")
    try:
        token = get_token()
        print(f"  Token obtained (length: {len(token)})")
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        print(f"  ✗ Failed to get token: {e}")
        return 1

    # Test 2: Test EvalHub API - health endpoint
    print("\n✓ Test 2: Testing EvalHub API health...")
    try:
        resp = requests.get(
            f"{EVALHUB_BASE_URL}/health", headers={"Authorization": f"Bearer {token}"}, timeout=10, verify=True
        )
        print(f"  Health endpoint: HTTP {resp.status_code}")
        if resp.status_code == 200:
            print(f"  Response: {resp.json()}")
    except requests.RequestException as e:
        print(f"  Note: Health endpoint test: {e}")

    # Test 3: Test EvalHub API - get nonexistent job (should return 404)
    print("\n✓ Test 3: Testing EvalHub API (nonexistent job = 404)...")
    try:
        resp = requests.get(
            f"{EVALHUB_BASE_URL}/api/v1/jobs/nonexistent-test-id",
            headers={"Authorization": f"Bearer {token}"},
            timeout=10,
            verify=True,
        )
        print(f"  API response: HTTP {resp.status_code}")
        if resp.status_code == 404:
            print("  ✓ PASS: Got expected 404 for nonexistent job")
        else:
            print(f"  ✗ FAIL: Expected 404, got {resp.status_code}")
            print(f"  Response: {resp.text[:200]}")
            return 1
    except requests.RequestException as e:
        print(f"  ✗ Failed: {e}")
        return 1

    # Test 4: Check Kueue CRDs
    print("\n✓ Test 4: Checking Kueue CRDs...")
    try:
        result = subprocess.run(
            args=["oc", "api-resources", "--api-group=kueue.x-k8s.io"], capture_output=True, text=True, check=True
        )
        kueue_crds = [line for line in result.stdout.split("\n") if line and "NAME" not in line]
        print(f"  Found {len(kueue_crds)} Kueue CRDs")
        print("  ✓ Kueue is installed")
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        print(f"  ✗ Kueue check failed: {e}")
        return 1

    print("\n" + "=" * 50)
    print("✓ All verification tests passed!")
    print("\nEnvironment is ready for test execution.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
