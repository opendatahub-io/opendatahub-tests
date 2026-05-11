#!/usr/bin/env bash
# Test kube-rbac-proxy integration with EvalHub
# Generates test results in the format specified in kube-rbac-proxy_tests_EH.md
#
# Usage:
#   ./test_kube_rbac_proxy.sh

set -euo pipefail

# Configuration
: "${EVALHUB_NAMESPACE:?error: EVALHUB_NAMESPACE must be set}"
: "${EVALHUB_BASE_URL:?error: EVALHUB_BASE_URL must be set}"
NAMESPACE="${EVALHUB_NAMESPACE}"
BASE_URL="${EVALHUB_BASE_URL}"
OUTPUT_FILE="${1:-/tmp/kube-rbac-proxy-test-results.md}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Counters
TESTS_PASSED=0
TESTS_FAILED=0

# Helper functions
log_test() {
    local test_num="$1"
    local test_name="$2"
    echo -e "${YELLOW}Running TEST $test_num: $test_name${NC}"
}

log_pass() {
    echo -e "${GREEN}✅ PASS${NC}"
    ((++TESTS_PASSED))
}

log_fail() {
    echo -e "${RED}✗ FAIL${NC}"
    ((++TESTS_FAILED))
}

# Get fresh token
get_token() {
    oc whoami -t
}

# Get evalhub pod name (the one that's Ready)
get_evalhub_pod() {
    oc get pod -n "$NAMESPACE" -l app=eval-hub -o json | jq -r '.items[] | select(.status.conditions[]? | select(.type=="Ready" and .status=="True")) | .metadata.name' | head -1
}

# Get configuration from pod
get_pod_config() {
    local pod="$1"
    echo "### Pod: $pod"
    echo "### Namespace: $NAMESPACE"
    echo ""
    echo "### kube-rbac-proxy Configuration:"
    echo '```'
    oc get pod "$pod" -n "$NAMESPACE" -o jsonpath='{.spec.containers[?(@.name=="kube-rbac-proxy")].args}' | \
        python3 -c 'import json,sys; args=json.load(sys.stdin); print("\n".join(args))'
    echo '```'
    echo ""
    echo "### evalhub Configuration:"
    echo '```yaml'
    oc get configmap evalhub-config -n "$NAMESPACE" -o jsonpath='{.data.config\.yaml}' | grep -A 5 "service:"
    echo '```'
}

# Initialize output file
init_output() {
    cat > "$OUTPUT_FILE" <<'EOF'
# EvalHub + kube-rbac-proxy Integration Test Results

## Configuration Summary

EOF

    local pod
    pod="$(get_evalhub_pod)" || {
        echo "ERROR: Failed to query EvalHub pod in namespace $NAMESPACE" >&2
        exit 1
    }
    if [ -z "$pod" ]; then
        echo "ERROR: No running evalhub pod found in namespace $NAMESPACE"
        exit 1
    fi

    get_pod_config "$pod" >> "$OUTPUT_FILE"

    cat >> "$OUTPUT_FILE" <<'EOF'

---

EOF
}

# Run a test and capture results
run_test() {
    local test_num="$1"
    local test_name="$2"
    local objective="$3"
    local expected_status="$4"
    shift 4
    local -a curl_cmd_array=("$@")

    log_test "$test_num" "$test_name"

    # Execute curl and capture response
    local temp_response
    local temp_headers
    temp_response="$(mktemp)" || {
        echo "ERROR: Failed to create temp response file for TEST $test_num" >&2
        exit 1
    }
    if [ -z "$temp_response" ]; then
        echo "ERROR: mktemp returned empty response file path for TEST $test_num" >&2
        exit 1
    fi
    temp_headers="$(mktemp)" || {
        echo "ERROR: Failed to create temp headers file for TEST $test_num" >&2
        rm -f "$temp_response"
        exit 1
    }
    if [ -z "$temp_headers" ]; then
        echo "ERROR: mktemp returned empty headers file path for TEST $test_num" >&2
        rm -f "$temp_response"
        exit 1
    fi

    if [ "${#curl_cmd_array[@]}" -eq 0 ] || [ "${curl_cmd_array[0]}" != "curl" ]; then
        echo "ERROR: TEST $test_num requires a curl command array" >&2
        rm -f "$temp_response" "$temp_headers"
        exit 1
    fi

    # Validate flags when command input is externally provided.
    local i=1
    while [ "$i" -lt "${#curl_cmd_array[@]}" ]; do
        local token="${curl_cmd_array[$i]}"
        case "$token" in
            -k|-s|-S|-L|-i|--silent|--show-error|--location|--insecure)
                ;;
            -H|--header|-X|--request|-d|--data|--data-raw|--url)
                i=$((i + 1))
                if [ "$i" -ge "${#curl_cmd_array[@]}" ]; then
                    echo "ERROR: Missing value for curl flag '$token' in TEST $test_num" >&2
                    rm -f "$temp_response" "$temp_headers"
                    exit 1
                fi
                ;;
            http://*|https://*)
                ;;
            *)
                echo "ERROR: Unsupported curl token '$token' in TEST $test_num" >&2
                rm -f "$temp_response" "$temp_headers"
                exit 1
                ;;
        esac
        i=$((i + 1))
    done

    local -a request_cmd=("${curl_cmd_array[@]}")
    local -a redacted_request_cmd=("${request_cmd[@]}")
    i=0
    while [ "$i" -lt "${#redacted_request_cmd[@]}" ]; do
        case "${redacted_request_cmd[$i]}" in
            -H|--header)
                if [ "$((i + 1))" -lt "${#redacted_request_cmd[@]}" ] && \
                    [[ "${redacted_request_cmd[$((i + 1))]}" == Authorization:\ Bearer* ]]; then
                    redacted_request_cmd[$((i + 1))]="Authorization: Bearer REDACTED"
                fi
                i=$((i + 1))
                ;;
            --header=Authorization:\ Bearer*)
                redacted_request_cmd[$i]="--header=Authorization: Bearer REDACTED"
                ;;
        esac
        i=$((i + 1))
    done
    local -a exec_cmd=("${curl_cmd_array[@]}" -w '\n%{http_code}' -D "$temp_headers")
    local curl_exit=0
    set +e
    "${exec_cmd[@]}" > "$temp_response" 2>&1
    curl_exit=$?
    set -e

    local actual_status
    local response_body
    local total_lines
    local body_lines

    if [ "$curl_exit" -ne 0 ]; then
        # Sentinel: not a real HTTP status; ensures comparison fails and output records transport failure
        actual_status="000"
        response_body="$(cat "$temp_response" 2>/dev/null || true)"
    else
        # Extract status code (last line)
        actual_status="$(tail -1 "$temp_response")" || {
            echo "ERROR: Failed to extract status code for TEST $test_num" >&2
            rm -f "$temp_response" "$temp_headers"
            exit 1
        }
        # Extract response body (everything except last line)
        total_lines="$(wc -l < "$temp_response" | tr -d ' ')" || {
            echo "ERROR: Failed to read response line count for TEST $test_num" >&2
            rm -f "$temp_response" "$temp_headers"
            exit 1
        }
        body_lines=$((total_lines - 1))
        response_body="$(head -n "$body_lines" "$temp_response")" || {
            echo "ERROR: Failed to extract response body for TEST $test_num" >&2
            rm -f "$temp_response" "$temp_headers"
            exit 1
        }
    fi

    # Determine pass/fail
    local result
    if [ "$actual_status" = "$expected_status" ]; then
        result="✅ PASS"
        log_pass
    else
        result="✗ FAIL"
        log_fail
    fi

    # Write to output file
    cat >> "$OUTPUT_FILE" <<EOF

## TEST $test_num: $test_name

**Objective:** $objective

**Request:**
\`\`\`bash
$(printf '%q ' "${redacted_request_cmd[@]}")
\`\`\`

**Response Status:** $actual_status

EOF
    if [ "$curl_exit" -ne 0 ]; then
        echo "**Transport:** curl exited with code ${curl_exit} (status ${actual_status} is a sentinel, not HTTP)." >> "$OUTPUT_FILE"
    fi

    if [ "$actual_status" = "$expected_status" ]; then
        echo "**Response Body (first 500 chars):**" >> "$OUTPUT_FILE"
    else
        echo "**Response Body:**" >> "$OUTPUT_FILE"
    fi

    echo '```json' >> "$OUTPUT_FILE"
    if [ ${#response_body} -gt 500 ]; then
        echo "${response_body:0:500}..." >> "$OUTPUT_FILE"
    else
        echo "$response_body" >> "$OUTPUT_FILE"
    fi
    echo '```' >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"

    # Get evalhub logs if request reached the backend
    if [ "$actual_status" = "200" ] || [ "$actual_status" = "202" ] || [ "$actual_status" = "409" ]; then
        echo "**evalhub Logs (showing received headers):**" >> "$OUTPUT_FILE"
        echo '```json' >> "$OUTPUT_FILE"
        local pod
        pod="$(get_evalhub_pod)" || {
            echo "ERROR: Failed to query EvalHub pod for log capture in TEST $test_num" >&2
            rm -f "$temp_response" "$temp_headers"
            exit 1
        }
        {
            oc logs "$pod" -n "$NAMESPACE" -c evalhub --tail=2 2>/dev/null | grep -E '"tenant"|"user"' || \
                echo "No recent logs found"
            echo '```'
            echo ""
        } >> "$OUTPUT_FILE"
    fi

    echo "**Result:** $result - ${test_name}" >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"

    # Cleanup
    rm -f "$temp_response" "$temp_headers"
}

# Main test execution
main() {
    echo "Starting kube-rbac-proxy integration tests..."
    echo "Namespace: $NAMESPACE"
    echo "Base URL: $BASE_URL"
    echo ""

    # Initialize output
    init_output

    # Get token
    TOKEN="$(get_token)" || {
        echo "ERROR: Failed to get OpenShift token"
        exit 1
    }
    if [ -z "$TOKEN" ]; then
        echo "ERROR: Failed to get OpenShift token"
        exit 1
    fi

    # TEST 1: Valid Authentication
    run_test 1 \
        "Authentication - Valid ServiceAccount Token" \
        "Verify that kube-rbac-proxy accepts valid Kubernetes ServiceAccount tokens" \
        "200" \
        curl -k -H "Authorization: Bearer $TOKEN" -H "X-Tenant: $NAMESPACE" "$BASE_URL/api/v1/evaluations/providers"

    # TEST 2: Missing Authentication
    run_test 2 \
        "Authentication - Invalid/Missing Token" \
        "Verify that kube-rbac-proxy rejects requests without valid authentication" \
        "401" \
        curl -k -H "X-Tenant: $NAMESPACE" "$BASE_URL/api/v1/evaluations/providers"

    # TEST 3: Configured Endpoint (Allowed)
    run_test 3 \
        "Authorization - Configured Endpoint (Allowed)" \
        "Verify that authorized endpoints in auth.yaml configuration are accessible" \
        "200" \
        curl -k -H "Authorization: Bearer $TOKEN" -H "X-Tenant: $NAMESPACE" "$BASE_URL/api/v1/evaluations/providers"

    # TEST 4: Unconfigured Endpoint (Blocked)
    run_test 4 \
        "Authorization - Unconfigured Endpoint (Blocked)" \
        "Verify that endpoints NOT in auth.yaml configuration are blocked" \
        "403" \
        curl -k -H "Authorization: Bearer $TOKEN" -H "X-Tenant: $NAMESPACE" "$BASE_URL/openapi.yaml"

    # TEST 5: Health Endpoint Bypass
    run_test 5 \
        "Health Endpoint Bypass (--ignore-paths)" \
        "Verify that /api/v1/health bypasses authentication and authorization checks" \
        "200" \
        curl -k "$BASE_URL/api/v1/health"

    # Generate summary
    cat >> "$OUTPUT_FILE" <<EOF

---

## TEST SUMMARY

### All Tests: $((TESTS_PASSED + TESTS_FAILED)) total

| Test # | Test Name | Result |
|--------|-----------|--------|
EOF

    local total=$((TESTS_PASSED + TESTS_FAILED))
    if [ $TESTS_PASSED -eq $total ]; then
        echo "| ALL | - | ✅ PASSED ($TESTS_PASSED/$total) |" >> "$OUTPUT_FILE"
    else
        echo "| - | - | ❌ FAILED ($TESTS_FAILED/$total) |" >> "$OUTPUT_FILE"
    fi

    echo "" >> "$OUTPUT_FILE"
    echo "**Test Results:**" >> "$OUTPUT_FILE"
    echo "- ✅ Passed: $TESTS_PASSED" >> "$OUTPUT_FILE"
    echo "- ❌ Failed: $TESTS_FAILED" >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"

    # Print results
    echo ""
    echo "================================="
    echo "Test Results:"
    echo "  Passed: $TESTS_PASSED"
    echo "  Failed: $TESTS_FAILED"
    echo "================================="
    echo ""
    echo "Full results written to: $OUTPUT_FILE"

    # Exit with error if any tests failed
    if [ $TESTS_FAILED -gt 0 ]; then
        exit 1
    fi
}

main
