#!/bin/bash

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Counters
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

# Check prerequisites
check_prerequisites() {
    echo "Checking prerequisites..."

    if ! command -v oc &> /dev/null; then
        echo -e "${RED}ERROR: oc CLI not found. Please install OpenShift CLI.${NC}"
        exit 1
    fi

    if ! command -v curl &> /dev/null; then
        echo -e "${RED}ERROR: curl not found. Please install curl.${NC}"
        exit 1
    fi

    if ! command -v jq &> /dev/null; then
        echo -e "${RED}ERROR: jq not found. Please install jq.${NC}"
        exit 1
    fi

    # Check if logged in to OpenShift
    if ! oc whoami &> /dev/null; then
        echo -e "${RED}ERROR: Not logged in to OpenShift. Please run 'oc login' first.${NC}"
        exit 1
    fi

    echo -e "${GREEN}All prerequisites met.${NC}\n"
}

# Get authentication token
get_auth_token() {
    oc whoami -t
}

# Get InferenceService URL
get_inference_service_url() {
    local namespace=$1
    local name=$2

    oc get isvc -n "$namespace" "$name" -o jsonpath='{.status.url}' 2>/dev/null
}

# Get LLMInferenceService URL
get_llm_service_url() {
    local namespace=$1
    local name=$2

    oc get llminferenceservice -n "$namespace" "$name" -o jsonpath='{.status.url}' 2>/dev/null
}

# Test InferenceService with V2 protocol
test_inference_service() {
    local namespace=$1
    local name=$2
    local model_name=$3
    local requires_auth=$4

    TOTAL_TESTS=$((TOTAL_TESTS + 1))

    echo "Testing InferenceService: $namespace/$name (model: $model_name)"

    # Get service URL
    local base_url=$(get_inference_service_url "$namespace" "$name")

    if [ -z "$base_url" ]; then
        echo -e "${RED}✗ FAILED: Could not retrieve service URL${NC}\n"
        FAILED_TESTS=$((FAILED_TESTS + 1))
        return 1
    fi

    local url="${base_url}/v2/models/${model_name}/infer"

    # Prepare payload
    local payload='{
      "inputs": [
        {
          "name": "input",
          "shape": [1, 1],
          "datatype": "FP32",
          "data": [1.0]
        }
      ]
    }'

    # Create temp file for response
    local temp_file=$(mktemp)

    # Execute request
    if [ "$requires_auth" = true ]; then
        local token=$(get_auth_token)
        local http_code=$(curl -s -w "%{http_code}" -o "$temp_file" -m 30 \
            -H "Authorization: Bearer $token" \
            -H "Content-Type: application/json" \
            -d "$payload" \
            "$url")
    else
        local http_code=$(curl -s -w "%{http_code}" -o "$temp_file" -m 30 \
            -H "Content-Type: application/json" \
            -d "$payload" \
            "$url")
    fi

    local body=$(cat "$temp_file")
    rm -f "$temp_file"

    # Check result
    if [ "$http_code" = "200" ]; then
        # Validate response has expected V2 protocol structure
        if echo "$body" | jq -e '.outputs' &> /dev/null || echo "$body" | jq -e '.model_name' &> /dev/null; then
            echo -e "${GREEN}✓ PASSED: HTTP $http_code, valid V2 response${NC}"
            PASSED_TESTS=$((PASSED_TESTS + 1))
        else
            echo -e "${YELLOW}⚠ PARTIAL: HTTP $http_code, but unexpected response format${NC}"
            echo "Response: $body"
            FAILED_TESTS=$((FAILED_TESTS + 1))
        fi
    else
        echo -e "${RED}✗ FAILED: HTTP $http_code${NC}"
        echo "Response: $body"
        FAILED_TESTS=$((FAILED_TESTS + 1))
    fi

    echo ""
}

# Test LLMInferenceService with OpenAI API
test_llm_service() {
    local namespace=$1
    local name=$2
    local model_name=$3
    local requires_auth=$4

    TOTAL_TESTS=$((TOTAL_TESTS + 1))

    echo "Testing LLMInferenceService: $namespace/$name (model: $model_name)"

    # Get service URL
    local base_url=$(get_llm_service_url "$namespace" "$name")

    if [ -z "$base_url" ]; then
        echo -e "${RED}✗ FAILED: Could not retrieve service URL${NC}\n"
        FAILED_TESTS=$((FAILED_TESTS + 1))
        return 1
    fi

    local url="${base_url}/v1/chat/completions"

    # Prepare payload
    local payload='{
      "model": "'"$model_name"'",
      "messages": [
        {
          "role": "user",
          "content": "Say hello"
        }
      ],
      "max_tokens": 10,
      "temperature": 0.0
    }'

    # Create temp file for response
    local temp_file=$(mktemp)

    # Execute request
    if [ "$requires_auth" = true ]; then
        local token=$(get_auth_token)
        local http_code=$(curl -s -w "%{http_code}" -o "$temp_file" -m 30 \
            -H "Authorization: Bearer $token" \
            -H "Content-Type: application/json" \
            -d "$payload" \
            "$url")
    else
        local http_code=$(curl -s -w "%{http_code}" -o "$temp_file" -m 30 \
            -H "Content-Type: application/json" \
            -d "$payload" \
            "$url")
    fi

    local body=$(cat "$temp_file")
    rm -f "$temp_file"

    # Check result
    if [ "$http_code" = "200" ]; then
        # Validate response has expected OpenAI API structure
        if echo "$body" | jq -e '.choices' &> /dev/null; then
            echo -e "${GREEN}✓ PASSED: HTTP $http_code, valid chat completion response${NC}"
            PASSED_TESTS=$((PASSED_TESTS + 1))
        else
            echo -e "${YELLOW}⚠ PARTIAL: HTTP $http_code, but unexpected response format${NC}"
            echo "Response: $body"
            FAILED_TESTS=$((FAILED_TESTS + 1))
        fi
    else
        echo -e "${RED}✗ FAILED: HTTP $http_code${NC}"
        echo "Response: $body"
        FAILED_TESTS=$((FAILED_TESTS + 1))
    fi

    echo ""
}

# Print summary
print_summary() {
    echo "=========================================="
    echo "           TEST SUMMARY"
    echo "=========================================="
    echo "Total tests:  $TOTAL_TESTS"
    echo -e "Passed:       ${GREEN}$PASSED_TESTS${NC}"
    echo -e "Failed:       ${RED}$FAILED_TESTS${NC}"
    echo "=========================================="

    if [ $FAILED_TESTS -eq 0 ]; then
        echo -e "${GREEN}All tests passed!${NC}"
        exit 0
    else
        echo -e "${RED}Some tests failed.${NC}"
        exit 1
    fi
}

# Main execution
main() {
    echo "=========================================="
    echo "  Inference Service Testing Script"
    echo "=========================================="
    echo ""

    check_prerequisites

    echo "Starting tests..."
    echo ""

    # Test InferenceServices WITH authentication
    echo "--- InferenceServices (with auth) ---"
    test_inference_service "kserve-raw-mm" "model1" "model1" true
    test_inference_service "kserve-raw-mm" "model2" "model2" true
    test_inference_service "up-raw-kserve" "raw-model1" "raw-model1" true

    # Test InferenceServices WITHOUT authentication
    echo "--- InferenceServices (without auth) ---"
    test_inference_service "kserve-raw-mm" "model2-2" "model2-2" false
    test_inference_service "up-raw-kserve" "oci-unauth" "oci-unauth" false
    test_inference_service "up-raw-kserve" "raw-model-unauth" "raw-model-unauth" false
    test_inference_service "upgarde-seless" "auth1-serv-raw" "auth1-serv-raw" false
    test_inference_service "upgarde-seless" "oci-serv-raw" "oci-serv-raw" false
    test_inference_service "upgarde-seless" "unauth-serv-raw" "unauth-serv-raw" false

    # Test LLMInferenceService
    echo "--- LLMInferenceServices ---"
    test_llm_service "llm-demo" "llm-basic" "llm-basic" true

    # Print summary
    print_summary
}

# Run main
main
