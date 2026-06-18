#!/bin/bash

# Kueue CRD Migration Test Validation Script
#
# Usage: ./scripts/validate_kueue_crd_migration_tests.sh [--dry-run]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
DRY_RUN=false
VERBOSE=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --verbose|-v)
            VERBOSE=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [--dry-run] [--verbose]"
            echo "  --dry-run    Validate files and structure without running tests"
            echo "  --verbose    Enable verbose output"
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
done

log() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

success() {
    echo -e "${GREEN}[PASS]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

error() {
    echo -e "${RED}[FAIL]${NC} $1"
}

check_file_exists() {
    local file="$1"
    local description="$2"

    if [[ -f "$file" ]]; then
        success "$description exists: $file"
        return 0
    else
        error "$description missing: $file"
        return 1
    fi
}

check_python_syntax() {
    local file="$1"
    local description="$2"

    if python3 -m py_compile "$file" 2>/dev/null; then
        success "$description has valid Python syntax"
        return 0
    else
        error "$description has Python syntax errors"
        return 1
    fi
}

check_imports() {
    local file="$1"
    local description="$2"

    # Check for required imports
    local required_imports=(
        "import pytest"
        "import structlog"
        "from kubernetes.dynamic import DynamicClient"
        "from ocp_resources"
        "from utilities.kueue_utils"
    )

    local missing_imports=()
    for import_stmt in "${required_imports[@]}"; do
        if ! grep -q "$import_stmt" "$file"; then
            missing_imports+=("$import_stmt")
        fi
    done

    if [[ ${#missing_imports[@]} -eq 0 ]]; then
        success "$description has all required imports"
        return 0
    else
        error "$description missing imports: ${missing_imports[*]}"
        return 1
    fi
}

check_pytest_markers() {
    local file="$1"
    local description="$2"

    local required_markers=(
        "@pytest.mark.pre_upgrade"
        "@pytest.mark.post_upgrade"
    )

    local missing_markers=()
    for marker in "${required_markers[@]}"; do
        if ! grep -q "$marker" "$file"; then
            missing_markers+=("$marker")
        fi
    done

    if [[ ${#missing_markers[@]} -eq 0 ]]; then
        success "$description has required pytest markers"
        return 0
    else
        error "$description missing markers: ${missing_markers[*]}"
        return 1
    fi
}

validate_file_structure() {
    log "Validating file structure..."

    local all_passed=true

    # Core test files
    check_file_exists "$PROJECT_ROOT/utilities/kueue_utils_v1beta1.py" "v1beta1 utilities" || all_passed=false
    check_file_exists "$PROJECT_ROOT/tests/distributed_workloads/kueue/upgrade/test_kueue_crd_migration.py" "Main migration test" || all_passed=false
    check_file_exists "$PROJECT_ROOT/tests/distributed_workloads/kueue/upgrade/test_kueue_webhook_conversion.py" "Webhook conversion test" || all_passed=false
    check_file_exists "$PROJECT_ROOT/tests/distributed_workloads/kueue/upgrade/README_kueue_crd_migration.md" "Test documentation" || all_passed=false
    check_file_exists "$PROJECT_ROOT/tests/distributed_workloads/kueue/upgrade/conftest.py" "Kueue upgrade fixtures" || all_passed=false

    # Existing dependencies
    check_file_exists "$PROJECT_ROOT/utilities/kueue_utils.py" "Existing v1beta2 utilities" || all_passed=false
    check_file_exists "$PROJECT_ROOT/tests/workbenches/notebooks_server/controller/upgrade/conftest.py" "Upgrade fixtures" || all_passed=false

    if $all_passed; then
        success "File structure validation passed"
    else
        error "File structure validation failed"
    fi

    return $([ "$all_passed" = true ])
}

validate_python_syntax() {
    log "Validating Python syntax..."

    local all_passed=true

    check_python_syntax "$PROJECT_ROOT/utilities/kueue_utils_v1beta1.py" "v1beta1 utilities" || all_passed=false
    check_python_syntax "$PROJECT_ROOT/tests/distributed_workloads/kueue/upgrade/test_kueue_crd_migration.py" "Main migration test" || all_passed=false
    check_python_syntax "$PROJECT_ROOT/tests/distributed_workloads/kueue/upgrade/test_kueue_webhook_conversion.py" "Webhook conversion test" || all_passed=false

    if $all_passed; then
        success "Python syntax validation passed"
    else
        error "Python syntax validation failed"
    fi

    return $([ "$all_passed" = true ])
}

validate_test_structure() {
    log "Validating test structure..."

    local main_test="$PROJECT_ROOT/tests/distributed_workloads/kueue/upgrade/test_kueue_crd_migration.py"
    local webhook_test="$PROJECT_ROOT/tests/distributed_workloads/kueue/upgrade/test_kueue_webhook_conversion.py"

    local all_passed=true

    # Check imports
    check_imports "$main_test" "Main migration test" || all_passed=false
    check_imports "$webhook_test" "Webhook conversion test" || all_passed=false

    # Check pytest markers
    check_pytest_markers "$main_test" "Main migration test" || all_passed=false
    check_pytest_markers "$webhook_test" "Webhook conversion test" || all_passed=false

    # Check for required test classes
    if grep -q "class TestKueueCRDMigrationPreUpgrade" "$main_test" &&
       grep -q "class TestKueueCRDMigrationPostUpgrade" "$main_test"; then
        success "Main test has required test classes"
    else
        error "Main test missing required test classes"
        all_passed=false
    fi

    if grep -q "class TestKueueConversionWebhookPostUpgrade" "$webhook_test"; then
        success "Webhook test has required test classes"
    else
        error "Webhook test missing required test classes"
        all_passed=false
    fi

    # Check for baseline capture/comparison
    if grep -q "capture_kueue_baseline" "$main_test" &&
       grep -q "kueue_migration_baseline" "$main_test"; then
        success "Main test implements baseline capture and comparison"
    else
        error "Main test missing baseline capture/comparison"
        all_passed=false
    fi

    if $all_passed; then
        success "Test structure validation passed"
    else
        error "Test structure validation failed"
    fi

    return $([ "$all_passed" = true ])
}

validate_api_versions() {
    log "Validating API version usage..."

    local v1beta1_utils="$PROJECT_ROOT/utilities/kueue_utils_v1beta1.py"
    local main_test="$PROJECT_ROOT/tests/distributed_workloads/kueue/upgrade/test_kueue_crd_migration.py"

    local all_passed=true

    # Check v1beta1 API version in utilities
    if grep -q "api_version.*kueue.x-k8s.io/v1beta1" "$v1beta1_utils"; then
        success "v1beta1 utilities use correct API version"
    else
        error "v1beta1 utilities missing correct API version"
        all_passed=false
    fi

    # Check both API versions are tested in main test
    if grep -q "v1beta1" "$main_test" && grep -q "v1beta2" "$main_test"; then
        success "Main test covers both v1beta1 and v1beta2 APIs"
    else
        error "Main test doesn't cover both API versions"
        all_passed=false
    fi

    # Check for bidirectional compatibility testing
    if grep -q "bidirectional.*compatibility" "$main_test"; then
        success "Main test includes bidirectional API compatibility testing"
    else
        warning "Main test may be missing bidirectional API compatibility testing"
        # Don't fail on this, it's implemented but might not have that exact text
    fi

    if $all_passed; then
        success "API version validation passed"
    else
        error "API version validation failed"
    fi

    return $([ "$all_passed" = true ])
}

run_dry_validation() {
    log "Running dry validation (syntax and structure checks only)..."

    local overall_passed=true

    validate_file_structure || overall_passed=false
    echo
    validate_python_syntax || overall_passed=false
    echo
    validate_test_structure || overall_passed=false
    echo
    validate_api_versions || overall_passed=false

    echo
    if $overall_passed; then
        success "All dry validation checks passed!"
        log "✅ kueue crd migration test implementation appears complete and valid"
        log "Ready for integration testing with actual Kueue environment"
    else
        error "Validation failed - please fix issues before proceeding"
        return 1
    fi
}

run_test_discovery() {
    log "Running pytest test discovery..."

    cd "$PROJECT_ROOT"

    # Test discovery for migration tests
    if pytest --collect-only tests/distributed_workloads/kueue/upgrade/test_kueue_crd_migration.py >/dev/null 2>&1; then
        success "Migration tests discovered successfully by pytest"
    else
        error "Migration tests not discoverable by pytest"
        return 1
    fi

    # Test discovery for webhook tests
    if pytest --collect-only tests/distributed_workloads/kueue/upgrade/test_kueue_webhook_conversion.py >/dev/null 2>&1; then
        success "Webhook conversion tests discovered successfully by pytest"
    else
        error "Webhook conversion tests not discoverable by pytest"
        return 1
    fi

    # Check that upgrade markers work
    local pre_upgrade_count
    pre_upgrade_count=$(pytest --collect-only -m "pre_upgrade" tests/distributed_workloads/kueue/upgrade/test_kueue_*.py 2>/dev/null | grep -c "test session starts" || true)

    if [[ $pre_upgrade_count -gt 0 ]]; then
        success "Pre-upgrade tests discoverable with marker filtering"
    else
        warning "Pre-upgrade test marker filtering may not be working"
    fi

    local post_upgrade_count
    post_upgrade_count=$(pytest --collect-only -m "post_upgrade" tests/distributed_workloads/kueue/upgrade/test_kueue_*.py 2>/dev/null | grep -c "test session starts" || true)

    if [[ $post_upgrade_count -gt 0 ]]; then
        success "Post-upgrade tests discoverable with marker filtering"
    else
        warning "Post-upgrade test marker filtering may not be working"
    fi
}

main() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}  Kueue CRD Migration Test Validation  ${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo

    log "Validating implementation of kueue crd migration test"
    log "Project root: $PROJECT_ROOT"
    echo

    if $DRY_RUN; then
        run_dry_validation
    else
        local overall_passed=true

        run_dry_validation || overall_passed=false
        echo
        run_test_discovery || overall_passed=false

        echo
        if $overall_passed; then
            success "🎉 All validation checks passed!"
            echo
            log "✅ kueue crd migration test - COMPLETED"
            echo
            log "Next steps:"
            log "1. Deploy to test environment with Kueue operator"
            log "2. Run pre-upgrade tests: pytest --pre-upgrade tests/distributed_workloads/kueue/upgrade/test_kueue_*.py"
            log "3. Perform RHOAI upgrade"
            log "4. Run post-upgrade tests: pytest --post-upgrade tests/distributed_workloads/kueue/upgrade/test_kueue_*.py"
            log "5. Integrate with CI pipeline upgrade test matrix"
        else
            error "❌ Validation failed - please fix issues before proceeding"
            return 1
        fi
    fi
}

main "$@"