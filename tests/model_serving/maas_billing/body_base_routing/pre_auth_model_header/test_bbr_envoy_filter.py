"""Tests verifying the EnvoyFilter configuration introduced by the BBR pre-auth ext_proc."""

from typing import Any, Self

import pytest
from kubernetes.dynamic import DynamicClient

from tests.model_serving.maas_billing.body_base_routing.pre_auth_model_header.utils import (
    verify_bbr_envoy_filter_cluster_names_contain_gateway_namespace,
    verify_bbr_envoy_filter_has_pre_and_post_auth_stages,
    verify_bbr_post_auth_processing_deployment_ready,
    verify_bbr_post_stage_is_insert_after_wasm_plugin,
    verify_bbr_pre_stage_is_insert_before_wasm_plugin,
)


@pytest.mark.usefixtures("maas_subscription_controller_enabled_latest")
class TestBBREnvoyFilter:
    """Tests for the BBR EnvoyFilter configuration: pre-auth stage, post-auth stage, and gRPC cluster names."""

    @pytest.mark.smoke
    def test_bbr_envoy_filter_has_pre_and_post_auth_stages(
        self: Self,
        bbr_gateway_namespace: str,
        bbr_envoy_filter_config_patches: list[Any],
    ) -> None:
        """Verify the BBR EnvoyFilter contains both the pre-auth (bbr-pre) and post-auth (bbr) filter patches."""
        verify_bbr_envoy_filter_has_pre_and_post_auth_stages(
            config_patches=bbr_envoy_filter_config_patches,
            gateway_namespace=bbr_gateway_namespace,
        )

    @pytest.mark.tier1
    def test_bbr_pre_stage_is_insert_before_wasm_plugin(
        self: Self,
        bbr_envoy_filter_config_patches: list[Any],
    ) -> None:
        """Verify the bbr-pre configPatch uses INSERT_BEFORE so body extraction runs before the WasmPlugin."""
        verify_bbr_pre_stage_is_insert_before_wasm_plugin(
            config_patches=bbr_envoy_filter_config_patches,
        )

    @pytest.mark.tier1
    def test_bbr_post_stage_is_insert_after_wasm_plugin(
        self: Self,
        bbr_envoy_filter_config_patches: list[Any],
    ) -> None:
        """Verify the bbr post-auth configPatch uses INSERT_AFTER so usage tracking runs after the WasmPlugin."""
        verify_bbr_post_stage_is_insert_after_wasm_plugin(
            config_patches=bbr_envoy_filter_config_patches,
        )

    @pytest.mark.tier1
    def test_bbr_envoy_filter_cluster_names_resolve_to_gateway_namespace(
        self: Self,
        bbr_gateway_namespace: str,
        bbr_envoy_filter_config_patches: list[Any],
    ) -> None:
        """Verify all gRPC cluster names in the BBR EnvoyFilter point to services in the gateway namespace."""
        verify_bbr_envoy_filter_cluster_names_contain_gateway_namespace(
            config_patches=bbr_envoy_filter_config_patches,
            gateway_namespace=bbr_gateway_namespace,
        )

    @pytest.mark.smoke
    def test_bbr_post_auth_processing_deployment_ready(
        self: Self,
        admin_client: DynamicClient,
        bbr_gateway_namespace: str,
    ) -> None:
        """Verify the post-auth payload-processing Deployment exists and all replicas are ready."""
        verify_bbr_post_auth_processing_deployment_ready(
            admin_client=admin_client,
            gateway_namespace=bbr_gateway_namespace,
        )
