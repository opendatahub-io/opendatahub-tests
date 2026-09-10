# Generated using https://github.com/RedHatQE/openshift-python-wrapper/blob/main/class_generator/README.md


from typing import Any

from ocp_resources.resource import Resource


class ClusterPolicy(Resource):
    """
    ClusterPolicy declares validation, mutation, and generation behaviors for matching resources.
    """

    api_group: str = "kyverno.io"

    def __init__(
        self,
        admission: bool | None = None,
        apply_rules: str | None = None,
        background: bool | None = None,
        failure_policy: str | None = None,
        generate_existing: bool | None = None,
        generate_existing_on_policy_update: bool | None = None,
        mutate_existing_on_policy_update: bool | None = None,
        rules: list[Any] | None = None,
        schema_validation: bool | None = None,
        use_server_side_apply: bool | None = None,
        validation_failure_action: str | None = None,
        validation_failure_action_overrides: list[Any] | None = None,
        webhook_configuration: dict[str, Any] | None = None,
        webhook_timeout_seconds: int | None = None,
        **kwargs: Any,
    ) -> None:
        r"""
        Args:
            admission (bool): Admission controls if rules are applied during admission. Optional.
              Default value is "true".

            apply_rules (str): ApplyRules controls how rules in a policy are applied. Rule are
              processed in the order of declaration. When set to `One`
              processing stops after a rule has been applied i.e. the rule
              matches and results in a pass, fail, or error. When set to `All`
              all rules in the policy are processed. The default is `All`.

            background (bool): Background controls if rules are applied to existing resources during
              a background scan. Optional. Default value is "true". The value
              must be set to "false" if the policy rule uses variables that are
              only available in the admission review request (e.g. user name).

            failure_policy (str): FailurePolicy defines how unexpected policy errors and webhook
              response timeout errors are handled. Rules within the same policy
              share the same failure behavior. This field should not be accessed
              directly, instead `GetFailurePolicy()` should be used. Allowed
              values are Ignore or Fail. Defaults to Fail.

            generate_existing (bool): GenerateExisting controls whether to trigger generate rule in existing
              resources If is set to "true" generate rule will be triggered and
              applied to existing matched resources. Defaults to "false" if not
              specified.

            generate_existing_on_policy_update (bool): Deprecated, use generateExisting instead

            mutate_existing_on_policy_update (bool): MutateExistingOnPolicyUpdate controls if a mutateExisting policy is
              applied on policy events. Default value is "false".

            rules (list[Any]): Rules is a list of Rule instances. A Policy contains multiple rules
              and each rule can validate, mutate, or generate resources.

            schema_validation (bool): Deprecated.

            use_server_side_apply (bool): UseServerSideApply controls whether to use server-side apply for
              generate rules If is set to "true" create & update for generate
              rules will use apply instead of create/update. Defaults to "false"
              if not specified.

            validation_failure_action (str): ValidationFailureAction defines if a validation policy rule violation
              should block the admission review request (enforce), or allow
              (audit) the admission review request and report an error in a
              policy report. Optional. Allowed values are audit or enforce. The
              default value is "Audit".

            validation_failure_action_overrides (list[Any]): ValidationFailureActionOverrides is a Cluster Policy
              attribute that specifies ValidationFailureAction namespace-wise. It overrides ValidationFailureAction
              for the specified namespaces.

            webhook_configuration (dict[str, Any]): WebhookConfiguration specifies the custom configuration for
              Kubernetes admission webhookconfiguration. Requires Kubernetes 1.27 or later.

            webhook_timeout_seconds (int): WebhookTimeoutSeconds specifies the maximum time in seconds allowed to
              apply this policy. After the configured time expires, the
              admission request may fail, or may simply ignore the policy
              results, based on the failure policy. The default timeout is 10s,
              the value must be between 1 and 30 seconds.

        """
        super().__init__(**kwargs)

        self.admission = admission
        self.apply_rules = apply_rules
        self.background = background
        self.failure_policy = failure_policy
        self.generate_existing = generate_existing
        self.generate_existing_on_policy_update = generate_existing_on_policy_update
        self.mutate_existing_on_policy_update = mutate_existing_on_policy_update
        self.rules = rules
        self.schema_validation = schema_validation
        self.use_server_side_apply = use_server_side_apply
        self.validation_failure_action = validation_failure_action
        self.validation_failure_action_overrides = validation_failure_action_overrides
        self.webhook_configuration = webhook_configuration
        self.webhook_timeout_seconds = webhook_timeout_seconds

    def to_dict(self) -> None:

        super().to_dict()

        if not self.kind_dict and not self.yaml_file:
            self.res["spec"] = {}
            _spec = self.res["spec"]

            if self.admission is not None:
                _spec["admission"] = self.admission

            if self.apply_rules is not None:
                _spec["applyRules"] = self.apply_rules

            if self.background is not None:
                _spec["background"] = self.background

            if self.failure_policy is not None:
                _spec["failurePolicy"] = self.failure_policy

            if self.generate_existing is not None:
                _spec["generateExisting"] = self.generate_existing

            if self.generate_existing_on_policy_update is not None:
                _spec["generateExistingOnPolicyUpdate"] = self.generate_existing_on_policy_update

            if self.mutate_existing_on_policy_update is not None:
                _spec["mutateExistingOnPolicyUpdate"] = self.mutate_existing_on_policy_update

            if self.rules is not None:
                _spec["rules"] = self.rules

            if self.schema_validation is not None:
                _spec["schemaValidation"] = self.schema_validation

            if self.use_server_side_apply is not None:
                _spec["useServerSideApply"] = self.use_server_side_apply

            if self.validation_failure_action is not None:
                _spec["validationFailureAction"] = self.validation_failure_action

            if self.validation_failure_action_overrides is not None:
                _spec["validationFailureActionOverrides"] = self.validation_failure_action_overrides

            if self.webhook_configuration is not None:
                _spec["webhookConfiguration"] = self.webhook_configuration

            if self.webhook_timeout_seconds is not None:
                _spec["webhookTimeoutSeconds"] = self.webhook_timeout_seconds

    # End of generated code
