"""Deployment names created by one DataSciencePipelinesApplication named dspa."""

DSPA_NAME: str = "dspa"
OBJECT_STORAGE_SECRET_NAME: str = "dspa-object-storage"
OPERATOR_DEPLOYMENT_NAME: str = "data-science-pipelines-operator-controller-manager"
OPERATOR_LABEL_SELECTOR: str = "app.kubernetes.io/name=data-science-pipelines-operator"
ISTIO_PROXY_CONTAINER_NAME: str = "istio-proxy"

PIPELINE_SERVER_DEPLOYMENTS: tuple[str, ...] = (
    f"ds-pipeline-{DSPA_NAME}",
    f"ds-pipeline-persistenceagent-{DSPA_NAME}",
    f"ds-pipeline-scheduledworkflow-{DSPA_NAME}",
    f"ds-pipeline-workflow-controller-{DSPA_NAME}",
    f"ds-pipeline-metadata-grpc-{DSPA_NAME}",
    f"ds-pipeline-metadata-envoy-{DSPA_NAME}",
    f"mariadb-{DSPA_NAME}",
)
