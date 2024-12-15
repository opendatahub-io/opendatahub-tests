import shlex
from typing import Dict

from kubernetes.dynamic import DynamicClient
from ocp_resources.pod import Pod

from tests.model_serving.model_server.utils import b64_encoded_string


def get_s3_secret_dict(
    aws_access_key: str,
    aws_secret_access_key: str,
    aws_s3_bucket: str,
    aws_s3_endpoint: str,
    aws_s3_region: str,
) -> Dict[str, str]:
    return {
        "AWS_ACCESS_KEY_ID": b64_encoded_string(string_to_encode=aws_access_key),
        "AWS_SECRET_ACCESS_KEY": b64_encoded_string(string_to_encode=aws_secret_access_key),
        "AWS_S3_BUCKET": b64_encoded_string(string_to_encode=aws_s3_bucket),
        "AWS_S3_ENDPOINT": b64_encoded_string(string_to_encode=aws_s3_endpoint),
        "AWS_DEFAULT_REGION": b64_encoded_string(string_to_encode=aws_s3_region),
    }


def download_model_data(
    admin_client: DynamicClient,
    aws_access_key_id: str,
    aws_secret_access_key: str,
    storage_uri: str,
    model_namespace: str,
    model_pvc_name: str,
) -> str:
    mount_path: str = "data"
    model_dir: str = "model-dir"
    containers = [
        {
            "name": "model-downloader",
            "image": "quay.io/redhat_msi/qe-tools-base-image",
            "args": [
                "sh",
                "-c",
                "sleep infinity",
            ],
            "env": [
                {"name": "AWS_ACCESS_KEY_ID", "value": aws_access_key_id},
                {"name": "AWS_SECRET_ACCESS_KEY", "value": aws_secret_access_key},
            ],
            "volumeMounts": [{"mountPath": mount_path, "name": model_pvc_name, "subPath": model_dir}],
        }
    ]
    volumes = [{"name": model_pvc_name, "persistentVolumeClaim": {"claimName": model_pvc_name}}]
    with Pod(
        client=admin_client,
        namespace=model_namespace,
        name="download-model-data",
        containers=containers,
        volumes=volumes,
    ) as pod:
        pod.wait_for_status(status=Pod.Status.RUNNING)
        pod.execute(command=shlex.split(f"aws s3 cp --recursive {storage_uri} /{mount_path}/{model_dir}"))

    return model_dir
