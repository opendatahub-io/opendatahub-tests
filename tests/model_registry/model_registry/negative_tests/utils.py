import base64

from ocp_resources.pod import Pod

from tests.model_registry.constants import MODEL_REGISTRY_DB_SECRET_STR_DATA


def create_mysql_credentials_file(model_registry_db_instance_pod: Pod) -> None:
    """
    Setup MySQL configuration file with credentials.
    """
    credentials_file_content = f"""[client]
    user={MODEL_REGISTRY_DB_SECRET_STR_DATA["database-user"]}
    password={MODEL_REGISTRY_DB_SECRET_STR_DATA["database-password"]}
    """
    b64_content = base64.b64encode(credentials_file_content.encode("utf-8")).decode("utf-8")

    model_registry_db_instance_pod.execute(
        command=["bash", "-c", f"echo '{b64_content}' | base64 --decode > /tmp/.my.cnf"]
    )


def execute_mysql_command(sql_query: str, model_registry_db_instance_pod: Pod) -> str:
    """
    Execute a MySQL command on the model registry database instance pod.
    """
    return model_registry_db_instance_pod.execute(
        command=[
            "mysql",
            "--defaults-file=/tmp/.my.cnf",
            "-e",
            sql_query,
            MODEL_REGISTRY_DB_SECRET_STR_DATA["database-name"],
        ]
    )
