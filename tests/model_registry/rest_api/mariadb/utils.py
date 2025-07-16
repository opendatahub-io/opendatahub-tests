from typing import Any


def get_mariadb_dict(base_name: str) -> dict[str, Any]:
    return {
        "connection": {"secretName": f"{base_name}conn", "secretTemplate": {"key": "dsn"}},
        "database": f"{base_name}db".replace("-", "_"),
        "primary_connection": {"secretName": f"{base_name}-conn-primary", "secretTemplate": {"key": "dsn"}},
        "primary_service": {"type": "ClusterIP"},
        "port": 3306,
        "root_password_secret_key_ref": {"generate": True, "key": "password", "name": f"{base_name}-root"},
        "galera": {"enabled": False},
        "metrics": {"enabled": False},
        "replicas": 1,
        "my_cnf": "[mariadb]\nbind-address=*\ndefault_storage_engine=InnoDB"
        "\nbinlog_format=row\ninnodb_autoinc_lock_mode=2"
        "\ninnodb_buffer_pool_size=1024M\nmax_allowed_packet=256M\n",  # pragma: allowlist secret
        "secondary_connection": {"secretName": f"{base_name}-conn-secondary", "secretTemplate": {"key": "dsn"}},
        "secondary_service": {"type": "ClusterIP"},
        "service": {"type": "ClusterIP"},
        "username": f"{base_name}user",
        "password_secret_key_ref": {"generate": True, "key": "password", "name": f"{base_name}-password"},
        "storage": {"size": "1Gi"},
        "update_strategy": {"type": "ReplicasFirstPrimaryLast"},
    }
