# Generated using https://github.com/RedHatQE/openshift-python-wrapper/blob/main/scripts/resource/README.md


from typing import Any

from ocp_resources.exceptions import MissingRequiredArgumentError
from ocp_resources.resource import NamespacedResource

from utilities.constants import ApiGroups


class SparkApplication(NamespacedResource):
    """
    SparkApplication is the Schema for the sparkapplications API
    """

    api_group: str = ApiGroups.SPARKOPERATOR_K8S_IO

    def __init__(
        self,
        arguments: list[Any] | None = None,
        batch_scheduler: str | None = None,
        batch_scheduler_options: dict[str, Any] | None = None,
        deps: dict[str, Any] | None = None,
        driver: dict[str, Any] | None = None,
        driver_ingress_options: list[Any] | None = None,
        dynamic_allocation: dict[str, Any] | None = None,
        executor: dict[str, Any] | None = None,
        failure_retries: int | None = None,
        hadoop_conf: dict[str, Any] | None = None,
        hadoop_config_map: str | None = None,
        image: str | None = None,
        image_pull_policy: str | None = None,
        image_pull_secrets: list[Any] | None = None,
        main_application_file: str | None = None,
        main_class: str | None = None,
        memory_overhead_factor: str | None = None,
        mode: str | None = None,
        monitoring: dict[str, Any] | None = None,
        node_selector: dict[str, Any] | None = None,
        proxy_user: str | None = None,
        python_version: str | None = None,
        restart_policy: dict[str, Any] | None = None,
        retry_interval: int | None = None,
        spark_conf: dict[str, Any] | None = None,
        spark_config_map: str | None = None,
        spark_ui_options: dict[str, Any] | None = None,
        spark_version: str | None = None,
        suspend: bool | None = None,
        time_to_live_seconds: int | None = None,
        type: str | None = None,
        volumes: list[Any] | None = None,
        **kwargs: Any,
    ) -> None:
        r"""
        Args:
            arguments (list[Any]): Arguments is a list of arguments to be passed to the application.

            batch_scheduler (str): BatchScheduler configures which batch scheduler will be used for
              scheduling

            batch_scheduler_options (dict[str, Any]): BatchSchedulerOptions provides fine-grained
              control on how to batch scheduling.

            deps (dict[str, Any]): Deps captures all possible types of dependencies of a Spark
              application.

            driver (dict[str, Any]): Driver is the driver specification.

            driver_ingress_options (list[Any]): DriverIngressOptions allows configuring the Service and the Ingress to
              expose ports inside Spark Driver

            dynamic_allocation (dict[str, Any]): DynamicAllocation configures dynamic allocation that becomes available
              for the Kubernetes scheduler backend since Spark 3.0.

            executor (dict[str, Any]): Executor is the executor specification.

            failure_retries (int): FailureRetries is the number of times to retry a failed application
              before giving up. This is best effort and actual retry attempts
              can be >= the value specified.

            hadoop_conf (dict[str, Any]): HadoopConf carries user-specified Hadoop configuration properties as
              they would use the "--conf" option in spark-submit. The
              SparkApplication controller automatically adds prefix
              "spark.hadoop." to Hadoop configuration properties.

            hadoop_config_map (str): HadoopConfigMap carries the name of the ConfigMap containing Hadoop
              configuration files such as core-site.xml. The controller will add
              environment variable HADOOP_CONF_DIR to the path where the
              ConfigMap is mounted to.

            image (str): Image is the container image for the driver, executor, and init-
              container. Any custom container images for the driver, executor,
              or init-container takes precedence over this.

            image_pull_policy (str): ImagePullPolicy is the image pull policy for the driver, executor, and
              init-container.

            image_pull_secrets (list[Any]): ImagePullSecrets is the list of image-pull secrets.

            main_application_file (str): MainFile is the path to a bundled JAR, Python, or R file of the
              application.

            main_class (str): MainClass is the fully-qualified main class of the Spark application.
              This only applies to Java/Scala Spark applications.

            memory_overhead_factor (str): This sets the Memory Overhead Factor that will allocate memory to non-
              JVM memory. For JVM-based jobs this value will default to 0.10,
              for non-JVM jobs 0.40. Value of this field will be overridden by
              `Spec.Driver.MemoryOverhead` and `Spec.Executor.MemoryOverhead` if
              they are set.

            mode (str): Mode is the deployment mode of the Spark application.

            monitoring (dict[str, Any]): Monitoring configures how monitoring is handled.

            node_selector (dict[str, Any]): NodeSelector is the Kubernetes node selector to be added to the driver
              and executor pods. This field is mutually exclusive with
              nodeSelector at podSpec level (driver or executor). This field
              will be deprecated in future versions (at SparkApplicationSpec
              level).

            proxy_user (str): ProxyUser specifies the user to impersonate when submitting the
              application. It maps to the command-line flag "--proxy-user" in
              spark-submit.

            python_version (str): This sets the major Python version of the docker image used to run the
              driver and executor containers. Can either be 2 or 3, default 2.

            restart_policy (dict[str, Any]): RestartPolicy defines the policy on if and in which conditions the
              controller should restart an application.

            retry_interval (int): RetryInterval is the unit of intervals in seconds between submission
              retries.

            spark_conf (dict[str, Any]): SparkConf carries user-specified Spark configuration properties as
              they would use the  "--conf" option in spark-submit.

            spark_config_map (str): SparkConfigMap carries the name of the ConfigMap containing Spark
              configuration files such as log4j.properties. The controller will
              add environment variable SPARK_CONF_DIR to the path where the
              ConfigMap is mounted to.

            spark_ui_options (dict[str, Any]): SparkUIOptions allows configuring the Service and the Ingress to
              expose the sparkUI

            spark_version (str): SparkVersion is the version of Spark the application uses.

            suspend (bool): Suspend indicates whether the SparkApplication should be suspended.
              When true, the controller skips submitting the Spark job. If a
              SparkApplication is suspended after creation (i.e. the flag goes
              from false to true), the Spark operator will delete all active
              Pods associated with this SparkApplication. Users must design
              their Spark application to gracefully handle this.

            time_to_live_seconds (int): TimeToLiveSeconds defines the Time-To-Live (TTL) duration in seconds
              for this SparkApplication after its termination. The
              SparkApplication object will be garbage collected if the current
              time is more than the TimeToLiveSeconds since its termination.

            type (str): Type tells the type of the Spark application.

            volumes (list[Any]): Volumes is the list of Kubernetes volumes that can be mounted by the
              driver and/or executors.

        """
        super().__init__(**kwargs)

        self.arguments = arguments
        self.batch_scheduler = batch_scheduler
        self.batch_scheduler_options = batch_scheduler_options
        self.deps = deps
        self.driver = driver
        self.driver_ingress_options = driver_ingress_options
        self.dynamic_allocation = dynamic_allocation
        self.executor = executor
        self.failure_retries = failure_retries
        self.hadoop_conf = hadoop_conf
        self.hadoop_config_map = hadoop_config_map
        self.image = image
        self.image_pull_policy = image_pull_policy
        self.image_pull_secrets = image_pull_secrets
        self.main_application_file = main_application_file
        self.main_class = main_class
        self.memory_overhead_factor = memory_overhead_factor
        self.mode = mode
        self.monitoring = monitoring
        self.node_selector = node_selector
        self.proxy_user = proxy_user
        self.python_version = python_version
        self.restart_policy = restart_policy
        self.retry_interval = retry_interval
        self.spark_conf = spark_conf
        self.spark_config_map = spark_config_map
        self.spark_ui_options = spark_ui_options
        self.spark_version = spark_version
        self.suspend = suspend
        self.time_to_live_seconds = time_to_live_seconds
        self.type = type
        self.volumes = volumes

    def to_dict(self) -> None:

        super().to_dict()

        if not self.kind_dict and not self.yaml_file:
            if self.driver is None:
                raise MissingRequiredArgumentError(argument="self.driver")

            if self.executor is None:
                raise MissingRequiredArgumentError(argument="self.executor")

            if self.main_application_file is None:
                raise MissingRequiredArgumentError(argument="self.main_application_file")

            if self.spark_version is None:
                raise MissingRequiredArgumentError(argument="self.spark_version")

            if self.type is None:
                raise MissingRequiredArgumentError(argument="self.type")

            self.res["spec"] = {}
            _spec = self.res["spec"]

            _spec["driver"] = self.driver
            _spec["executor"] = self.executor
            _spec["mainApplicationFile"] = self.main_application_file
            _spec["sparkVersion"] = self.spark_version
            _spec["type"] = self.type

            if self.arguments is not None:
                _spec["arguments"] = self.arguments

            if self.batch_scheduler is not None:
                _spec["batchScheduler"] = self.batch_scheduler

            if self.batch_scheduler_options is not None:
                _spec["batchSchedulerOptions"] = self.batch_scheduler_options

            if self.deps is not None:
                _spec["deps"] = self.deps

            if self.driver_ingress_options is not None:
                _spec["driverIngressOptions"] = self.driver_ingress_options

            if self.dynamic_allocation is not None:
                _spec["dynamicAllocation"] = self.dynamic_allocation

            if self.failure_retries is not None:
                _spec["failureRetries"] = self.failure_retries

            if self.hadoop_conf is not None:
                _spec["hadoopConf"] = self.hadoop_conf

            if self.hadoop_config_map is not None:
                _spec["hadoopConfigMap"] = self.hadoop_config_map

            if self.image is not None:
                _spec["image"] = self.image

            if self.image_pull_policy is not None:
                _spec["imagePullPolicy"] = self.image_pull_policy

            if self.image_pull_secrets is not None:
                _spec["imagePullSecrets"] = self.image_pull_secrets

            if self.main_class is not None:
                _spec["mainClass"] = self.main_class

            if self.memory_overhead_factor is not None:
                _spec["memoryOverheadFactor"] = self.memory_overhead_factor

            if self.mode is not None:
                _spec["mode"] = self.mode

            if self.monitoring is not None:
                _spec["monitoring"] = self.monitoring

            if self.node_selector is not None:
                _spec["nodeSelector"] = self.node_selector

            if self.proxy_user is not None:
                _spec["proxyUser"] = self.proxy_user

            if self.python_version is not None:
                _spec["pythonVersion"] = self.python_version

            if self.restart_policy is not None:
                _spec["restartPolicy"] = self.restart_policy

            if self.retry_interval is not None:
                _spec["retryInterval"] = self.retry_interval

            if self.spark_conf is not None:
                _spec["sparkConf"] = self.spark_conf

            if self.spark_config_map is not None:
                _spec["sparkConfigMap"] = self.spark_config_map

            if self.spark_ui_options is not None:
                _spec["sparkUIOptions"] = self.spark_ui_options

            if self.suspend is not None:
                _spec["suspend"] = self.suspend

            if self.time_to_live_seconds is not None:
                _spec["timeToLiveSeconds"] = self.time_to_live_seconds

            if self.volumes is not None:
                _spec["volumes"] = self.volumes

    # End of generated code
