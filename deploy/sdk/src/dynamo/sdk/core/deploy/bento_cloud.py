# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
import re
import sys
import typing as t

from bentoml._internal.cloud.base import Spinner
from bentoml._internal.cloud.deployment import DeploymentConfigParameters
from bentoml._internal.configuration.containers import BentoMLContainer
from bentoml.exceptions import BentoMLException, CLIException
from rich.console import Console

from dynamo.sdk.core.protocol.deployment import Deployment as ProtocolDeployment
from dynamo.sdk.core.protocol.deployment import DeploymentManager, DeploymentStatus

logger = logging.getLogger(__name__)
console = Console(highlight=False)


class BentoCloudDeploymentManager(DeploymentManager):
    """
    Implementation of DeploymentManager that talks to the BentoCloud deployment API.
    Handles all BentoCloud-specific config parameter building, error handling, and API calls.
    Accepts **kwargs for backend-specific options.
    """

    def __init__(self, endpoint: str):
        self.endpoint = endpoint.rstrip("/")
        self.console = console

    def create_deployment(self, deployment: ProtocolDeployment, **kwargs) -> str:
        wait = kwargs.get("wait", True)
        timeout = kwargs.get("timeout", 3600)
        envs = kwargs.get("envs")
        config_file = kwargs.get("config_file")
        pipeline = kwargs.get("pipeline")
        dev = kwargs.get("dev", False)
        # Load config from file and serialize to env
        service_configs = None
        if config_file:
            try:
                service_configs = json.load(config_file)
            except Exception as e:
                logger.error(f"Failed to load config file: {e}")
                raise
        env_dicts = []
        if service_configs:
            config_json = json.dumps(service_configs)
            logger.info(f"Deployment service configuration: {config_json}")
            env_dicts.append({"name": "DYN_DEPLOYMENT_CONFIG", "value": config_json})
        if envs:
            for env in envs:
                if "=" not in env:
                    raise CLIException(f"Invalid env format: {env}. Use KEY=VALUE.")
                key, value = env.split("=", 1)
                env_dicts.append({"name": key, "value": value})
        config_params = DeploymentConfigParameters(
            name=deployment.name,
            bento=pipeline or deployment.namespace,
            envs=env_dicts,
            secrets=None,
            cli=True,
            dev=dev,
        )
        try:
            config_params.verify()
        except BentoMLException as e:
            print(f"Error: {str(e)}")
            sys.exit(1)
        with Spinner(console=self.console) as spinner:
            try:
                spinner.update("Creating deployment on BentoCloud...")
                _cloud_client = BentoMLContainer.bentocloud_client()
                deployment_obj = _cloud_client.deployment.create(
                    deployment_config_params=config_params
                )
                deployment_obj.admin_console = self._get_urls(deployment_obj)
                spinner.log(
                    f':white_check_mark: Created deployment "{deployment_obj.name}" in cluster "{deployment_obj.cluster}"'
                )
                if wait:
                    spinner.log(
                        "[bold blue]Waiting for deployment to be ready, you can use --no-wait to skip this process[/]"
                    )
                    retcode = deployment_obj.wait_until_ready(
                        timeout=timeout, spinner=spinner
                    )
                    if retcode != 0:
                        sys.exit(retcode)
                self._display_deployment_info(spinner, deployment_obj)
                return deployment_obj.name
            except BentoMLException as e:
                error_msg = str(e)
                if "already exists" in error_msg:
                    match = re.search(r'"([^"]+?)(?:\\+)?" already exists', error_msg)
                    dep_name = match.group(1).rstrip("\\") if match else deployment.name
                    spinner.log(
                        "[red]:x: Error:[/] "
                        f'Deployment "{dep_name}" already exists. To create a new deployment:\n'
                        "  1. Use a different name with the --name flag\n"
                        f"  2. Or delete the existing deployment with: dynamo deployment delete {dep_name}"
                    )
                    sys.exit(1)
                spinner.log(f"[red]:x: Error:[/] {str(e)}")
                sys.exit(1)

    def update_deployment(
        self, deployment_id: str, deployment: ProtocolDeployment, **kwargs
    ) -> None:
        # Not implemented for BentoCloud in this example
        raise NotImplementedError

    def get_deployment(self, deployment_id: str, **kwargs) -> dict[str, t.Any]:
        _cloud_client = BentoMLContainer.bentocloud_client()
        with Spinner(console=self.console) as spinner:
            try:
                spinner.update(
                    f'Getting deployment "{deployment_id}" from BentoCloud...'
                )
                deployment_obj = _cloud_client.deployment.get(name=deployment_id)
                spinner.log(
                    f':white_check_mark: Found deployment "{deployment_obj.name}" in cluster "{deployment_obj.cluster}"'
                )
                self._display_deployment_info(spinner, deployment_obj)
                return (
                    deployment_obj.to_dict()
                    if hasattr(deployment_obj, "to_dict")
                    else vars(deployment_obj)
                )
            except BentoMLException as e:
                error_msg = str(e)
                spinner.log(f"[red]:x: Error:[/] Failed to get deployment: {error_msg}")
                sys.exit(1)

    def list_deployments(self, **kwargs) -> list[dict[str, t.Any]]:
        _cloud_client = BentoMLContainer.bentocloud_client()
        with Spinner(console=self.console) as spinner:
            try:
                spinner.update("Getting deployments from BentoCloud...")
                deployments = _cloud_client.deployment.list()
                if not deployments:
                    spinner.log("No deployments found")
                    return []
                spinner.log(":white_check_mark: Found deployments:")
                for deployment_obj in deployments:
                    spinner.log(
                        f"\n• {deployment_obj.name} (cluster: {deployment_obj.cluster})"
                    )
                    self._display_deployment_info(spinner, deployment_obj)
                return [
                    d.to_dict() if hasattr(d, "to_dict") else vars(d)
                    for d in deployments
                ]
            except BentoMLException as e:
                spinner.log(f"[red]:x: Error:[/] Failed to list deployments: {str(e)}")
                sys.exit(1)

    def delete_deployment(self, deployment_id: str, **kwargs) -> None:
        _cloud_client = BentoMLContainer.bentocloud_client()
        with Spinner(console=self.console) as spinner:
            try:
                spinner.update(
                    f'Deleting deployment "{deployment_id}" from BentoCloud...'
                )
                _cloud_client.deployment.delete(name=deployment_id)
                spinner.log(
                    f':white_check_mark: Successfully deleted deployment "{deployment_id}"'
                )
            except BentoMLException as e:
                spinner.log(f"[red]:x: Error:[/] {str(e)}")
                sys.exit(1)

    def get_status(self, deployment_id: str, **kwargs) -> DeploymentStatus:
        dep = self.get_deployment(deployment_id)
        status = dep.get("status", "unknown")
        if status == "running":
            return DeploymentStatus.RUNNING
        elif status == "failed":
            return DeploymentStatus.FAILED
        elif status == "deploying":
            return DeploymentStatus.IN_PROGRESS
        elif status == "terminated":
            return DeploymentStatus.TERMINATED
        else:
            return DeploymentStatus.PENDING

    def wait_until_ready(
        self, deployment_id: str, timeout: int = 3600, **kwargs
    ) -> bool:
        # This is handled by the deployment_obj.wait_until_ready in create_deployment
        # Here, just poll status for generic interface
        import time

        start = time.time()
        while time.time() - start < timeout:
            status = self.get_status(deployment_id)
            if status == DeploymentStatus.RUNNING:
                return True
            elif status == DeploymentStatus.FAILED:
                return False
            time.sleep(5)
        return False

    def get_endpoint_urls(self, deployment_id: str, **kwargs) -> list[str]:
        dep = self.get_deployment(deployment_id)
        return dep.get("urls", [])

    def _get_urls(self, deployment_obj) -> t.List[str]:
        latest = deployment_obj._client.v2.get_deployment(
            deployment_obj.name, deployment_obj.cluster
        )
        urls = latest.urls if hasattr(latest, "urls") else None
        return urls if urls is not None else []

    def _display_deployment_info(self, spinner, deployment_obj) -> None:
        status = (
            getattr(deployment_obj._schema, "status", None)
            if hasattr(deployment_obj, "_schema")
            else None
        )
        reformatted_status = status.replace("[", "\\[") if status else "unknown"
        spinner.log(f"[bold]Status:[/] {reformatted_status}")
        spinner.log("[bold]Ingress URLs:[/]")
        try:
            urls = self._get_urls(deployment_obj)
            if urls:
                for url in urls:
                    spinner.log(f"  - {url}")
            else:
                spinner.log("    No URLs available")
        except Exception:
            if hasattr(deployment_obj, "_urls") and deployment_obj._urls:
                for url in deployment_obj._urls:
                    spinner.log(f"  - {url}")
            else:
                spinner.log("    No URLs available")
