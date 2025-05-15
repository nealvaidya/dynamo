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
import typing as t

from bentoml._internal.cloud.deployment import DeploymentConfigParameters
from bentoml._internal.configuration.containers import BentoMLContainer
from bentoml.exceptions import BentoMLException
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
    Raises exceptions for errors; CLI handles user interaction.
    """

    def __init__(self, endpoint: str):
        self.endpoint = endpoint.rstrip("/")

    def create_deployment(self, deployment: ProtocolDeployment, **kwargs) -> str:
        wait = kwargs.get("wait", True)
        timeout = kwargs.get("timeout", 3600)
        envs = kwargs.get("envs")
        config_file = kwargs.get("config_file")
        pipeline = kwargs.get("pipeline")
        dev = kwargs.get("dev", False)
        # args = kwargs.get("args")
        service_configs = None
        if config_file:
            try:
                service_configs = json.load(config_file)
            except Exception as e:
                raise RuntimeError((400, f"Failed to load config file: {e}", None))
        env_dicts = []
        if service_configs:
            config_json = json.dumps(service_configs)
            env_dicts.append({"name": "DYN_DEPLOYMENT_CONFIG", "value": config_json})
        if envs:
            for env in envs:
                if "=" not in env:
                    raise RuntimeError(
                        (400, f"Invalid env format: {env}. Use KEY=VALUE.", None)
                    )
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
            raise RuntimeError((400, f"Config verification error: {str(e)}", None))
        try:
            _cloud_client = BentoMLContainer.bentocloud_client()
            deployment_obj = _cloud_client.deployment.create(
                deployment_config_params=config_params
            )
            if wait:
                retcode = deployment_obj.wait_until_ready(timeout=timeout)
                if retcode != 0:
                    raise RuntimeError(
                        (500, "Deployment did not become ready in time.", None)
                    )
            return deployment_obj.name
        except BentoMLException as e:
            error_msg = str(e)
            if "already exists" in error_msg:
                raise RuntimeError((409, error_msg, None))
            raise RuntimeError((500, error_msg, None))

    def update_deployment(
        self, deployment_id: str, deployment: ProtocolDeployment, **kwargs
    ) -> None:
        raise NotImplementedError

    def get_deployment(self, deployment_id: str, **kwargs) -> dict[str, t.Any]:
        try:
            _cloud_client = BentoMLContainer.bentocloud_client()
            deployment_obj = _cloud_client.deployment.get(name=deployment_id)
            return (
                deployment_obj.to_dict()
                if hasattr(deployment_obj, "to_dict")
                else vars(deployment_obj)
            )
        except BentoMLException as e:
            error_msg = str(e)
            raise RuntimeError((404, error_msg, None))

    def list_deployments(self, **kwargs) -> list[dict[str, t.Any]]:
        try:
            _cloud_client = BentoMLContainer.bentocloud_client()
            deployments = _cloud_client.deployment.list()
            return [
                d.to_dict() if hasattr(d, "to_dict") else vars(d) for d in deployments
            ]
        except BentoMLException as e:
            error_msg = str(e)
            raise RuntimeError((500, error_msg, None))

    def delete_deployment(self, deployment_id: str, **kwargs) -> None:
        try:
            _cloud_client = BentoMLContainer.bentocloud_client()
            _cloud_client.deployment.delete(name=deployment_id)
        except BentoMLException as e:
            error_msg = str(e)
            raise RuntimeError((404, error_msg, None))

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
