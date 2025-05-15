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

import typing as t

import requests

from dynamo.sdk.core.protocol.deployment import (
    Deployment,
    DeploymentManager,
    DeploymentStatus,
)


class KubernetesDeploymentManager(DeploymentManager):
    """
    Implementation of DeploymentManager that talks to the dynamo_store deployment API.
    """

    def __init__(self, endpoint: str):
        self.endpoint = endpoint.rstrip("/")
        self.session = requests.Session()
        self.namespace = "default"

    def create_deployment(self, deployment: Deployment) -> str:
        url = f"{self.endpoint}/api/v2/deployments"
        # Map Deployment dataclass to CreateDeploymentSchema
        payload = {
            "name": deployment.name,
            "dynamo": deployment.namespace,  # This should reference the dynamo component or graph
            # TODO: Map envs, services, etc. as needed
        }
        resp = self.session.post(url, json=payload)
        resp.raise_for_status()
        return resp.json()["name"]

    def update_deployment(self, deployment_id: str, deployment: Deployment) -> None:
        url = f"{self.endpoint}/api/v2/deployments/{deployment_id}"
        payload = {
            "name": deployment.name,
            "dynamo": deployment.namespace,
            # TODO: Map envs, services, etc. as needed
        }
        resp = self.session.put(url, json=payload)
        resp.raise_for_status()

    def get_deployment(self, deployment_id: str) -> dict[str, t.Any]:
        url = f"{self.endpoint}/api/v2/deployments/{deployment_id}"
        resp = self.session.get(url)
        resp.raise_for_status()
        return resp.json()

    def list_deployments(self) -> list[dict[str, t.Any]]:
        url = f"{self.endpoint}/api/v2/deployments"
        resp = self.session.get(url)
        resp.raise_for_status()
        data = resp.json()
        return data.get("items", [])

    def delete_deployment(self, deployment_id: str) -> None:
        url = f"{self.endpoint}/api/v2/deployments/{deployment_id}"
        resp = self.session.delete(url)
        resp.raise_for_status()

    def get_status(self, deployment_id: str) -> DeploymentStatus:
        dep = self.get_deployment(deployment_id)
        status = dep.get("status", "unknown")
        # Map API status to DeploymentStatus enum
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

    def wait_until_ready(self, deployment_id: str, timeout: int = 3600) -> bool:
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

    def get_endpoint_urls(self, deployment_id: str) -> list[str]:
        dep = self.get_deployment(deployment_id)
        return dep.get("urls", [])
