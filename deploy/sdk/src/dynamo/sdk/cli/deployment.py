#  SPDX-FileCopyrightText: Copyright (c) 2020 Atalaya Tech. Inc
#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0
#  #
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#  http://www.apache.org/licenses/LICENSE-2.0
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  Modifications Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES

from __future__ import annotations

import typing as t

import typer
from rich.console import Console

from dynamo.sdk.core.deploy.bento_cloud import BentoCloudDeploymentManager
from dynamo.sdk.core.deploy.kubernetes import KubernetesDeploymentManager
from dynamo.sdk.core.protocol.deployment import Deployment

app = typer.Typer(
    help="Deploy Dynamo applications to Dynamo Cloud Kubernetes Platform",
    add_completion=True,
    no_args_is_help=True,
)

console = Console(highlight=False)


def get_deployment_manager(target: str, endpoint: str):
    """Return the appropriate DeploymentManager for the given target and endpoint."""
    if target == "kubernetes":
        return KubernetesDeploymentManager(endpoint)
    elif target == "bento_cloud":
        return BentoCloudDeploymentManager(endpoint)
    else:
        raise ValueError(f"Unknown deployment target: {target}")


@app.command()
def create(
    ctx: typer.Context,
    pipeline: t.Optional[str] = typer.Argument(None, help="Dynamo pipeline to deploy"),
    name: t.Optional[str] = typer.Option(None, "--name", "-n", help="Deployment name"),
    config_file: t.Optional[typer.FileText] = typer.Option(
        None, "--config-file", "-f", help="Configuration file path"
    ),
    wait: bool = typer.Option(
        True, "--wait/--no-wait", help="Do not wait for deployment to be ready"
    ),
    timeout: int = typer.Option(
        3600, "--timeout", help="Timeout for deployment to be ready in seconds"
    ),
    endpoint: str = typer.Option(
        ..., "--endpoint", "-e", help="Dynamo Cloud endpoint", envvar="DYNAMO_CLOUD"
    ),
    envs: t.Optional[t.List[str]] = typer.Option(
        None,
        "--env",
        help="Environment variable(s) to set (format: KEY=VALUE). Note: These environment variables will be set on ALL services in your Dynamo pipeline.",
    ),
    target: str = typer.Option(..., "--target", "-t", help="Deployment target"),
    dev: bool = typer.Option(False, "--dev", help="Development mode for deployment"),
) -> None:
    """Create a deployment on Dynamo Cloud."""
    deployment_manager = get_deployment_manager(target, endpoint)
    deployment = Deployment(
        name=name or (pipeline if pipeline else "unnamed-deployment"),
        namespace="default",
        services=[],
    )
    deployment_manager.create_deployment(
        deployment,
        wait=wait,
        timeout=timeout,
        envs=envs,
        config_file=config_file,
        pipeline=pipeline,
        dev=dev,
        args=ctx.args if hasattr(ctx, "args") else None,
    )


@app.command()
def deploy(
    ctx: typer.Context,
    pipeline: t.Optional[str] = typer.Argument(None, help="Dynamo pipeline to deploy"),
    name: t.Optional[str] = typer.Option(None, "--name", "-n", help="Deployment name"),
    config_file: t.Optional[typer.FileText] = typer.Option(
        None, "--config-file", "-f", help="Configuration file path"
    ),
    wait: bool = typer.Option(
        True, "--wait/--no-wait", help="Do not wait for deployment to be ready"
    ),
    timeout: int = typer.Option(
        3600, "--timeout", help="Timeout for deployment to be ready in seconds"
    ),
    endpoint: str = typer.Option(
        ..., "--endpoint", "-e", help="Dynamo Cloud endpoint", envvar="DYNAMO_CLOUD"
    ),
    envs: t.Optional[t.List[str]] = typer.Option(
        None,
        "--env",
        help="Environment variable(s) to set (format: KEY=VALUE). Note: These environment variables will be set on ALL services in your Dynamo pipeline.",
    ),
    target: str = typer.Option(..., "--target", "-t", help="Deployment target"),
    dev: bool = typer.Option(False, "--dev", help="Development mode for deployment"),
) -> None:
    """Create a deployment on Dynamo Cloud."""
    deployment_manager = get_deployment_manager(target, endpoint)
    deployment = Deployment(
        name=name or (pipeline if pipeline else "unnamed-deployment"),
        namespace="default",
        services=[],
    )
    deployment_manager.create_deployment(
        deployment,
        wait=wait,
        timeout=timeout,
        envs=envs,
        config_file=config_file,
        pipeline=pipeline,
        dev=dev,
        args=ctx.args if hasattr(ctx, "args") else None,
    )


@app.command()
def get(
    name: str = typer.Argument(..., help="Deployment name"),
    target: str = typer.Option(..., "--target", "-t", help="Deployment target"),
    endpoint: str = typer.Option(
        ..., "--endpoint", "-e", help="Dynamo Cloud endpoint", envvar="DYNAMO_CLOUD"
    ),
) -> None:
    """Get details for a specific deployment by name."""
    deployment_manager = get_deployment_manager(target, endpoint)
    deployment = deployment_manager.get_deployment(name)
    console.print(deployment)


@app.command("list")
def list_deployments(
    target: str = typer.Option(..., "--target", "-t", help="Deployment target"),
    endpoint: str = typer.Option(
        ..., "--endpoint", "-e", help="Dynamo Cloud endpoint", envvar="DYNAMO_CLOUD"
    ),
) -> None:
    """List all deployments."""
    deployment_manager = get_deployment_manager(target, endpoint)
    deployments = deployment_manager.list_deployments()
    for dep in deployments:
        console.print(dep)


@app.command()
def delete(
    name: str = typer.Argument(..., help="Deployment name"),
    target: str = typer.Option(..., "--target", "-t", help="Deployment target"),
    endpoint: str = typer.Option(
        ..., "--endpoint", "-e", help="Dynamo Cloud endpoint", envvar="DYNAMO_CLOUD"
    ),
) -> None:
    """Delete a deployment by name."""
    deployment_manager = get_deployment_manager(target, endpoint)
    deployment_manager.delete_deployment(name)
    console.print(f"Deleted deployment {name}")
