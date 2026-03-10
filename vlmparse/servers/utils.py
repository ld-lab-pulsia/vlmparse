from urllib.parse import parse_qsl, urlparse

import docker
import docker.errors
from loguru import logger


def normalize_uri(uri: str) -> tuple:
    u = urlparse(uri)

    # --- Normalize scheme ---
    scheme = (u.scheme or "http").lower()

    # --- Normalize host ---
    host = (u.hostname or "").lower()
    if host in ("localhost", "0.0.0.0"):
        host = "localhost"

    # --- Normalize port (apply defaults) ---
    if u.port:
        port = u.port
    else:
        port = 443 if scheme == "https" else 80

    # --- Normalize path ---
    # Treat empty path as "/" and remove trailing slash (except root)
    path = u.path or "/"
    if path != "/" and path.endswith("/"):
        path = path.rstrip("/")

    # Collapse duplicate slashes
    while "//" in path:
        path = path.replace("//", "/")

    # --- Normalize query parameters (sorted) ---
    query_pairs = parse_qsl(u.query, keep_blank_values=True)
    query = "&".join(f"{k}={v}" for k, v in sorted(query_pairs))

    return (scheme, host, port, path, query)


def get_model_from_uri(uri: str) -> str:
    model = None
    client = docker.from_env()
    containers = client.containers.list()

    uri_tuple = normalize_uri(uri)

    for container in containers:
        c_uri = container.labels.get("vlmparse_uri")
        c_model = container.labels.get("vlmparse_model_name")

        if c_uri and uri_tuple[2] == normalize_uri(c_uri)[2]:
            # Infer model if not provided
            if model is None and c_model:
                logger.debug(f"Inferred model {c_model} from container")
                model = c_model
            break
    if model is None:
        raise ValueError(f"No model found for URI {uri}")
    return model


def _get_container_labels(container) -> dict[str, str]:
    labels: dict[str, str] = {}
    try:
        labels.update(getattr(container, "labels", None) or {})
    except Exception:
        pass

    try:
        labels.update((container.attrs or {}).get("Config", {}).get("Labels", {}) or {})
    except Exception:
        pass

    return labels


def _get_container_vlmparse_network(container) -> str | None:
    """Return the name of the first vlmparse* Docker network the container belongs to."""
    try:
        networks = container.attrs.get("NetworkSettings", {}).get("Networks", {})
        for net_name in networks:
            if net_name.startswith("vlmparse"):
                return net_name
    except Exception:
        pass
    return None


def _get_vlmparse_groups(running_only: bool = False) -> tuple[dict, set]:
    """Return ``(groups, network_keys)`` for all vlmparse deployments.

    *groups* maps a group key to a list of containers belonging to that group.
    *network_keys* is the subset of group keys that correspond to Docker networks
    (i.e. ContainerGroupServer deployments).

    Discovery priority:
    1. Containers connected to a Docker network whose name starts with ``vlmparse``.
    2. Remaining containers whose name starts with ``vlmparse``, grouped by their
       Docker Compose project label or by container name when standalone.
    3. Remaining containers that belong to a compose project starting with ``vlmparse``.
    """
    client = docker.from_env()
    containers_list = (
        client.containers.list() if running_only else client.containers.list(all=True)
    )
    groups: dict[str, list] = {}
    assigned: set[str] = set()
    network_keys: set[str] = set()

    # 1. vlmparse Docker network groups
    try:
        vlmparse_networks = [
            n for n in client.networks.list() if n.name.startswith("vlmparse")
        ]
    except Exception:
        vlmparse_networks = []

    for c in containers_list:
        net_name = _get_container_vlmparse_network(c)
        if net_name:
            groups.setdefault(net_name, []).append(c)
            network_keys.add(net_name)
            assigned.add(c.id)

    # Also sweep running containers reported directly by each network (covers edge
    # cases where container.attrs may be stale).
    for network in vlmparse_networks:
        try:
            network.reload()
            for c in network.containers:
                if c.id not in assigned:
                    groups.setdefault(network.name, []).append(c)
                    network_keys.add(network.name)
                    assigned.add(c.id)
        except Exception:
            pass

    # 2. Remaining containers – compose projects or standalone
    for c in containers_list:
        if c.id in assigned:
            continue
        lbl = _get_container_labels(c)
        project = lbl.get("com.docker.compose.project") or lbl.get(
            "vlmparse_compose_project"
        )
        if c.name.startswith("vlmparse"):
            key = project if project else c.name
            groups.setdefault(key, []).append(c)
            assigned.add(c.id)
        elif project and project.startswith("vlmparse"):
            groups.setdefault(project, []).append(c)
            assigned.add(c.id)

    return groups, network_keys


def _stop_network_group(network_name: str) -> bool:
    """Stop all containers in *network_name* and remove the Docker network.

    Returns True if the network was found and a stop was attempted.
    """
    client = docker.from_env()
    try:
        network = client.networks.get(network_name)
    except docker.errors.NotFound:
        return False

    network.reload()
    containers = list(network.containers)
    logger.info(
        f"Stopping network group '{network_name}' ({len(containers)} container(s))..."
    )
    for c in containers:
        try:
            c.stop(timeout=10)
        except Exception:
            pass
        try:
            c.remove(force=True)
        except Exception:
            pass

    try:
        network.remove()
        logger.info(f"✓ Removed Docker network '{network_name}'")
    except Exception as e:
        logger.warning(f"Could not remove network '{network_name}': {e}")

    logger.info("✓ Network group stopped and removed successfully")
    return True


def _stop_compose_stack_for_container(target_container) -> bool:
    """If container belongs to a docker-compose project, stop+remove the whole stack.

    Returns True if a compose stack was detected and a stack stop was attempted.
    """

    import subprocess

    labels = _get_container_labels(target_container)

    project = labels.get("com.docker.compose.project") or labels.get(
        "vlmparse_compose_project"
    )
    compose_file = labels.get("vlmparse_compose_file")

    if not project:
        return False

    # Preferred: docker compose down (stops + removes all services/networks consistently)
    if compose_file:
        logger.info(
            f"Detected docker-compose project '{project}'. Bringing stack down (stop + remove)..."
        )
        subprocess.run(
            [
                "docker",
                "compose",
                "-f",
                compose_file,
                "--project-name",
                project,
                "down",
                "--remove-orphans",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        logger.info("✓ Compose stack brought down")
        return True

    # Fallback: remove all containers in the compose project via Docker labels
    import docker

    logger.info(
        f"Detected docker-compose project '{project}' (compose file unknown). "
        "Stopping + removing all project containers via Docker API..."
    )
    client = docker.from_env()
    containers = client.containers.list(
        all=True, filters={"label": [f"com.docker.compose.project={project}"]}
    )
    for c in containers:
        try:
            c.stop()
        except Exception:
            pass
        try:
            c.remove(force=True)
        except Exception:
            pass

    logger.info(
        f"✓ Removed {len(containers)} container(s) from compose project '{project}'"
    )
    return True
