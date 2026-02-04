from urllib.parse import parse_qsl, urlparse

import docker
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

    uri = normalize_uri(uri)

    for container in containers:
        c_uri = container.labels.get("vlmparse_uri")
        c_model = container.labels.get("vlmparse_model_name")

        if c_uri and uri == normalize_uri(c_uri):
            # Infer model if not provided
            if model is None and c_model:
                logger.info(f"Inferred model {c_model} from container")
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
