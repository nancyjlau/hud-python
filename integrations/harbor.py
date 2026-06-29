"""Harbor integration: load Harbor task dirs as a Taskset; export HUD tasks to Harbor.

Harbor task structure (terminal-bench layout)::

    task_name/
    ├── instruction.md          # agent prompt
    ├── task.toml               # config: timeouts, metadata, resources
    ├── environment/Dockerfile  # container the agent runs in
    ├── tests/test.sh           # verification -> writes reward.txt
    └── solution/               # optional (ignored)

:func:`load` parses a task dir (or a dataset of them) into rows sharing one
env name per distinct ``environment/`` build context — no codegen, no
roundtrip. Like every row, the result is runnable once a placement is supplied.
Use :class:`HarborRuntime` for local Docker-backed execution of Harbor tasks, or
``runtime=Runtime(url)`` to attach to a substrate served elsewhere.

:func:`export` is the reverse direction: turn a HUD task source into
self-contained Harbor task folders (``task.toml`` + ``instruction.md`` +
``environment/`` + ``tests/test.sh``). Convertible iff the env's capabilities
are ``ssh``/``mcp`` only (Harbor is shell-centric; ``rfb``/``cdp`` don't map).

Export lifecycle mapping (HUD setup/evaluate → Harbor image/verifier):

* The env's build context is copied into ``environment/`` and a ``hud_entrypoint.sh``
  is baked in as the image ENTRYPOINT (Harbor overrides CMD with ``sleep infinity``).
  At container start it serves the env control channel (``hud serve``) and runs the
  task's **setup** (``hud task start``), which parks the paused run on the env so a
  later connection can grade it, then ``exec "$@"`` into the container command.
* The agent then works in the container and writes its answer to ``answer_file``.
* ``tests/test.sh`` runs the task's **evaluate** (``hud task grade``) against the
  parked run and writes the reward to ``/logs/verifier/reward.txt``.

The exported task grades over the HUD control channel, so it is *not* a
harness-agnostic Harbor task — it depends on the baked ENTRYPOINT serving that
channel.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import re
import shlex
import shutil
import tempfile
import tomllib
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from hud.environment import Environment
from hud.environment.server import TaskRunner
from hud.environment.workspace import Workspace
from hud.eval import Task, Taskset

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterator, Callable

    import asyncssh

    from hud.eval.runtime import Runtime

LOGGER = logging.getLogger(__name__)

#: Capability protocols that map onto Harbor's shell/tool model.
ALLOWED_PROTOCOLS = ("ssh", "mcp")

#: Where the agent writes its final answer (the contract between the instruction
#: and the verifier). Matches the Workspace default guest path.
DEFAULT_ANSWER_FILE = "/workspace/answer.txt"

#: Port the in-container env control channel is served on.
CONTROL_PORT = 8765

#: Build-context entries never copied into the Harbor ``environment/`` dir.
_BUILD_CONTEXT_IGNORE = shutil.ignore_patterns(
    "__pycache__", "*.pyc", ".git", ".venv", "venv", "*.egg-info", ".pytest_cache"
)


# ─── load: Harbor dirs -> Taskset ──────────────────────────────────────


def detect(path: str | Path) -> bool:
    """True when *path* is a Harbor task dir or a dataset of them."""
    return bool(_task_dirs(path))


def _task_dirs(path: str | Path) -> list[Path]:
    root = Path(path)
    if _is_harbor_task(root):
        return [root]
    if root.is_dir():
        return sorted(d for d in root.iterdir() if d.is_dir() and _is_harbor_task(d))
    return []


def load(path: str | Path) -> Taskset:
    """Load a Harbor task dir (or dataset dir) into a :class:`Taskset`.

    One row per task dir (``id`` = the dir name, ``task.toml`` ``[metadata]``
    as columns); rows share one env name per distinct ``environment/`` build
    context (content-hashed), derived from the dataset name.
    """
    root = Path(path).resolve()
    task_dirs = _task_dirs(root)
    dataset_name = root.parent.name if _is_harbor_task(root) else root.name
    if not task_dirs:
        raise ValueError(f"no Harbor tasks found in {path}")

    parsed = [task for task_dir in task_dirs if (task := _parse_task(task_dir)) is not None]
    if not parsed:
        raise ValueError(f"all Harbor tasks under {path} failed to parse")
    if len(parsed) < len(task_dirs):
        LOGGER.warning(
            "skipped %d Harbor task(s) that failed to parse", len(task_dirs) - len(parsed)
        )

    groups: dict[str, list[_HarborTask]] = {}
    for harbor_task in parsed:
        groups.setdefault(harbor_task.env_hash, []).append(harbor_task)
    sorted_groups = sorted(groups.values(), key=lambda group: -len(group))

    base_name = _slugify(dataset_name)
    tasks: list[Task] = []
    for idx, group in enumerate(sorted_groups, start=1):
        env_name = base_name if len(sorted_groups) == 1 else f"{base_name}-g{idx}"
        tasks.extend(Task(env=env_name, id=harbor_task.task_id) for harbor_task in group)
    return Taskset(base_name, tasks)


class HarborRuntime:
    """Run Harbor task directories through HUD's local rollout engine.

    The provider builds the Harbor task's ``environment/`` Docker context, runs
    a fresh container with a writable host workspace mounted at ``/app``, and
    serves a small HUD control channel from the host process. If the task ships a
    ``docker-compose.yaml``/``.yml``, the provider starts it with an overlay that
    keeps the ``main`` service idle while preserving sidecars such as databases.
    The agent receives normal HUD SSH/SFTP access; shell commands execute inside
    the main container via ``docker exec`` while file transfer edits the mounted
    host workspace. Grading runs the Harbor ``tests/test.sh`` inside the same
    main container and reads ``/logs/verifier/reward.json`` or ``reward.txt``.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        ready_timeout: float = 120.0,
    ) -> None:
        self.root = Path(path).resolve()
        self.ready_timeout = ready_timeout
        self._task_dirs = {task_dir.name: task_dir for task_dir in _task_dirs(self.root)}
        if not self._task_dirs:
            raise ValueError(f"no Harbor tasks found in {path}")
        self._image_cache: dict[Path, str] = {}

    @contextlib.asynccontextmanager
    async def __call__(self, task: Task) -> AsyncIterator[Runtime]:
        from hud.eval.runtime import Runtime, _local

        task_dir = self._task_dirs.get(task.id)
        if task_dir is None:
            raise KeyError(f"HarborRuntime has no task directory for {task.id!r}")
        env_dir = task_dir / "environment"
        tests_dir = task_dir / "tests"
        if not (env_dir / "Dockerfile").is_file():
            raise FileNotFoundError(f"Harbor task {task.id!r} has no environment/Dockerfile")
        if not (tests_dir / "test.sh").is_file():
            raise FileNotFoundError(f"Harbor task {task.id!r} has no tests/test.sh")

        with tempfile.TemporaryDirectory(prefix=f"hud-harbor-{_slugify(task.id)}-") as tmp:
            tmp_path = Path(tmp)
            workspace = tmp_path / "workspace"
            logs = tmp_path / "logs"
            shutil.copytree(env_dir, workspace)
            _ensure_start_script(workspace)
            _ensure_dockerfile_created_dirs(workspace)
            preserved_paths = _preserved_image_paths(workspace)
            logs.mkdir(parents=True, exist_ok=True)

            compose_file = _compose_file(env_dir)
            if compose_file is not None:
                async with self._compose_container(
                    task,
                    compose_file,
                    workspace,
                    tests_dir,
                    logs,
                    preserved_paths,
                ) as (
                    container,
                    provider,
                ):
                    env = self._environment_for(task, task_dir, workspace, logs, container)
                    async with _local(env) as runtime:
                        yield Runtime(
                            runtime.url,
                            params={
                                **runtime.params,
                                "provider": provider,
                                "container": container,
                                "ready_timeout": self.ready_timeout,
                            },
                            config=runtime.config,
                        )
            else:
                async with self._single_container(
                    task,
                    task_dir,
                    workspace,
                    tests_dir,
                    logs,
                    preserved_paths,
                ) as (
                    container,
                    provider,
                ):
                    env = self._environment_for(task, task_dir, workspace, logs, container)
                    async with _local(env) as runtime:
                        yield Runtime(
                            runtime.url,
                            params={
                                **runtime.params,
                                "provider": provider,
                                "container": container,
                                "ready_timeout": self.ready_timeout,
                            },
                            config=runtime.config,
                        )

    @contextlib.asynccontextmanager
    async def _single_container(
        self,
        task: Task,
        task_dir: Path,
        workspace: Path,
        tests_dir: Path,
        logs: Path,
        preserved_paths: list[str],
    ) -> AsyncIterator[tuple[str, str]]:
        from hud.eval.runtime import _docker

        image = await self._image_for(task_dir)
        env_dir = task_dir / "environment"
        await _restore_image_generated_files(image, workspace)
        container_name = f"hud-harbor-{_slugify(task.id)}-{uuid.uuid4().hex[:8]}"
        preserved_volume_args = [arg for path in preserved_paths for arg in ("--volume", path)]
        out, _ = await _docker(
            "run",
            "--detach",
            "--name",
            container_name,
            "--workdir",
            "/app",
            "--entrypoint",
            "sleep",
            "--volume",
            f"{workspace}:/app",
            "--volume",
            f"{tests_dir}:/tests:ro",
            "--volume",
            f"{logs}:/logs",
            *preserved_volume_args,
            image,
            "infinity",
        )
        container = out.strip()
        try:
            yield container, "harbor"
        finally:
            await _release_mount_permissions(container)
            await _docker("rm", "--force", "--volumes", container, check=False)
            await _docker("image", "rm", image, check=False)
            self._image_cache.pop(env_dir, None)

    @contextlib.asynccontextmanager
    async def _compose_container(
        self,
        task: Task,
        compose_file: Path,
        workspace: Path,
        tests_dir: Path,
        logs: Path,
        preserved_paths: list[str],
    ) -> AsyncIterator[tuple[str, str]]:
        from hud.eval.runtime import _docker

        project = f"hud-harbor-{_slugify(task.id)}-{uuid.uuid4().hex[:8]}"
        overlay = workspace.parent / "compose.hud.yaml"
        overlay.write_text(
            _compose_overlay(
                workspace=workspace,
                tests_dir=tests_dir,
                logs=logs,
                preserved_paths=preserved_paths,
            ),
            encoding="utf-8",
            newline="\n",
        )
        compose_args = ("compose", "-f", str(compose_file), "-f", str(overlay), "-p", project)
        await _docker(*compose_args, "up", "--detach", "--build")
        out, _ = await _docker(*compose_args, "ps", "-q", "main")
        container = out.strip()
        if not container:
            raise RuntimeError(f"docker compose project {project} did not create a main service")
        try:
            yield container, "harbor-compose"
        finally:
            await _release_mount_permissions(container)
            await _docker(
                *compose_args,
                "down",
                "--volumes",
                "--remove-orphans",
                "--rmi",
                "local",
                check=False,
            )

    async def _image_for(self, task_dir: Path) -> str:
        from hud.eval.runtime import _docker

        env_dir = task_dir / "environment"
        cached = self._image_cache.get(env_dir)
        if cached is not None:
            return cached
        tag = f"hud-harbor:{_hash_directory(env_dir)}"
        await _docker("build", "--tag", tag, str(env_dir))
        self._image_cache[env_dir] = tag
        return tag

    def _environment_for(
        self,
        task: Task,
        task_dir: Path,
        workspace: Path,
        logs: Path,
        container: str,
    ) -> Environment:
        env = Environment(task.env)
        workspace_daemon = _DockerWorkspace(workspace, container=container, guest_path="/app")

        @env.initialize
        async def _up() -> None:
            await workspace_daemon.start()
            env.add_capability(workspace_daemon.capability("shell"))

        @env.shutdown
        async def _down() -> None:
            await workspace_daemon.stop()

        @env.template(id=task.id, description=f"Harbor task {task.id}")
        async def _run_harbor_task() -> AsyncGenerator[Any, Any]:
            answer = yield (task_dir / "instruction.md").read_text(encoding="utf-8")
            yield await self._grade(container, logs, answer)

        return env

    async def _grade(self, container: str, logs: Path, answer: Any) -> dict[str, Any]:
        from hud.eval.runtime import _docker

        answer_file = logs / "agent_answer.txt"
        answer_file.parent.mkdir(parents=True, exist_ok=True)
        answer_file.write_text("" if answer is None else str(answer), encoding="utf-8")
        out, err = await _docker(
            "exec",
            "--workdir",
            "/app",
            container,
            "bash",
            "/tests/test.sh",
            check=False,
        )
        reward, info = _read_harbor_reward(logs / "verifier")
        info.update(
            {
                "stdout": out[-4000:],
                "stderr": err[-4000:],
            }
        )
        if reward is None:
            return {
                "score": 0.0,
                "isError": True,
                "content": "Harbor verifier did not write reward.json or reward.txt",
                "info": info,
            }
        return {"score": reward, "info": info}


class _DockerWorkspace(Workspace):
    """Workspace SFTP over a host bind mount, shell commands via docker exec."""

    def __init__(self, *args: Any, container: str, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._container = container

    async def _handle_process(self, process: asyncssh.SSHServerProcess[bytes]) -> None:
        import asyncio

        command = process.command or "bash -l"
        proc = await asyncio.create_subprocess_exec(
            "docker",
            "exec",
            "-i",
            "--workdir",
            self._guest_path,
            self._container,
            "bash",
            "-lc",
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout_data, stderr_data = await asyncio.wait_for(proc.communicate(), timeout=3600.0)
        except TimeoutError:
            proc.kill()
            await proc.wait()
            process.stderr.write(b"workspace: command timed out after 3600s\n")
            process.exit(1)
            return
        except asyncio.CancelledError:
            proc.kill()
            await proc.wait()
            raise

        if stdout_data:
            process.stdout.write(stdout_data)
        if stderr_data:
            process.stderr.write(stderr_data)
        process.exit(proc.returncode if proc.returncode is not None else 0)


def _read_harbor_reward(verifier_logs: Path) -> tuple[float | None, dict[str, Any]]:
    reward_json = verifier_logs / "reward.json"
    if reward_json.is_file():
        data = json.loads(reward_json.read_text(encoding="utf-8"))
        if isinstance(data, int | float):
            return float(data), {"reward_file": str(reward_json)}
        if isinstance(data, dict):
            for key in ("reward", "score"):
                value = data.get(key)
                if isinstance(value, int | float):
                    return float(value), {"reward_file": str(reward_json), "reward_json": data}
            numeric = [float(value) for value in data.values() if isinstance(value, int | float)]
            if numeric:
                return sum(numeric) / len(numeric), {
                    "reward_file": str(reward_json),
                    "reward_json": data,
                }
        return None, {"reward_file": str(reward_json), "reward_parse_error": "no numeric reward"}

    reward_txt = verifier_logs / "reward.txt"
    if reward_txt.is_file():
        text = reward_txt.read_text(encoding="utf-8").strip()
        try:
            return float(text), {"reward_file": str(reward_txt)}
        except ValueError:
            return None, {"reward_file": str(reward_txt), "reward_parse_error": text}

    return None, {}


async def _release_mount_permissions(container: str) -> None:
    """Let the host user delete files that container-root created in mounts."""
    from hud.eval.runtime import _docker

    await _docker(
        "exec",
        container,
        "sh",
        "-lc",
        "chmod -R a+rwX /app /logs 2>/dev/null || true",
        check=False,
    )


def _slugify(name: str) -> str:
    """A valid env name (lowercase ``[a-z0-9-]``) from a dataset dir name."""
    normalized = re.sub(r"[^a-z0-9-]", "", name.strip().lower().replace(" ", "-").replace("_", "-"))
    return re.sub(r"-+", "-", normalized).strip("-") or "harbor"


def _is_harbor_task(path: Path) -> bool:
    return path.is_dir() and (path / "task.toml").exists() and (path / "instruction.md").exists()


def _compose_file(env_dir: Path) -> Path | None:
    for name in ("docker-compose.yaml", "docker-compose.yml", "compose.yaml", "compose.yml"):
        path = env_dir / name
        if path.is_file():
            return path
    return None


def _compose_overlay(
    *,
    workspace: Path,
    tests_dir: Path,
    logs: Path,
    preserved_paths: list[str] | None = None,
) -> str:
    """Compose override that keeps Harbor's main service idle for agent work."""
    preserved_paths = preserved_paths or []
    volume_lines = [
        f"      - {json.dumps(f'{workspace}:/app')}",
        f"      - {json.dumps(f'{tests_dir}:/tests:ro')}",
        f"      - {json.dumps(f'{logs}:/logs')}",
    ]
    volume_lines.extend(f"      - {json.dumps(path)}" for path in preserved_paths)
    return "\n".join(
        [
            "services:",
            "  main:",
            "    build:",
            f"      context: {json.dumps(str(workspace))}",
            "    working_dir: /app",
            '    entrypoint: ["sleep"]',
            '    command: ["infinity"]',
            "    volumes:",
            *volume_lines,
            "",
        ],
    )


def _preserved_image_paths(workspace: Path) -> list[str]:
    """Image-populated subpaths that should survive the editable ``/app`` mount."""
    paths: list[str] = []
    if (workspace / "package.json").is_file():
        paths.append("/app/node_modules")
        if _node_build_output_is_image_populated(workspace, "dist"):
            paths.append("/app/dist")
    if (workspace / "composer.json").is_file():
        paths.append("/app/vendor")
    return paths


def _node_build_output_is_image_populated(workspace: Path, dirname: str) -> bool:
    if (workspace / dirname).exists():
        return False
    dockerfile = workspace / "Dockerfile"
    if not dockerfile.is_file():
        return False
    dockerfile_text = dockerfile.read_text(encoding="utf-8")
    entrypoint = workspace / "docker-entrypoint.sh"
    entrypoint_text = entrypoint.read_text(encoding="utf-8") if entrypoint.is_file() else ""
    return (
        "npm run build" in dockerfile_text
        or f"/app/{dirname}" in dockerfile_text
        or f" {dirname}/" in entrypoint_text
        or f" {dirname}" in entrypoint_text
    )


def _ensure_start_script(workspace: Path) -> None:
    """Preserve build-generated /app/start_app.sh hidden by the workspace mount."""
    start = workspace / "start_app.sh"
    entrypoint = workspace / "docker-entrypoint.sh"
    if not entrypoint.is_file():
        _restore_dockerfile_script(workspace, entrypoint, "/app/docker-entrypoint.sh")
    if entrypoint.is_file():
        entrypoint.chmod(entrypoint.stat().st_mode | 0o111)
    if start.exists():
        start.chmod(start.stat().st_mode | 0o111)
        return
    text = _script_from_dockerfile(workspace, "/app/start_app.sh")
    if text is None and entrypoint.is_file():
        text = "#!/usr/bin/env bash\nset -e\ncd /app\nexec sh /app/docker-entrypoint.sh\n"
    if text is None:
        return
    start.write_text(text, encoding="utf-8", newline="\n")
    start.chmod(0o755)


def _ensure_dockerfile_created_dirs(workspace: Path) -> None:
    """Recreate simple Dockerfile-created ``/app`` dirs hidden by the bind mount."""
    for path in _dockerfile_created_app_dirs(workspace):
        path.mkdir(parents=True, exist_ok=True)


async def _restore_image_generated_files(image: str, workspace: Path) -> None:
    """Copy selected build-generated files from the image into the workspace.

    Some Harbor images initialize file-backed databases during ``docker build``.
    The editable ``/app`` bind mount hides those generated files, so copy them
    out of the built image before starting the task container.
    """
    container_paths = _dockerfile_declared_generated_app_files(workspace)
    if not container_paths:
        return

    from hud.eval.runtime import _docker

    out, _ = await _docker("create", image, "true")
    container = out.strip()
    try:
        for container_path in container_paths:
            host_path = _host_path_for_app_file(workspace, container_path)
            if host_path is None or host_path.exists():
                continue
            host_path.parent.mkdir(parents=True, exist_ok=True)
            await _docker("cp", f"{container}:{container_path}", str(host_path), check=False)
    finally:
        await _docker("rm", "--force", "--volumes", container, check=False)


def _dockerfile_declared_generated_app_files(workspace: Path) -> list[str]:
    """Find Dockerfile-declared file-backed DB paths under ``/app``."""
    dockerfile = workspace / "Dockerfile"
    if not dockerfile.is_file():
        return []

    paths: list[str] = []
    for instruction in _dockerfile_logical_lines(dockerfile.read_text(encoding="utf-8")):
        stripped = instruction.strip()
        if not stripped.startswith("ENV "):
            continue
        for key, value in _env_pairs(stripped.removeprefix("ENV ").strip()):
            if not _is_generated_db_env_key(key):
                continue
            if _is_app_database_path(value):
                paths.append(value)
    return list(dict.fromkeys(paths))


def _env_pairs(body: str) -> list[tuple[str, str]]:
    try:
        tokens = shlex.split(body)
    except ValueError:
        return []
    if not tokens:
        return []

    pairs: list[tuple[str, str]] = []
    if all("=" in token for token in tokens):
        for token in tokens:
            key, value = token.split("=", 1)
            pairs.append((key, value))
        return pairs

    if len(tokens) >= 2:
        pairs.append((tokens[0], tokens[1]))
    return pairs


def _is_generated_db_env_key(key: str) -> bool:
    normalized = key.upper()
    return normalized in {
        "DB_PATH",
        "DATABASE_PATH",
        "SQLITE_PATH",
        "SQLITE_DB_PATH",
        "SQLITE_DATABASE_PATH",
    } or normalized.endswith(("_DB_PATH", "_DATABASE_PATH", "_SQLITE_PATH"))


def _is_app_database_path(path: str) -> bool:
    lowered = path.lower()
    return lowered.startswith("/app/") and lowered.endswith((".db", ".sqlite", ".sqlite3"))


def _host_path_for_app_file(workspace: Path, container_path: str) -> Path | None:
    if not container_path.startswith("/app/"):
        return None
    rel = container_path.removeprefix("/app/")
    if rel.startswith("../") or "/../" in rel or rel == "..":
        return None
    return workspace / rel


def _dockerfile_created_app_dirs(workspace: Path) -> list[Path]:
    dockerfile = workspace / "Dockerfile"
    if not dockerfile.is_file():
        return []
    paths: list[Path] = []
    for instruction in _dockerfile_logical_lines(dockerfile.read_text(encoding="utf-8")):
        stripped = instruction.strip()
        if not stripped.startswith("RUN "):
            continue
        command = stripped.removeprefix("RUN ").strip()
        try:
            tokens = shlex.split(command)
        except ValueError:
            continue
        index = 0
        while index < len(tokens):
            if tokens[index] != "mkdir":
                index += 1
                continue
            index += 1
            while index < len(tokens):
                token = tokens[index]
                if token in {"&&", "||", ";"}:
                    break
                if token.startswith("-"):
                    index += 1
                    continue
                host_path = _app_dir_from_mkdir_token(workspace, token)
                if host_path is not None:
                    paths.append(host_path)
                index += 1
    return paths


def _app_dir_from_mkdir_token(workspace: Path, token: str) -> Path | None:
    if not token or any(char in token for char in "$*?["):
        return None
    raw = token.rstrip("/")
    if raw in {"", "."}:
        return None
    if raw.startswith("/app/"):
        rel = raw.removeprefix("/app/")
    elif raw == "/app":
        return workspace
    elif raw.startswith("/"):
        return None
    else:
        rel = raw
    if rel.startswith("../") or "/../" in rel or rel == "..":
        return None
    return workspace / rel


def _restore_dockerfile_script(workspace: Path, host_path: Path, container_path: str) -> None:
    """Restore a Dockerfile-generated script hidden by a bind mount."""
    text = _script_from_dockerfile(workspace, container_path)
    if text is None:
        return
    host_path.write_text(text, encoding="utf-8", newline="\n")
    host_path.chmod(0o755)


def _script_from_dockerfile(workspace: Path, container_path: str) -> str | None:
    """Extract a Dockerfile-generated script from a simple ``RUN printf`` command."""
    dockerfile = workspace / "Dockerfile"
    if not dockerfile.is_file():
        return None
    for instruction in _dockerfile_logical_lines(dockerfile.read_text(encoding="utf-8")):
        stripped = instruction.strip()
        if not stripped.startswith("RUN ") or container_path not in stripped:
            continue
        command = stripped.removeprefix("RUN ").strip()
        try:
            tokens = shlex.split(command)
        except ValueError:
            continue
        redirect = _redirect_index(tokens, container_path)
        if redirect is None or redirect < 2 or tokens[0] != "printf":
            continue
        text = _script_from_printf_args(tokens[1:redirect])
        if text is not None:
            return text
    return None


def _redirect_index(tokens: list[str], target: str) -> int | None:
    for index, token in enumerate(tokens):
        if token in {">", ">>"} and index + 1 < len(tokens) and tokens[index + 1] == target:
            return index
        if token in {f">{target}", f">>{target}"}:
            return index
    return None


def _script_from_printf_args(args: list[str]) -> str | None:
    if not args:
        return None
    if args[0] in {"%s\\n", "%s\n"}:
        if len(args) < 2:
            return None
        return "\n".join(args[1:]) + "\n"
    if len(args) == 1:
        return args[0].replace("\\r", "\r").replace("\\n", "\n").replace("\\t", "\t")
    return None


def _dockerfile_logical_lines(text: str) -> list[str]:
    """Join backslash-continued Dockerfile lines for simple instruction parsing."""
    lines: list[str] = []
    current = ""
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if line.endswith("\\"):
            current += line[:-1] + " "
            continue
        lines.append(current + line)
        current = ""
    if current:
        lines.append(current)
    return lines


def _hash_directory(path: Path) -> str:
    """Content-hash a directory for grouping tasks by identical environments."""
    hasher = hashlib.sha256()
    if not path.exists():
        return "empty"
    for file_path in sorted(path.rglob("*")):
        if file_path.is_file():
            hasher.update(str(file_path.relative_to(path)).encode())
            hasher.update(file_path.read_bytes())
    return hasher.hexdigest()[:16]


@dataclass(frozen=True, slots=True)
class _HarborTask:
    """One parsed Harbor task dir."""

    task_id: str
    config: dict[str, Any]
    env_hash: str


def _parse_task(task_dir: Path) -> _HarborTask | None:
    if not (task_dir / "instruction.md").is_file():
        LOGGER.warning("failed to read instruction.md in %s", task_dir)
        return None
    try:
        config: dict[str, Any] = tomllib.loads((task_dir / "task.toml").read_text("utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        LOGGER.warning("failed to parse task.toml in %s", task_dir)
        config = {}
    env_dir = task_dir / "environment"
    return _HarborTask(
        task_id=task_dir.name,
        config=config,
        env_hash=_hash_directory(env_dir) if env_dir.exists() else "no-env",
    )


# ─── export: HUD tasks -> Harbor task folders ───────────────────────────


def _write_text(path: Path, text: str) -> None:
    """Write a generated file with LF endings (these run in Linux containers;
    the default Windows ``\\r\\n`` translation breaks shebangs and shell scripts)."""
    path.write_text(text, encoding="utf-8", newline="\n")


def _check_capabilities(env: Environment) -> None:
    bad = [
        c.protocol for c in env.capabilities if c.protocol.split("/", 1)[0] not in ALLOWED_PROTOCOLS
    ]
    if bad:
        raise ValueError(
            f"env {env.name!r} declares non-Harbor capabilities {bad}; "
            f"only {'/'.join(ALLOWED_PROTOCOLS)} are convertible.",
        )


async def _materialize_prompt(env: Environment, task: str, args: dict[str, Any]) -> str:
    """Run a task's first yield locally to get its concrete prompt (deterministic)."""
    runner = TaskRunner(env.tasks[task], args)
    try:
        payload = await runner.start()
    finally:
        await runner.cancel()
    prompt = payload.get("prompt")
    return prompt if isinstance(prompt, str) else json.dumps(prompt, indent=2, default=str)


def _resolve_env(task: Task, authored: dict[str, Environment]) -> Environment:
    """Resolve a task row's env name to a local, authored env defining the task.

    Rows reference envs by name; export materializes prompts in-process, so
    the authored ``Environment`` must be defined in (or next to) the task
    source. A row whose name matches nothing exportable fails loudly.
    """
    env = authored.get(task.env)
    if env is None or task.id not in env.tasks:
        raise TypeError(
            f"harbor export needs a local env defining task {task.id!r} "
            f"(an env.py named {task.env!r} next to the tasks); none was found.",
        )
    return env


# ─── generated files ───────────────────────────────────────────────────

_ENTRYPOINT_SH = """\
#!/bin/sh
# Baked ENTRYPOINT (POSIX sh — slim base images have no bash): serve the HUD
# control channel, run the task setup (parking the paused run), then exec the
# container command. Harbor overrides CMD with `sleep infinity`, so setup must
# run via ENTRYPOINT; `exec "$@"` keeps the channel alive alongside it. The
# agent and the verifier both run in this same container, so the verifier
# reaches the parked run on 127.0.0.1:{port} to grade.
set -u

hud serve env:env --port {port} &

# Wait for the control channel to accept connections (python is always present).
python3 -c 'import socket, sys, time
port = int(sys.argv[1])
for _ in range(120):
    try:
        socket.create_connection(("127.0.0.1", port), 0.5).close()
        break
    except OSError:
        time.sleep(0.5)' {port} || true

# Run the task setup phase and park the run for grading.
hud task start '{task}' --args '{args_json}' --url tcp://127.0.0.1:{port} >/dev/null 2>&1 || true

exec "$@"
"""

_TEST_SH = """\
#!/bin/sh
# Grade the parked HUD run against the agent's work, writing the Harbor reward.
set -u
mkdir -p /logs/verifier

ANSWER_FILE='{answer_file}'
[ -f "$ANSWER_FILE" ] || : > "$ANSWER_FILE"

if hud task grade '{task}' --args '{args_json}' --answer-file "$ANSWER_FILE" \\
    --url tcp://127.0.0.1:{port} > /logs/verifier/reward.txt 2> /logs/verifier/grade.err; then
    :
else
    echo 0 > /logs/verifier/reward.txt
fi
"""

_INSTRUCTION_SUFFIX = """\

---
When you have finished, write your final answer to `{answer_file}`.
"""


def _adapt_env_dockerfile(content: str) -> str:
    """Neutralize the env's own CMD/ENTRYPOINT and bake the boot ENTRYPOINT.

    ENTRYPOINT (not CMD) because Harbor overrides the container command with
    ``sleep infinity``; our entrypoint runs setup then ``exec "$@"`` into it.
    """
    lines: list[str] = []
    for line in content.splitlines():
        stripped = line.strip().upper()
        if stripped.startswith(("CMD ", "CMD[", "ENTRYPOINT ", "ENTRYPOINT[")):
            lines.append(f"# [hud original] {line}")
        else:
            lines.append(line)
    boot_layer = (
        "\n# ─── HUD → Harbor boot entrypoint ───\n"
        "COPY hud_entrypoint.sh /hud_entrypoint.sh\n"
        "RUN chmod +x /hud_entrypoint.sh\n"
        'ENTRYPOINT ["/hud_entrypoint.sh"]\n'
        "# Default command for standalone `docker run`; Harbor injects its own.\n"
        'CMD ["sh", "-c", "sleep infinity"]\n'
    )
    return "\n".join(lines) + "\n" + boot_layer


def _harbor_task_toml(name: str, task: str, args: dict[str, Any], timeout: float) -> str:
    """A Harbor-native ``task.toml`` (``name``/``version`` required by the registry)."""
    return (
        'version = "1.0"\n'
        f'name = "{name}"\n'
        "\n[metadata]\n"
        f'hud_task = "{task}"\n'
        f"hud_args = {json.dumps(json.dumps(args))}\n"
        "\n[agent]\n"
        f"timeout_sec = {timeout}\n"
        "\n[verifier]\n"
        f"timeout_sec = {timeout}\n"
    )


def _find_dockerfile(source_dir: Path) -> Path | None:
    return next(
        (source_dir / n for n in ("Dockerfile.hud", "Dockerfile") if (source_dir / n).exists()),
        None,
    )


def _make_ignore(out_root: Path) -> Callable[[str, list[str]], set[str]]:
    """Ignore the standard caches plus the export output dir (which may live under
    the source dir, e.g. ``./harbor_tasks`` next to ``env.py``)."""
    out_resolved = out_root.resolve()

    def _ignore(dirpath: str, names: list[str]) -> set[str]:
        ignored = set(_BUILD_CONTEXT_IGNORE(dirpath, names))
        base = Path(dirpath)
        ignored.update(n for n in names if (base / n).resolve() == out_resolved)
        return ignored

    return _ignore


def _write_environment(
    task_dir: Path,
    source_dir: Path,
    dockerfile: Path,
    task: str,
    args: dict[str, Any],
    out_root: Path,
) -> None:
    """Copy the env build context into ``environment/`` and bake the boot entrypoint."""
    env_out = task_dir / "environment"
    if env_out.exists():
        shutil.rmtree(env_out)
    shutil.copytree(source_dir, env_out, ignore=_make_ignore(out_root))

    # Drop any copied taskset files and the source Dockerfile name we don't use.
    for stale in env_out.glob("*.json"):
        stale.unlink()
    for name in ("Dockerfile.hud", "dockerfile"):
        leftover = env_out / name
        if leftover.exists() and leftover.name != "Dockerfile":
            leftover.unlink()

    _write_text(env_out / "Dockerfile", _adapt_env_dockerfile(dockerfile.read_text("utf-8")))
    _write_text(
        env_out / "hud_entrypoint.sh",
        _ENTRYPOINT_SH.format(port=CONTROL_PORT, task=task, args_json=json.dumps(args)),
    )


async def export(
    source: str,
    out_dir: str | Path,
    *,
    answer_file: str = DEFAULT_ANSWER_FILE,
    timeout_sec: float = 600.0,
) -> list[Path]:
    """Export HUD tasks from *source* into Harbor task folders under *out_dir*.

    *source* is either a **tasks file** (``.json`` / ``.jsonl`` of ``{env, task,
    args}`` entries) or a ``.py`` file/dir exposing ``Task``s. One folder is
    written per task (task + args), each a self-contained Harbor task. Requires the
    env's build context (a ``Dockerfile.hud``/``Dockerfile`` next to the source).
    Returns the created task directories.
    """
    from hud.utils.modules import iter_modules

    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    src = Path(source).resolve()
    source_dir = src.parent if src.is_file() else src

    tasks = list(Taskset.from_file(src))
    # Rows reference envs by name; collect the authored envs (defined in the
    # source, or next to a tasks file) to materialize prompts in-process.
    scan = source_dir if src.suffix in (".json", ".jsonl") else src
    authored = {
        env.name: env
        for module in iter_modules(scan)
        for env in vars(module).values()
        if isinstance(env, Environment)
    }

    dockerfile = _find_dockerfile(source_dir)
    if dockerfile is None:
        raise FileNotFoundError(
            f"no Dockerfile(.hud) next to {source_dir}; harbor export needs the env's "
            "build context to rebuild the image under Harbor.",
        )

    created: list[Path] = []
    for task in tasks:
        env = _resolve_env(task, authored)
        _check_capabilities(env)

        slug = task.slug or task.default_slug()
        task_dir = out / slug
        (task_dir / "tests").mkdir(parents=True, exist_ok=True)

        prompt = await _materialize_prompt(env, task.id, task.args)
        instruction = prompt + _INSTRUCTION_SUFFIX.format(answer_file=answer_file)
        _write_text(task_dir / "instruction.md", instruction)

        _write_text(
            task_dir / "task.toml",
            _harbor_task_toml(slug, task.id, task.args, timeout_sec),
        )

        _write_environment(task_dir, source_dir, dockerfile, task.id, task.args, out)

        _write_text(
            task_dir / "tests" / "test.sh",
            _TEST_SH.format(
                port=CONTROL_PORT,
                task=task.id,
                args_json=json.dumps(task.args),
                answer_file=answer_file,
            ),
        )

        created.append(task_dir)

    return created


__all__ = [
    "ALLOWED_PROTOCOLS",
    "CONTROL_PORT",
    "DEFAULT_ANSWER_FILE",
    "HarborRuntime",
    "detect",
    "export",
    "load",
]
