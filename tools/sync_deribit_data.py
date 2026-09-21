from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path


# ==================================================
# CONFIGURATION
# ==================================================

VM_NAME = "deribit-collector"

LOCAL_DATA_ROOT = Path(
    r"D:\MarketData\Deribit"
)

SYNC_STATE_ROOT = (
    LOCAL_DATA_ROOT
    / "_sync"
)

MANIFEST_FILE = (
    SYNC_STATE_ROOT
    / "manifest.json"
)

LOG_FILE = (
    SYNC_STATE_ROOT
    / "sync.log"
)


# ==================================================
# LOGGING
# ==================================================

def configure_logging() -> None:

    SYNC_STATE_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(message)s"
        ),
        handlers=[
            logging.FileHandler(
                LOG_FILE,
                encoding="utf-8"
            ),
            logging.StreamHandler(
                sys.stdout
            )
        ]
    )


# ==================================================
# GCLOUD
# ==================================================

def find_gcloud() -> str:

    explicit = os.getenv(
        "GCLOUD_EXE"
    )

    if explicit:
        return explicit

    for candidate in [
        "gcloud",
        "gcloud.cmd"
    ]:

        path = shutil.which(
            candidate
        )

        if path:
            return path

    raise RuntimeError(
        "gcloud was not found.\n"
        "Run 'where gcloud' in Windows CMD "
        "and verify Google Cloud CLI is installed."
    )


GCLOUD = ""


def build_gcloud_command(
    arguments: list[str]
) -> list[str]:
    """
    Windows .cmd files are safest when invoked
    explicitly through cmd.exe.
    """

    if (
        os.name == "nt"
        and GCLOUD.lower().endswith(
            (".cmd", ".bat")
        )
    ):

        return [
            os.environ.get(
                "COMSPEC",
                "cmd.exe"
            ),
            "/d",
            "/c",
            GCLOUD,
            *arguments
        ]

    return [
        GCLOUD,
        *arguments
    ]


def run_gcloud(
    arguments: list[str]
) -> str:

    command = build_gcloud_command(
        arguments
    )

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace"
    )

    if result.returncode != 0:

        raise RuntimeError(
            "\ngcloud command failed.\n\n"
            f"STDOUT:\n"
            f"{result.stdout}\n\n"
            f"STDERR:\n"
            f"{result.stderr}"
        )

    return result.stdout.strip()


# ==================================================
# GOOGLE CLOUD DISCOVERY
# ==================================================

def get_project() -> str:

    project = run_gcloud(
        [
            "config",
            "get-value",
            "project",
            "--quiet"
        ]
    )

    if (
        not project
        or project == "(unset)"
    ):

        raise RuntimeError(
            "No active Google Cloud project "
            "was found."
        )

    return project


def get_vm_zone(
    project: str
) -> str:

    output = run_gcloud(
        [
            "compute",
            "instances",
            "list",

            "--project",
            project,

            "--filter",
            f"name={VM_NAME}",

            "--format=value(zone.basename())"
        ]
    )

    zones = [
        line.strip()
        for line in output.splitlines()
        if line.strip()
    ]

    if not zones:

        raise RuntimeError(
            f"VM '{VM_NAME}' was not found."
        )

    return zones[0]


def remote_command(
    project: str,
    zone: str,
    command: str
) -> str:

    return run_gcloud(
        [
            "compute",
            "ssh",
            VM_NAME,

            "--project",
            project,

            "--zone",
            zone,

            "--quiet",

            "--command",
            command
        ]
    )


def get_remote_home(
    project: str,
    zone: str
) -> str:

    home = remote_command(
        project,
        zone,
        'printf "%s" "$HOME"'
    ).strip()

    if not home.startswith("/"):

        raise RuntimeError(
            f"Unexpected VM home directory: "
            f"{home}"
        )

    return home


# ==================================================
# MANIFEST
# ==================================================

def load_manifest() -> dict:

    if not MANIFEST_FILE.exists():

        return {
            "files": {}
        }

    try:

        with open(
            MANIFEST_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(
                file
            )

        if "files" not in data:

            data[
                "files"
            ] = {}

        return data

    except Exception:

        logging.warning(
            "Could not read existing manifest. "
            "Starting with an empty manifest."
        )

        return {
            "files": {}
        }


def save_manifest(
    manifest: dict
) -> None:

    temporary = (
        MANIFEST_FILE.with_suffix(
            ".tmp"
        )
    )

    with open(
        temporary,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            manifest,
            file,
            indent=2,
            sort_keys=True
        )

    os.replace(
        temporary,
        MANIFEST_FILE
    )


# ==================================================
# REMOTE FILE LISTING
# ==================================================

def parse_remote_listing(
    output: str
) -> list[dict]:

    files = []

    for line in output.splitlines():

        line = line.strip()

        if not line:
            continue

        parts = line.split(
            "\t"
        )

        if len(parts) != 3:
            continue

        path = parts[0]

        try:

            size = int(
                parts[1]
            )

        except ValueError:

            continue

        mtime = parts[2]

        files.append(
            {
                "path": path,
                "size": size,
                "mtime": mtime
            }
        )

    return files


def list_files_under(
    project: str,
    zone: str,
    remote_root: str,
    name_pattern: str
) -> list[dict]:

    command = (
        f'if [ -d "{remote_root}" ]; then '
        f'find "{remote_root}" '
        f'-type f '
        f'-name "{name_pattern}" '
        f'-printf "%p\\t%s\\t%T@\\n"; '
        f'fi'
    )

    output = remote_command(
        project,
        zone,
        command
    )

    return parse_remote_listing(
        output
    )


# ==================================================
# DATASET DISCOVERY
# ==================================================

def list_daily_options(
    project: str,
    zone: str,
    remote_home: str
) -> list[dict]:

    root = (
        f"{remote_home}/"
        f"deribit-data/"
        f"daily/"
        f"option_grid"
    )

    return list_files_under(
        project,
        zone,
        root,
        "option_grid.parquet"
    )


def list_futures(
    project: str,
    zone: str,
    remote_home: str
) -> list[dict]:

    root = (
        f"{remote_home}/"
        f"deribit-data/"
        f"futures/"
        f"5m"
    )

    return list_files_under(
        project,
        zone,
        root,
        "futures_5m.parquet"
    )


def get_daily_expiry_names(
    daily_files: list[dict]
) -> set[str]:

    expiries = set()

    for file_info in daily_files:

        path = file_info[
            "path"
        ]

        for component in (
            path.split("/")
        ):

            if component.startswith(
                "expiry="
            ):

                expiries.add(
                    component
                )

    return expiries


def list_raw_sessions(
    project: str,
    zone: str,
    remote_home: str
) -> list[str]:

    root = (
        f"{remote_home}/"
        f"deribit-data/"
        f"raw/"
        f"option_snapshots"
    )

    command = (
        f'if [ -d "{root}" ]; then '
        f'for d in "{root}"/expiry=*; do '
        f'[ -d "$d" ] && '
        f'printf "%s\\n" "$d"; '
        f'done; '
        f'fi'
    )

    output = remote_command(
        project,
        zone,
        command
    )

    return sorted(
        [
            line.strip()
            for line in output.splitlines()
            if line.strip()
        ]
    )


def list_uncompacted_raw_options(
    project: str,
    zone: str,
    remote_home: str,
    completed_expiries: set[str]
) -> list[dict]:
    """
    Synchronize raw option sessions only when
    no completed daily Parquet exists yet.

    Normally this means only the current session.
    """

    sessions = list_raw_sessions(
        project,
        zone,
        remote_home
    )

    files = []

    for session_path in sessions:

        session_name = (
            session_path
            .rstrip("/")
            .split("/")[-1]
        )

        if (
            session_name
            in completed_expiries
        ):

            continue

        session_files = (
            list_files_under(
                project,
                zone,
                session_path,
                "snapshot_*.parquet"
            )
        )

        files.extend(
            session_files
        )

    return files


# ==================================================
# LOCAL PATH MAPPING
# ==================================================

def remote_to_local(
    remote_path: str
) -> Path:

    marker = (
        "/deribit-data/"
    )

    if marker not in remote_path:

        raise RuntimeError(
            "Remote file is outside "
            "DERIBIT_DATA_ROOT:\n"
            f"{remote_path}"
        )

    relative = remote_path.split(
        marker,
        1
    )[1]

    components = relative.split(
        "/"
    )

    return LOCAL_DATA_ROOT.joinpath(
        *components
    )


# ==================================================
# SYNC DECISION
# ==================================================

def file_needs_download(
    file_info: dict,
    manifest: dict
) -> bool:

    remote_path = (
        file_info[
            "path"
        ]
    )

    local_path = (
        remote_to_local(
            remote_path
        )
    )

    if not local_path.exists():

        return True

    previous = (
        manifest[
            "files"
        ].get(
            remote_path
        )
    )

    if previous is None:

        return True

    if (
        previous.get("size")
        != file_info["size"]
    ):

        return True

    if (
        previous.get("mtime")
        != file_info["mtime"]
    ):

        return True

    if (
        local_path.stat().st_size
        != file_info["size"]
    ):

        return True

    return False


# ==================================================
# DOWNLOAD
# ==================================================

def download_file(
    project: str,
    zone: str,
    file_info: dict
) -> Path:

    remote_path = (
        file_info[
            "path"
        ]
    )

    local_path = remote_to_local(
        remote_path
    )

    local_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    temporary_file = (
        local_path.parent
        / (
            "."
            + local_path.name
            + ".download"
        )
    )

    if temporary_file.exists():

        temporary_file.unlink()

    logging.info(
        "Downloading: %s",
        remote_path
    )

    run_gcloud(
        [
            "compute",
            "scp",

            (
                f"{VM_NAME}:"
                f"{remote_path}"
            ),

            str(
                temporary_file
            ),

            "--project",
            project,

            "--zone",
            zone,

            "--quiet"
        ]
    )

    if not temporary_file.exists():

        raise RuntimeError(
            "gcloud completed but "
            "the downloaded file "
            "does not exist:\n"
            f"{temporary_file}"
        )

    local_size = (
        temporary_file
        .stat()
        .st_size
    )

    if (
        local_size
        != file_info["size"]
    ):

        temporary_file.unlink(
            missing_ok=True
        )

        raise RuntimeError(
            "File size mismatch after download.\n"
            f"Remote: "
            f"{file_info['size']:,} bytes\n"
            f"Local:  "
            f"{local_size:,} bytes\n"
            f"File: {remote_path}"
        )

    os.replace(
        temporary_file,
        local_path
    )

    logging.info(
        "Saved: %s",
        local_path
    )

    return local_path


# ==================================================
# COLLECTION SYNC
# ==================================================

def sync_collection(
    name: str,
    files: list[dict],
    project: str,
    zone: str,
    manifest: dict
) -> tuple[int, int]:

    downloaded = 0
    unchanged = 0

    logging.info(
        "%s remote files: %d",
        name,
        len(files)
    )

    for file_info in files:

        remote_path = (
            file_info[
                "path"
            ]
        )

        if file_needs_download(
            file_info,
            manifest
        ):

            download_file(
                project,
                zone,
                file_info
            )

            manifest[
                "files"
            ][
                remote_path
            ] = {
                "size":
                    file_info[
                        "size"
                    ],

                "mtime":
                    file_info[
                        "mtime"
                    ]
            }

            # Save incrementally so an interruption
            # does not lose already-completed state.
            save_manifest(
                manifest
            )

            downloaded += 1

        else:

            unchanged += 1

    logging.info(
        "%s: downloaded=%d unchanged=%d",
        name,
        downloaded,
        unchanged
    )

    return (
        downloaded,
        unchanged
    )


# ==================================================
# LOCAL RAW CLEANUP
# ==================================================

def cleanup_compacted_raw_options(
    completed_expiries: set[str]
) -> None:
    """
    Once a completed daily Parquet exists locally,
    the corresponding local raw snapshot folder is
    redundant and can be removed.

    Remote raw data is NEVER touched.
    """

    raw_root = (
        LOCAL_DATA_ROOT
        / "raw"
        / "option_snapshots"
    )

    if not raw_root.exists():
        return

    for folder in raw_root.glob(
        "expiry=*"
    ):

        if not folder.is_dir():
            continue

        if (
            folder.name
            not in completed_expiries
        ):

            continue

        daily_file = (
            LOCAL_DATA_ROOT
            / "daily"
            / "option_grid"
            / folder.name
            / "option_grid.parquet"
        )

        if not daily_file.exists():

            continue

        logging.info(
            "Removing redundant local raw "
            "session: %s",
            folder
        )

        shutil.rmtree(
            folder
        )


# ==================================================
# MAIN
# ==================================================

def main() -> None:

    global GCLOUD

    configure_logging()

    logging.info(
        "=" * 60
    )

    logging.info(
        "Starting Deribit VM -> PC sync"
    )

    GCLOUD = find_gcloud()

    project = get_project()

    zone = get_vm_zone(
        project
    )

    remote_home = get_remote_home(
        project,
        zone
    )

    logging.info(
        "Project: %s",
        project
    )

    logging.info(
        "VM: %s",
        VM_NAME
    )

    logging.info(
        "Zone: %s",
        zone
    )

    logging.info(
        "Remote home: %s",
        remote_home
    )

    manifest = load_manifest()

    # --------------------------------------------------
    # 1. Completed option sessions
    # --------------------------------------------------

    daily_options = (
        list_daily_options(
            project,
            zone,
            remote_home
        )
    )

    sync_collection(
        name="Daily options",
        files=daily_options,
        project=project,
        zone=zone,
        manifest=manifest
    )

    completed_expiries = (
        get_daily_expiry_names(
            daily_options
        )
    )

    # --------------------------------------------------
    # 2. Current / uncompacted raw option sessions
    # --------------------------------------------------

    raw_options = (
        list_uncompacted_raw_options(
            project,
            zone,
            remote_home,
            completed_expiries
        )
    )

    sync_collection(
        name="Raw options",
        files=raw_options,
        project=project,
        zone=zone,
        manifest=manifest
    )

    # --------------------------------------------------
    # 3. Futures 5-minute Parquets
    # --------------------------------------------------

    futures = list_futures(
        project,
        zone,
        remote_home
    )

    sync_collection(
        name="Futures",
        files=futures,
        project=project,
        zone=zone,
        manifest=manifest
    )

    # --------------------------------------------------
    # 4. Remove redundant LOCAL raw sessions
    # --------------------------------------------------

    cleanup_compacted_raw_options(
        completed_expiries
    )

    save_manifest(
        manifest
    )

    logging.info(
        "Synchronization complete."
    )


if __name__ == "__main__":

    main()