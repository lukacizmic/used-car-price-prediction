import argparse
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def run_step(command: list[str], cwd: Path) -> None:
    print(f"\n>> Running: {' '.join(command)}")
    completed = subprocess.run(command, cwd=cwd)
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {completed.returncode}: {command}")


def wait_for_api(url: str, timeout_seconds: int) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as response:  # nosec B310
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError):
            time.sleep(1.0)
    raise TimeoutError(f"API did not become healthy within {timeout_seconds}s at {url}")


def terminate_process(proc: subprocess.Popen | None, name: str) -> None:
    if proc is None or proc.poll() is not None:
        return
    print(f">> Stopping {name}...")
    try:
        proc.terminate()
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def start_streamlit_with_port_fallback(
    python: str,
    project_root: Path,
    env: dict[str, str],
    initial_port: int,
    max_attempts: int = 20,
) -> tuple[subprocess.Popen, int]:
    port = initial_port
    for attempt in range(max_attempts):
        proc = subprocess.Popen(
            [
                python,
                "-m",
                "streamlit",
                "run",
                "streamlit_app/app.py",
                "--server.port",
                str(port),
            ],
            cwd=project_root,
            env=env,
        )
        time.sleep(2.0)
        if proc.poll() is None:
            return proc, port
        if attempt < max_attempts - 1:
            print(
                f">> Streamlit exited on port {port} (code {proc.returncode}); trying {port + 1}..."
            )
            port += 1
            continue
        raise RuntimeError(
            f"Streamlit exited unexpectedly with code {proc.returncode} after trying ports "
            f"{initial_port}-{port}."
        )
    raise RuntimeError("Unable to start Streamlit.")


def is_port_available(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
            return True
        except OSError:
            return False


def pick_available_port(host: str, preferred_port: int, label: str) -> int:
    if is_port_available(host, preferred_port):
        return preferred_port
    for candidate in range(preferred_port + 1, preferred_port + 51):
        if is_port_available(host, candidate):
            print(
                f">> {label} port {preferred_port} is busy; using {candidate} instead."
            )
            return candidate
    raise RuntimeError(
        f"No free {label} port found in range {preferred_port}-{preferred_port + 50}."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="One-command local deploy: install, train artifacts, run API + Streamlit."
    )
    parser.add_argument("--skip-install", action="store_true", help="Skip pip install step.")
    parser.add_argument("--skip-train", action="store_true", help="Skip artifact training step.")
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=None,
        help="Optional sample size for faster artifact training.",
    )
    parser.add_argument("--api-host", default="127.0.0.1")
    parser.add_argument("--api-port", type=int, default=8000)
    parser.add_argument("--streamlit-port", type=int, default=8501)
    parser.add_argument("--health-timeout", type=int, default=120)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    python = sys.executable

    if not args.skip_install:
        run_step([python, "-m", "pip", "install", "-r", "requirements-serving.txt"], project_root)

    if not args.skip_train:
        train_command = [python, "-m", "serving.training"]
        if args.sample_rows is not None:
            train_command.extend(["--sample-rows", str(args.sample_rows)])
        run_step(train_command, project_root)

    api_port = pick_available_port(args.api_host, args.api_port, "API")
    streamlit_port = pick_available_port("127.0.0.1", args.streamlit_port, "Streamlit")
    if streamlit_port == api_port and args.api_host in {"127.0.0.1", "localhost"}:
        streamlit_port = pick_available_port("127.0.0.1", streamlit_port + 1, "Streamlit")

    api_url = f"http://{args.api_host}:{api_port}"
    health_url = f"{api_url}/health"

    api_process: subprocess.Popen | None = None
    streamlit_process: subprocess.Popen | None = None
    try:
        print("\n>> Starting FastAPI...")
        api_process = subprocess.Popen(
            [
                python,
                "-m",
                "uvicorn",
                "api.main:app",
                "--host",
                args.api_host,
                "--port",
                str(api_port),
            ],
            cwd=project_root,
        )

        print(f">> Waiting for API health at {health_url} ...")
        wait_for_api(health_url, args.health_timeout)

        env = os.environ.copy()
        env["CAR_API_URL"] = api_url
        print(">> Starting Streamlit...")
        streamlit_process, streamlit_port = start_streamlit_with_port_fallback(
            python=python,
            project_root=project_root,
            env=env,
            initial_port=streamlit_port,
        )

        print("\nLocal deploy is up:")
        print(f"   API:       {api_url}")
        print(f"   Swagger:   {api_url}/docs")
        print(f"   Streamlit: http://127.0.0.1:{streamlit_port}")
        print("\nPress Ctrl+C to stop both services.")

        while True:
            api_code = api_process.poll()
            ui_code = streamlit_process.poll()
            if api_code is not None:
                raise RuntimeError(f"FastAPI exited unexpectedly with code {api_code}")
            if ui_code is not None:
                raise RuntimeError(f"Streamlit exited unexpectedly with code {ui_code}")
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n>> Interrupted by user.")
    finally:
        terminate_process(streamlit_process, "Streamlit")
        terminate_process(api_process, "FastAPI")


if __name__ == "__main__":
    if os.name == "nt":
        signal.signal(signal.SIGINT, signal.default_int_handler)
    main()
