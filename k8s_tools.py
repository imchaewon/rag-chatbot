import subprocess


def _run(args: list[str], timeout: int = 60) -> tuple[str, str, int]:
    result = subprocess.run(
        ["kubectl"] + args,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return result.stdout.strip(), result.stderr.strip(), result.returncode


def get_deployments(namespace: str = None) -> str:
    args = ["get", "deployments", "-o", "wide"]
    args += ["-A"] if namespace is None else ["-n", namespace]
    out, err, rc = _run(args)
    return out if rc == 0 else f"오류: {err}"


def get_pods(namespace: str = None) -> str:
    args = ["get", "pods", "-o", "wide"]
    args += ["-A"] if namespace is None else ["-n", namespace]
    out, err, rc = _run(args)
    return out if rc == 0 else f"오류: {err}"


def restart_deployment(name: str, namespace: str = "default") -> str:
    _, err, rc = _run(["rollout", "restart", f"deployment/{name}", "-n", namespace])
    if rc != 0:
        return f"재시작 실패: {err}"
    out, err, rc = _run(
        ["rollout", "status", f"deployment/{name}", "-n", namespace, "--timeout=60s"],
        timeout=70,
    )
    return f"재시작 완료\n{out}" if rc == 0 else f"재시작 후 상태 확인 실패: {err}"


def scale_deployment(name: str, replicas: int, namespace: str = "default") -> str:
    _, err, rc = _run(
        ["scale", f"deployment/{name}", f"--replicas={replicas}", "-n", namespace]
    )
    if rc != 0:
        return f"스케일 조정 실패: {err}"
    label = "중지" if replicas == 0 else f"{replicas}개로 스케일 조정"
    return f"{namespace}/{name} {label} 완료"


def get_logs(name: str, namespace: str = "default", tail: int = 20) -> str:
    pod, err, rc = _run([
        "get", "pods", "-n", namespace,
        "-l", f"app={name}",
        "-o", "jsonpath={.items[0].metadata.name}",
    ])
    if rc != 0 or not pod:
        return f"Pod를 찾을 수 없습니다 (label app={name}): {err}"
    out, err, rc = _run(["logs", pod, "-n", namespace, f"--tail={tail}"])
    return out if rc == 0 else f"로그 조회 실패: {err}"


def find_deployment(name: str) -> tuple[str, str] | None:
    """전체 네임스페이스에서 deployment 이름으로 (name, namespace) 반환. 없으면 None."""
    out, _, rc = _run([
        "get", "deployments", "-A",
        "-o", "jsonpath={range .items[*]}{.metadata.name}{'|'}{.metadata.namespace}{'\\n'}{end}",
    ])
    if rc != 0:
        return None
    for line in out.splitlines():
        parts = line.split("|")
        if len(parts) == 2 and parts[0].strip() == name:
            return parts[0].strip(), parts[1].strip()
    return None
