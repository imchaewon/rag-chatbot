import subprocess


def _run(args: list[str], timeout: int = 60) -> tuple[str, str, int]:
    result = subprocess.run(
        ["kubectl"] + args,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return result.stdout.strip(), result.stderr.strip(), result.returncode


def _to_md_table(text: str) -> str:
    """kubectl 표 출력을 마크다운 테이블로 변환 (위치 기반 파싱)."""
    lines = [l for l in text.splitlines() if l.strip()]
    if len(lines) < 2:
        return text

    header = lines[0]
    # 헤더에서 각 컬럼의 시작 위치 추출
    positions = []
    i = 0
    while i < len(header):
        if header[i] != ' ':
            positions.append(i)
            while i < len(header) and header[i] != ' ':
                i += 1
            while i < len(header) and header[i] == ' ':
                i += 1
        else:
            i += 1

    def slice_row(line):
        cells = []
        for j, pos in enumerate(positions):
            end = positions[j + 1] if j + 1 < len(positions) else len(line)
            cells.append(line[pos:end].strip() if pos < len(line) else "")
        return cells

    headers = slice_row(header)
    rows = [slice_row(l) for l in lines[1:]]

    col_widths = [max(len(h), max((len(r[i]) for r in rows), default=3), 3)
                  for i, h in enumerate(headers)]

    def fmt_row(cells):
        return "| " + " | ".join(c.ljust(col_widths[i]) for i, c in enumerate(cells)) + " |"

    sep = "| " + " | ".join("-" * w for w in col_widths) + " |"
    return "\n".join([fmt_row(headers), sep] + [fmt_row(r) for r in rows])


def get_deployments(namespace: str = None) -> str:
    args = ["get", "deployments"]
    if namespace == "all":
        args += ["-A"]
    elif namespace:
        args += ["-n", namespace]
    out, err, rc = _run(args)
    return _to_md_table(out) if rc == 0 else f"오류: {err}"


def get_pods(namespace: str = None) -> str:
    args = ["get", "pods"]
    if namespace == "all":
        args += ["-A"]
    elif namespace:
        args += ["-n", namespace]
    out, err, rc = _run(args)
    return _to_md_table(out) if rc == 0 else f"오류: {err}"


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
