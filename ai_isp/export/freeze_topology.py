"""以 JSON + safetensors 冻结发布拓扑，避免依赖 Python Pickle。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from safetensors.torch import save_file

from ai_isp.models.mobile_nafnet import MobileNAFNetW16


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def freeze_topology(model: MobileNAFNetW16, output_dir: str | Path) -> dict[str, object]:
    """保存连续权重、拓扑 JSON 及其 Hash。"""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    weights_path = output_dir / "model_fp32.safetensors"
    topology_path = output_dir / "topology.json"
    state = {name: value.detach().cpu().contiguous() for name, value in model.state_dict().items()}
    save_file(state, str(weights_path))
    topology = model.topology_manifest()
    topology["weights_file"] = weights_path.name
    topology["weights_sha256"] = sha256_file(weights_path)
    topology_path.write_text(json.dumps(topology, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"weights": str(weights_path), "topology": str(topology_path), **topology}

