from wavtts.model.lance.aslp.data.mp3data import Mp3Data
from wavtts.model.lance.aslp.data.npydata import IntData

_CO_CLS_MAP = {
    "Mp3Data": Mp3Data,
    "IntData": IntData,
}


def load_li(li_path: str) -> list[str]:
    with open(li_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def load_co(co_path: str) -> list[tuple[str, type]]:
    with open(co_path, "r", encoding="utf-8") as f:
        co = [line.strip().split("|") for line in f if line.strip()]
    result: list[tuple[str, type]] = []
    for entry in co:
        cls_name = entry[1]
        if cls_name not in _CO_CLS_MAP:
            raise KeyError(f"Unknown Lance data class {cls_name!r} in {co_path}")
        result.append((entry[0], _CO_CLS_MAP[cls_name]))
    return result
