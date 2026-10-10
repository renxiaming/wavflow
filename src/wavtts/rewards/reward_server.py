from __future__ import annotations

import argparse
import json
import socketserver
import threading
import traceback

from wavtts.rewards import SimWerReward


class RewardHandler(socketserver.StreamRequestHandler):
    def handle(self):
        for line in self.rfile:
            try:
                request = json.loads(line.decode("utf-8"))
                if request.get("op") == "ping":
                    response = {"ok": True, "status": "ready"}
                elif request.get("op") == "score":
                    with self.server.reward_lock:  # type: ignore[attr-defined]
                        result = self.server.rewarder.score(  # type: ignore[attr-defined]
                            generated_paths=request["generated_paths"],
                            prompt_paths=request["prompt_paths"],
                            target_texts=request["target_texts"],
                            languages=request.get("languages"),
                            reference_paths=request.get("reference_paths"),
                        )
                    response = {
                        "ok": True,
                        "result": {
                            key: value.detach().cpu().tolist()
                            for key, value in result.items()
                        },
                    }
                else:
                    raise ValueError(f"Unknown reward server op: {request.get('op')}")
            except Exception as exc:
                traceback.print_exc()
                response = {"ok": False, "error": repr(exc)}
            self.wfile.write((json.dumps(response, ensure_ascii=False) + "\n").encode("utf-8"))
            self.wfile.flush()


class RewardServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, server_address, rewarder):
        super().__init__(server_address, RewardHandler)
        self.rewarder = rewarder
        self.reward_lock = threading.Lock()


def main():
    parser = argparse.ArgumentParser(description="Resident WER/SIM/MOS reward server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=29631)
    parser.add_argument("--sim-device", default="cuda:0")
    parser.add_argument("--wer-device", default="cuda:0")
    parser.add_argument("--mos-device", default="cuda:0")
    parser.add_argument(
        "--sim-root",
        default=(
            "/home/A02_tmpdata0/xmren/code/grpo_flow/wavtts3.1_grpo/"
            "portable_wavtts_grpo/third_party/sim_wer_test"
        ),
    )
    parser.add_argument("--sim-python", default=None)
    parser.add_argument("--wer-python", default=None)
    parser.add_argument("--mos-python", default=None)
    parser.add_argument("--wer-weight", type=float, default=0.2)
    parser.add_argument("--sim-weight", type=float, default=0.8)
    parser.add_argument("--mos-weight", type=float, default=0.0)
    args = parser.parse_args()

    rewarder = SimWerReward(
        device=args.wer_device,
        sim_device=args.sim_device,
        wer_device=args.wer_device,
        mos_device=args.mos_device,
        sim_root=args.sim_root,
        wer_weight=args.wer_weight,
        sim_weight=args.sim_weight,
        mos_weight=args.mos_weight,
        backend="resident",
        sim_python=args.sim_python,
        wer_python=args.wer_python,
        mos_python=args.mos_python,
    )
    server = RewardServer((args.host, args.port), rewarder)
    print(
        f"Reward server listening on {args.host}:{args.port}; "
        f"sim={args.sim_device} wer={args.wer_device} mos={args.mos_device}",
        flush=True,
    )
    try:
        server.serve_forever()
    finally:
        server.shutdown()
        server.server_close()
        rewarder.close()


if __name__ == "__main__":
    main()
