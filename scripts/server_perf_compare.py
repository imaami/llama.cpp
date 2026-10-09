#!/usr/bin/env python3
"""Repeat fixed, uncached requests against a dedicated llama-server; compare saved runs."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, Request, build_opener


HTTP = build_opener(ProxyHandler({}))


COUNTERS = (
    "prompt_tokens_total", "tokens_predicted_total",
    "spec_decode_num_draft_tokens_total", "spec_decode_num_accepted_tokens_total",
    "spec_decode_num_drafts_total",
)
GAUGES = ("requests_processing", "requests_deferred")
FIELDS = (
    "tokens", "tokens_predicted", "tokens_evaluated", "generation_settings",
    "truncated", "stop_type", "timings",
)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def request(base, path, timeout, body=None, as_json=True):
    data = None if body is None else json.dumps(body).encode()
    req = Request(base + path, data=data, headers={"Content-Type": "application/json"})
    try:
        with HTTP.open(req, timeout=timeout) as response:
            result = response.read().decode()
    except HTTPError as exc:
        # Do not echo an error body which may contain the prompt.
        raise ValueError(f"{path}: HTTP {exc.code}; check server log (use --metrics and --slots)") from exc
    return json.loads(result) if as_json else result


def metrics(base, timeout):
    values = {}
    for line in request(base, "/metrics", timeout, as_json=False).splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0].startswith("llamacpp:"):
            name = parts[0].removeprefix("llamacpp:")
            if name in COUNTERS + GAUGES:
                values[name] = float(parts[1])
    if any(name not in values or not math.isfinite(values[name]) for name in COUNTERS + GAUGES):
        raise ValueError("server metrics are missing required counters")
    if any(values[name] != 0 for name in GAUGES):
        raise ValueError("server is busy; use a dedicated idle server with --parallel 1")
    return values


def checked_sample(response, before, after, n_predict, elapsed, n_ctx):
    timing = response["timings"]
    count = timing["predicted_n"]
    if response["truncated"] or response["tokens_evaluated"] + count >= n_ctx:
        raise ValueError("context exhausted; shorten the prompt/output or enlarge the server context")
    if response["stop_type"] != "limit" or count != n_predict:
        raise ValueError(f"expected {n_predict} generated tokens, got {count} ({response['stop_type']}); discard this run")
    if response["tokens_predicted"] != count or len(response["tokens"]) != count:
        raise ValueError("response token counts disagree")
    if timing["cache_n"] != 0 or timing["prompt_n"] != response["tokens_evaluated"]:
        raise ValueError("prompt was cached or not fully evaluated")
    if any(not math.isfinite(timing[key]) or timing[key] <= 0 for key in ("predicted_ms", "predicted_per_second")):
        raise ValueError("invalid generation timing")
    delta = {key: after[key] - before[key] for key in COUNTERS}
    expected = {
        "prompt_tokens_total": timing["prompt_n"],
        "tokens_predicted_total": count,
        "spec_decode_num_draft_tokens_total": timing.get("draft_n", 0),
        "spec_decode_num_accepted_tokens_total": timing.get("draft_n_accepted", 0),
    }
    if any(value < 0 or value != int(value) for value in delta.values()) or any(delta[key] != value for key, value in expected.items()):
        raise ValueError("metrics disagree with this request (other traffic, restart, or rounded large counters); restart a dedicated server")
    drafted = delta["spec_decode_num_draft_tokens_total"]
    accepted = delta["spec_decode_num_accepted_tokens_total"]
    steps = delta["spec_decode_num_drafts_total"]
    if accepted > drafted or (drafted > 0 and steps <= 0):
        raise ValueError("invalid speculative decoding counters")
    return {
        "timings": timing, "wall_seconds": elapsed,
        "output_tokens_sha256": digest(response["tokens"]),
        "verification_steps": int(steps),
        "draft_acceptance": accepted / drafted if drafted else None,
        "mean_accepted_length": 1 + accepted / steps if steps else None,
    }


def settings_signature(settings):
    settings = dict(settings)
    for key in ("generation_prompt", "grammar", "grammar_triggers", "dry_sequence_breakers"):
        if key in settings:
            settings[key + "_sha256"] = digest(settings.pop(key))
    return settings


def run(args):
    base = args.url.rstrip("/")
    parsed = urlsplit(base)
    if parsed.scheme != "http" or parsed.hostname not in ("localhost", "127.0.0.1", "::1") or parsed.path or parsed.query or parsed.fragment or parsed.username:
        raise ValueError("--url must be a local HTTP server origin, e.g. http://127.0.0.1:8080")
    if args.output.exists():
        raise ValueError(f"output already exists: {args.output}")
    prompt_bytes = args.prompt_file.read_bytes()
    prompt = prompt_bytes.decode("utf-8")
    if not prompt.strip():
        raise ValueError("prompt file is empty")
    props = request(base, "/props", args.timeout)
    slots = request(base, "/slots", args.timeout)
    if props["total_slots"] != 1 or len(slots) != 1 or slots[0]["is_processing"]:
        raise ValueError("use a dedicated idle server with --parallel 1")
    n_ctx = slots[0]["n_ctx"]
    tokens = request(base, "/tokenize", args.timeout, {"content": prompt, "add_special": True, "parse_special": True})["tokens"]
    if len(tokens) + args.tokens >= n_ctx:
        raise ValueError("prompt plus requested output does not fit the server context")
    body = {
        "prompt": tokens, "n_predict": args.tokens, "seed": args.seed,
        "cache_prompt": False, "ignore_eos": True, "stop": [], "stream": False,
        "t_max_predict_ms": 0, "return_tokens": True, "response_fields": FIELDS,
    }
    record = {
        "schema": 1, "label": args.label, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "server": {key: props.get(key) for key in ("build_info", "model_path")},
        "n_ctx": n_ctx, "prompt_sha256": hashlib.sha256(prompt_bytes).hexdigest(),
        "prompt_tokens_sha256": digest(tokens), "prompt_tokens": len(tokens),
        "request": {key: value for key, value in body.items() if key != "prompt"},
        "artifacts": [], "warmup": [], "samples": [],
    }
    for path in args.artifact:
        sha = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                sha.update(chunk)
        record["artifacts"].append({"path": str(path), "bytes": path.stat().st_size, "sha256": sha.hexdigest()})
    previous = None
    for i in range(args.warmup + args.repeats):
        before = metrics(base, args.timeout)
        if previous is not None and any(before[key] != previous[key] for key in COUNTERS):
            raise ValueError("other requests ran between samples; discard this run")
        start = time.monotonic()
        response = request(base, "/completion", args.timeout, body)
        elapsed = time.monotonic() - start
        after = metrics(base, args.timeout)
        sample = checked_sample(response, before, after, args.tokens, elapsed, n_ctx)
        settings = settings_signature(response["generation_settings"])
        if "generation_settings" in record and settings != record["generation_settings"]:
            raise ValueError("effective generation settings changed between requests")
        record["generation_settings"] = settings
        kind = "warmup" if i < args.warmup else "samples"
        record[kind].append(sample)
        previous = after
        print(f"{args.label} {kind} {len(record[kind])}: {sample['timings']['predicted_per_second']:.2f} tok/s, acceptance={sample['draft_acceptance']}", file=sys.stderr)
    # Exclusive creation avoids replacing an earlier measurement.
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(record, output, indent=2, allow_nan=False)
        output.write("\n")
    print(args.output)


def compare(args):
    records = [json.loads(path.read_text()) for path in args.files]
    baseline = records[0]
    fields = ("schema", "prompt_sha256", "prompt_tokens_sha256", "prompt_tokens", "request", "n_ctx", "generation_settings")
    for record in records:
        if record.get("schema") != 1 or not record["samples"]:
            raise ValueError("unsupported or empty run record")
        changed = [key for key in fields if record[key] != baseline[key]]
        if changed:
            raise ValueError(f"{record['label']}: incomparable fields: {', '.join(changed)}")
    baseline_rate = statistics.median(sample["timings"]["predicted_per_second"] for sample in baseline["samples"])
    baseline_hashes = {sample["output_tokens_sha256"] for sample in baseline["samples"]}
    print("label\tmedian tok/s\tmin..max\tchange\tacceptance\tmean len\toutput")
    for record in records:
        samples = record["samples"]
        rates = [sample["timings"]["predicted_per_second"] for sample in samples]
        drafted = sum(sample["timings"].get("draft_n", 0) for sample in samples)
        accepted = sum(sample["timings"].get("draft_n_accepted", 0) for sample in samples)
        steps = sum(sample["verification_steps"] for sample in samples)
        rate = statistics.median(rates)
        hashes = {sample["output_tokens_sha256"] for sample in samples}
        output = "same" if len(hashes) == len(baseline_hashes) == 1 and hashes == baseline_hashes else "DIFFERS"
        acceptance = f"{accepted / drafted:.4f}" if drafted else "n/a"
        mean_len = f"{1 + accepted / steps:.3f}" if steps else "n/a"
        print(f"{record['label']}\t{rate:.2f}\t{min(rates):.2f}..{max(rates):.2f}\t{100 * (rate / baseline_rate - 1):+.2f}%\t{acceptance}\t{mean_len}\t{output}")
        if Path(record["server"].get("model_path") or "").name != Path(baseline["server"].get("model_path") or "").name:
            print(f"warning: {record['label']} has a different model filename; verify identical GGUF bytes", file=sys.stderr)
    print("Different output or acceptance means the decode workload changed; this alone does not isolate kernel performance.")


def positive(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    measure = sub.add_parser("run", help="send repeat requests to a dedicated, already-running local server")
    measure.add_argument("--url", default="http://127.0.0.1:8080")
    measure.add_argument("--prompt-file", type=Path, required=True, help="UTF-8 prompt, including chat template tokens if desired")
    measure.add_argument("--label", required=True)
    measure.add_argument("--output", type=Path, required=True)
    measure.add_argument("--tokens", type=positive, default=2048)
    measure.add_argument("--seed", type=int, default=1234)
    measure.add_argument("--repeats", type=positive, default=3)
    measure.add_argument("--warmup", type=int, choices=range(0, 11), default=1)
    measure.add_argument("--timeout", type=positive, default=3600, help="HTTP timeout in seconds")
    measure.add_argument("--artifact", type=Path, action="append", default=[], help="optional binary/library to hash; repeatable")
    diff = sub.add_parser("compare", help="compare compatible saved runs; no server needed")
    diff.add_argument("files", type=Path, nargs="+")
    args = parser.parse_args()
    if args.command == "run" and not 0 <= args.seed < 4294967295:
        parser.error("--seed must be in 0..4294967294 (exclude the random seed sentinel)")
    try:
        (run if args.command == "run" else compare)(args)
    except (OSError, URLError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f"error: {exc}\n")


if __name__ == "__main__":
    main()
