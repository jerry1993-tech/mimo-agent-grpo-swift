#!/usr/bin/env python3
"""Generic verify runner for harbor sidecar.

Extracts agent_output from session logs, builds post_state from SQLite DBs,
calls task-specific verify.py, writes /logs/verifier/reward.json.

Env switches:
  VERIFY_DETERMINISTIC=1 (default) — include deterministic rubrics
  VERIFY_AGENT_JUDGE=0   (default) — skip agent_judge rubrics
"""
import importlib.util, json, glob, os, sqlite3, sys, traceback


def extract_agent_output():
    """Extract the LAST assistant message from agent session logs."""
    last_text = ""
    for jf in sorted(glob.glob("/tmp/agent_output/sessions/**/*.jsonl", recursive=True)):
        with open(jf) as f:
            for line in f:
                try:
                    ev = json.loads(line)
                    if ev.get("type") == "assistant":
                        parts = []
                        for blk in ev.get("message", {}).get("content", []):
                            if blk.get("type") == "text" and blk.get("text", "").strip():
                                parts.append(blk["text"])
                        if parts:
                            last_text = "\n".join(parts)
                except Exception:
                    pass
    return last_text


def build_post_state(system_dir="/work/system"):
    post_state = {}
    if not os.path.isdir(system_dir):
        return post_state
    for mcp in sorted(os.listdir(system_dir)):
        db_path = os.path.join(system_dir, mcp, "state.db")
        if not os.path.isfile(db_path):
            continue
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]
        post_state[mcp] = {
            t: [dict(r) for r in conn.execute(f'SELECT * FROM "{t}" LIMIT 200000')]
            for t in tables
        }
        conn.close()
    return post_state


def filter_rubrics():
    """Filter rubrics.json based on VERIFY_* env vars before verify.py reads it."""
    rubrics_path = "/work/rubrics.json"
    if not os.path.isfile(rubrics_path):
        return
    run_det = os.environ.get("VERIFY_DETERMINISTIC", "1") == "1"
    run_judge = os.environ.get("VERIFY_AGENT_JUDGE", "0") == "1"
    if run_det and run_judge:
        return  # no filtering needed
    rubrics = json.loads(open(rubrics_path).read())
    filtered = []
    for r in rubrics:
        rtype = r.get("type", "deterministic")
        if rtype == "deterministic" and run_det:
            filtered.append(r)
        elif rtype == "agent_judge" and run_judge:
            filtered.append(r)
    with open(rubrics_path, "w") as f:
        json.dump(filtered, f, ensure_ascii=False)
    print(f"[verify] rubrics filtered: {len(filtered)}/{len(rubrics)} (det={run_det}, judge={run_judge})")


def run_verify(agent_output, post_state, verify_path="/work/verify.py"):
    if not os.path.isfile(verify_path):
        return {"error": "verify.py not found"}
    try:
        spec = importlib.util.spec_from_file_location("task_verify", verify_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.verify(
            env_path="/work",
            ground_truth=None,
            model_answer={},
            pre_state={},
            post_state=post_state,
            agent_output=agent_output,
        )
    except Exception as exc:
        return {"error": str(exc), "traceback": traceback.format_exc()}



def main():
    filter_rubrics()
    agent_output = extract_agent_output()
    post_state = build_post_state()
    result = run_verify(agent_output, post_state)

    # Masking contract: on ANY testbed-side failure (verify crashed, judge
    # unavailable) reward.json gets NO parseable "reward" key. The agent
    # runner then hits missing_or_invalid_reward_json → the rollout is
    # scored REWARD_TESTBED_CORRUPTED and the trainer masks the whole sequence,
    # instead of training on a false 0. Never print "[verify] score=" on this
    # path — downstream infra classifiers key on that line's absence.
    reward_error = None
    score = 0.0
    if "error" in result:
        reward_error = "verify_crashed"
        detail = {"reward_error": reward_error, "error": result["error"]}
        if "traceback" in result:
            detail["traceback"] = result["traceback"]
        print(f"[verify] ERROR: {result['error']}", file=sys.stderr)
        if "traceback" in result:
            print(result["traceback"], file=sys.stderr)
    elif result.get("reward_error"):
        reward_error = str(result["reward_error"])
        detail = {"reward_error": reward_error, "detail": result.get("detail")}
        print(f"[verify] judge unavailable: {reward_error}", file=sys.stderr)
    else:
        results_list = result.get("results", [])
        # Use verify()'s own weighted score (single source of truth).
        score = float(result.get("score", 0.0) or 0.0)
        detail = {
            "reward": score,
            "score": score,
            "strict_pass": result.get("strict_pass"),
            "deterministic_score": result.get("deterministic_score"),
            "judge_score": result.get("judge_score"),
            "judge_all_passed": result.get("judge_all_passed"),
            "passed": sum(1 for r in results_list if r.get("passed")),
            "total": len(results_list),
            "results": results_list,
        }
        for r in results_list:
            status = "PASS" if r.get("passed") else "FAIL"
            print(f"  [{status}] {r.get('id','?')}: {r.get('message','')}")

    payload = {"reward": score} if reward_error is None else {"reward_error": reward_error}
    with open("/logs/verifier/reward.json", "w") as f:
        json.dump(payload, f)
    # Structured per-rubric breakdown (reward.json stays minimal for the
    # harbor reward contract; this is for downstream analysis/filtering).
    with open("/logs/verifier/reward_detail.json", "w") as f:
        json.dump(detail, f, ensure_ascii=False, indent=2)
    if reward_error is None:
        print(f"[verify] score={score:.3f}")
    else:
        print(f"[verify] masked reward_error={reward_error}")


if __name__ == "__main__":
    main()
