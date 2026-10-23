"""Publish an evidence-checked research candidate, or atomically roll back."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.config import Settings
from app.model_registry import RegistryError, register_candidate, rollback_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    operation = parser.add_mutually_exclusive_group(required=True)
    operation.add_argument("--candidate", type=Path)
    operation.add_argument("--rollback", metavar="MODEL_ID")
    parser.add_argument("--evaluation", type=Path, help="Directory containing the complete frozen evaluation artifacts")
    parser.add_argument("--integration", type=Path, help="integration-check.json for this exact candidate")
    parser.add_argument("--storage", type=Path, default=Settings.from_env().storage)
    parser.add_argument("--research-only", action="store_true")
    parser.add_argument("--usage", choices=("research_only",))
    parser.add_argument("--operator", required=True)
    parser.add_argument("--reason", required=True)
    args = parser.parse_args()
    try:
        if args.rollback:
            if args.evaluation or args.integration or args.research_only or args.usage:
                parser.error("--rollback uses the stored immutable version; candidate/evaluation/usage options do not apply")
            active = rollback_model(args.rollback, args.storage, operator=args.operator, reason=args.reason)
        else:
            if not args.evaluation or not args.integration:
                parser.error("--candidate requires --evaluation and --integration")
            if not args.research_only and not args.usage:
                parser.error("registration requires explicit --research-only or --usage research_only")
            active = register_candidate(args.candidate, args.evaluation, args.integration, args.storage,
                                        operator=args.operator, reason=args.reason,
                                        research_only=args.research_only, usage=args.usage)
    except (RegistryError, OSError, RuntimeError) as exc:
        parser.exit(2, f"Registration refused: {exc}\n")
    print(json.dumps({"operation": "rollback" if args.rollback else "register", "model_id": active["model_id"],
                      "checkpoint_sha256": active["checkpoint_sha256"], "usage": active.get("usage"),
                      "target_passed": active.get("evaluation", {}).get("target_passed"),
                      "active": str(args.storage / "models" / "active.json")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
