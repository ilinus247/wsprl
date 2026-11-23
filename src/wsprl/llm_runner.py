"""Preview or execute a budgeted LLM trial plan with a persistent response cache."""
import argparse
from collections import Counter
from contextlib import closing
from dataclasses import dataclass
from decimal import Decimal, ROUND_CEILING
import fcntl
from hashlib import sha256
import json
from pathlib import Path
import sqlite3

from openai import APIStatusError, OpenAI
from pydantic import BaseModel, Field

from wsprl.llm_plan import Trial, build_trials
from wsprl.serialization import canonical_json


NANO_DOLLARS_PER_DOLLAR = 10**9


class BudgetExceeded(ValueError):
    pass


class UsageUncertain(ValueError):
    pass


class RequestConfig(BaseModel):
    model: str = Field(min_length=1)
    input_price: Decimal = Field(gt=0, le=10**6)
    output_price: Decimal = Field(gt=0, le=10**6)
    budget: Decimal = Field(default=Decimal("8"), gt=0, le=10**6)
    max_output_tokens: int = Field(default=1800, ge=1, le=100_000)
    framing_allowance: int = Field(default=128, ge=0, le=10_000)

    def cost(self, input_tokens: int, output_tokens: int) -> int:
        dollars = (input_tokens * self.input_price + output_tokens * self.output_price) / 10**6
        return int((dollars * NANO_DOLLARS_PER_DOLLAR).to_integral_value(rounding=ROUND_CEILING))

    def budget_units(self) -> int:
        return int(self.budget * NANO_DOLLARS_PER_DOLLAR)


@dataclass(frozen=True)
class Request:
    key: str
    trial: Trial
    body: dict
    estimated_input_tokens: int
    estimated_cost: int


def prepare_requests(trials: list[Trial], config: RequestConfig) -> list[Request]:
    requests = []
    for trial in trials:
        body = {
            "model": config.model,
            "messages": trial.messages,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": trial.task, "strict": True, "schema": trial.schema()},
            },
            "max_completion_tokens": config.max_output_tokens,
            "store": False,
        }
        # Bound message plus schema BPE length by UTF-8 bytes and allow for framing.
        estimated_input = len(canonical_json(body).encode("utf-8")) + config.framing_allowance
        identity = canonical_json({"trial": trial.assignment(), "body": body})
        key = sha256(identity.encode("utf-8")).hexdigest()
        requests.append(Request(
            key, trial, body, estimated_input, config.cost(estimated_input, config.max_output_tokens),
        ))
    if len({request.key for request in requests}) != len(requests):
        raise ValueError("Duplicate trial identities in request plan")
    return requests


class Ledger:
    def __init__(
        self, db: sqlite3.Connection, requests: list[Request], config: RequestConfig, experiment_id: str,
    ):
        self.db = db
        self.config = config
        self.experiment_id = experiment_id
        self.plan = canonical_json({
            "schema_version": 1,
            "configuration": config.model_dump(mode="json"),
            "trials": [{"key": item.key, **item.trial.assignment()} for item in requests],
        })
        db.executescript("""
            CREATE TABLE IF NOT EXISTS project_budget (
                id INTEGER PRIMARY KEY CHECK (id = 1), budget_nusd INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS experiments (
                id TEXT PRIMARY KEY, plan TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS requests (
                experiment_id TEXT NOT NULL REFERENCES experiments(id),
                key TEXT NOT NULL,
                status TEXT NOT NULL,
                request_json TEXT NOT NULL,
                response_body TEXT,
                prediction_json TEXT,
                request_id TEXT,
                failure_reason TEXT,
                estimated_input_tokens INTEGER NOT NULL,
                max_output_tokens INTEGER NOT NULL,
                input_tokens INTEGER,
                output_tokens INTEGER,
                reserved_nusd INTEGER NOT NULL,
                billed_nusd INTEGER,
                started_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                finished_at TEXT,
                PRIMARY KEY (experiment_id, key)
            );
        """)
        with db:
            db.execute("INSERT OR IGNORE INTO project_budget VALUES (1, ?)", (config.budget_units(),))
            budget, = db.execute("SELECT budget_nusd FROM project_budget WHERE id = 1").fetchone()
            if budget != config.budget_units():
                raise ValueError("Project budget differs from the saved ledger budget")
            db.execute("INSERT OR IGNORE INTO experiments VALUES (?, ?)", (experiment_id, self.plan))
            saved, = db.execute("SELECT plan FROM experiments WHERE id = ?", (experiment_id,)).fetchone()
            if saved != self.plan:
                raise ValueError("Saved experiment differs from this plan; choose a new experiment ID")

    def status(self, key: str) -> str | None:
        row = self.db.execute("SELECT status FROM requests WHERE experiment_id=? AND key=?",
                              (self.experiment_id, key)).fetchone()
        return row[0] if row else None

    def check_resolved(self) -> None:
        unresolved = self.db.execute(
            "SELECT 1 FROM requests WHERE status IN ('uncertain', 'reserved') LIMIT 1"
        ).fetchone()
        if unresolved:
            raise UsageUncertain("A previous request has unresolved billing; inspect the ledger before further calls")

    def reserve(self, request: Request) -> bool:
        with self.db:
            self.db.execute("BEGIN IMMEDIATE")
            if self.status(request.key) is not None:
                return False
            committed, = self.db.execute(
                "SELECT COALESCE(SUM(COALESCE(billed_nusd, reserved_nusd)), 0) FROM requests"
            ).fetchone()
            if committed + request.estimated_cost > self.config.budget_units():
                raise BudgetExceeded("Remaining project budget cannot cover the next request estimate")
            self.db.execute("""
                INSERT INTO requests
                    (experiment_id, key, status, request_json, estimated_input_tokens, max_output_tokens, reserved_nusd)
                VALUES (?, ?, 'reserved', ?, ?, ?, ?)
            """, (
                self.experiment_id, request.key, canonical_json(request.body), request.estimated_input_tokens,
                self.config.max_output_tokens, request.estimated_cost,
            ))
        return True

    def finish(
        self, request: Request, *, status: str, response_body=None, prediction=None,
        request_id=None, failure_reason=None, input_tokens=None, output_tokens=None,
    ) -> int | None:
        cost = None
        if (
            type(input_tokens) is int and input_tokens > 0
            and type(output_tokens) is int and output_tokens >= 0
        ):
            cost = self.config.cost(input_tokens, output_tokens)
        with self.db:
            self.db.execute("""
                UPDATE requests SET status=?, response_body=?, prediction_json=?,
                    request_id=?, failure_reason=?, input_tokens=?, output_tokens=?,
                    billed_nusd=?, finished_at=CURRENT_TIMESTAMP
                WHERE experiment_id=? AND key=? AND status='reserved'
            """, (
                status, response_body, prediction, request_id, failure_reason,
                input_tokens, output_tokens, cost, self.experiment_id, request.key,
            ))
        return cost


def execute_requests(
    requests: list[Request], config: RequestConfig, ledger_path: Path,
    request_limit: int, client_factory=OpenAI, *, experiment_id: str = "pilot",
) -> Counter:
    if request_limit < 1:
        raise ValueError("Request limit must be positive")
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    lock_path = ledger_path.with_suffix(ledger_path.suffix + ".lock")
    with lock_path.open("a") as lock, closing(sqlite3.connect(ledger_path)) as db:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("Another runner is using this experiment ledger") from error
        ledger = Ledger(db, requests, config, experiment_id)
        ledger.check_resolved()
        client = None
        try:
            for request in requests:
                status = ledger.status(request.key)
                if status is not None:
                    counts[f"cached_{status}"] += 1
                    continue
                if counts["sent"] >= request_limit:
                    break
                if not ledger.reserve(request):
                    continue
                if client is None:
                    try:
                        client = client_factory(base_url="https://api.openai.com/v1", max_retries=0)
                        client.models.retrieve(config.model)
                    except Exception as error:
                        # Model preflight runs before any completion could be submitted.
                        with db:
                            db.execute(
                                "DELETE FROM requests WHERE experiment_id=? AND key=? AND status='reserved'",
                                (experiment_id, request.key),
                            )
                        raise ValueError(f"Model availability check failed ({type(error).__name__})") from None
                counts["sent"] += 1
                call_completion(client, ledger, request, counts)
        finally:
            if client is not None:
                client.close()
    return counts


def call_completion(client, ledger: Ledger, request: Request, counts: Counter) -> None:
    response_body = request_id = None
    try:
        raw = client.chat.completions.with_raw_response.create(**request.body)
        response_body = raw.text
        request_id = raw.headers.get("x-request-id")
        response = raw.parse()
    except Exception as error:
        if isinstance(error, APIStatusError):
            response_body = error.response.text
            request_id = error.request_id
        ledger.finish(
            request, status="uncertain", response_body=response_body, request_id=request_id,
            failure_reason=type(error).__name__,
        )
        raise UsageUncertain("Completion failed with unknown billing; its reservation was retained") from None

    usage = response.usage
    input_tokens = usage.prompt_tokens if usage else None
    output_tokens = usage.completion_tokens if usage else None
    prediction = None
    reason = None
    status = "valid"
    try:
        choice, = response.choices
        if choice.finish_reason != "stop" or choice.message.refusal or choice.message.content is None:
            raise ValueError("Refused, incomplete, or missing response")
        prediction = canonical_json(request.trial.validate(choice.message.content).model_dump())
    except ValueError:
        status = "invalid"
        reason = "Refusal, incomplete response, or failed schema/case validation"
    if (
        type(input_tokens) is not int or input_tokens <= 0
        or type(output_tokens) is not int or output_tokens < 0
    ):
        status = "uncertain"
        reason = "Missing or invalid token usage"
    cost = ledger.finish(
        request, status=status, response_body=response_body, prediction=prediction,
        request_id=request_id, failure_reason=reason, input_tokens=input_tokens,
        output_tokens=output_tokens,
    )
    counts[status] += 1
    if cost is None:
        raise UsageUncertain("Token usage unavailable; the estimated cost remains reserved")
    if cost > request.estimated_cost:
        raise BudgetExceeded("Reported usage exceeded the estimate; recorded actual cost and stopped")


def dollars(units: int) -> str:
    return str(Decimal(units) / NANO_DOLLARS_PER_DOLLAR)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-root", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--input-price", type=Decimal, required=True, help="USD per million input tokens")
    parser.add_argument("--output-price", type=Decimal, required=True, help="USD per million output tokens")
    parser.add_argument("--budget", type=Decimal, default=Decimal("8"))
    parser.add_argument("--max-output-tokens", type=int, default=1800)
    parser.add_argument("--framing-allowance", type=int, default=128)
    parser.add_argument("--replicates", type=int, default=1)
    parser.add_argument("--order-seed", type=int, default=44)
    parser.add_argument("--task", choices=("individual", "paired"), action="append")
    parser.add_argument("--paired-intervention", default="mask_probability")
    parser.add_argument("--paired-severity", type=float, default=0.3)
    parser.add_argument("--failure-fraction", type=float, default=0.25)
    parser.add_argument("--ledger", type=Path)
    parser.add_argument("--experiment-id", default="pilot")
    launch = parser.add_mutually_exclusive_group()
    launch.add_argument("--dry-run", action="store_true", help="Preview the plan locally (default)")
    launch.add_argument("--execute", action="store_true", help="Submit requests after explicit approval")
    parser.add_argument("--request-limit", type=int, help="Maximum new completions this invocation")
    args = parser.parse_args()
    if args.execute and (args.ledger is None or args.request_limit is None or args.request_limit < 1):
        parser.error("--execute requires --ledger and a positive --request-limit")

    try:
        config = RequestConfig(
            model=args.model, input_price=args.input_price, output_price=args.output_price,
            budget=args.budget, max_output_tokens=args.max_output_tokens,
            framing_allowance=args.framing_allowance,
        )
        trials = build_trials(
            args.batch_root, replicates=args.replicates, order_seed=args.order_seed,
            tasks=tuple(args.task or ("individual", "paired")),
            paired_case=(args.paired_intervention, args.paired_severity),
            failure_fraction=args.failure_fraction,
        )
        requests = prepare_requests(trials, config)
        if args.execute:
            counts = execute_requests(
                requests, config, args.ledger, args.request_limit, experiment_id=args.experiment_id,
            )
            print(json.dumps(counts, sort_keys=True))
            return
        estimate = sum(request.estimated_cost for request in requests)
        print(json.dumps({
            "mode": "dry-run",
            "requests": len(requests),
            "by_task": dict(Counter(request.trial.task for request in requests)),
            "estimated_cost_usd": dollars(estimate),
            "first_request_estimated_cost_usd": dollars(requests[0].estimated_cost),
            "budget_usd": str(config.budget),
            "within_budget_estimate": estimate <= config.budget_units(),
        }, indent=2))
    except (ValueError, OSError, sqlite3.Error) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
