"""The small-step interview of a model that declares inputs.

A model with ``input_keys`` asks its questions in parts; this package keeps the state the
questions do not: which parts are saved, the tickets that address them and when the full
answer is published. The model owns the number of parts, their text, their schema and the
checks; the interview owns their order and their storage, so every model with inputs is
interviewed the same way and the server knows nothing about the model.

One writer per model and date is assumed: the workspace belongs to one server process.
"""

from threading import RLock

from mst import inputs, store

from . import journal, state, tickets, versions

LOCK = RLock()


class Session:
    """The interview of one model on one date, driven only through the model's own hooks."""

    def __init__(self, model, context):
        self.model = model
        self.context = context

    @property
    def folder(self):
        """Where the saved parts of this model and date live."""
        return (
            self.context.inputs
            / ".steps"
            / self.model.model_id
            / self.context.as_of.isoformat()
        )

    def _parts_path(self, key):
        return self.folder / f"{key}.json"

    def _steps(self, key):
        specs = self.model.input_steps(self.context, key)
        names = [row["id"] for row in specs]
        if not names or len(set(names)) != len(names):
            raise ValueError("model input steps must have unique ids and cannot be empty")
        return specs

    def _declared(self, key):
        if key not in self.model.input_keys(self.context.as_of):
            raise ValueError(
                f"model {self.model.model_id!r} declares no input {key!r} on "
                f"{self.context.as_of.isoformat()}"
            )
        return key

    def _touch(self, key, agent):
        """Check the claim of ``key`` and extend the caller's own; a claim is own or refused.

        An agent without a claim works too: a single agent answers without owning anything,
        and a claim is what turns the ownership into a rule between several agents.
        """
        current = state.live(state.read(self.folder), key)
        if current is None:
            return None
        if current["agent"] != agent:
            raise ValueError(
                f"{key!r} is claimed by {current['agent']} until {current['expires']}: "
                "release it or leave the claim to its owner"
            )
        return state.claim(self.folder, key, agent)

    def claim(self, key, agent, minutes=45):
        """Give one input to one agent, so a second agent does not answer the same shock."""
        self._declared(key)
        if not agent:
            raise ValueError("claim needs agent: the name of the answering agent")
        if key in inputs.read(self.context.inputs, self.context.as_of):
            raise ValueError(f"input {key!r} is already published: reopen it to change it")
        with LOCK:
            claim = state.claim(self.folder, key, agent, minutes)
            journal.record(self.folder, "claim", key=key, agent=agent)
            return {
                "model": self.model.model_id,
                "as_of": self.context.as_of.isoformat(),
                "key": key,
                "claim": claim,
            }

    def release(self, key, agent):
        """Drop the claim of ``agent``; another agent's claim is refused, not overwritten."""
        self._declared(key)
        with LOCK:
            dropped = state.release(self.folder, key, agent)
            if dropped:
                journal.record(self.folder, "release", key=key, agent=agent)
        return {
            "model": self.model.model_id,
            "as_of": self.context.as_of.isoformat(),
            "key": key,
            "released": bool(dropped),
        }

    def journal(self, key=None, limit: int = 50):
        """The last events of the interview or of one key, oldest first."""
        return {
            "model": self.model.model_id,
            "as_of": self.context.as_of.isoformat(),
            "key": key,
            "events": journal.tail(self.folder, key, limit),
        }

    def progress(self):
        """Every declared input with its owner, answered parts and publication."""
        day = self.context.as_of
        found = {
            row["key"]: row
            for row in inputs.status(self.context.inputs, self.model, day)["keys"]
        }
        audited = {row["key"]: row for row in self.audit()["keys"]}
        known = state.read(self.folder)
        rows = []
        for key in self.model.input_keys(day):
            specs = self._steps(key)
            parts = store.read_json(self._parts_path(key)) or {}
            published = found.get(key, {})
            claim = state.live(known, key)
            accepted = known["accepted"].get(key)
            findings = audited.get(key, {}).get("findings", [])
            warnings = len(findings)
            if published.get("written"):
                if accepted and accepted.get("hash") == published.get("hash"):
                    status = "accepted"
                elif warnings:
                    status = "needs_review"
                else:
                    status = "published"
            elif claim:
                status = "claimed"
            elif parts:
                status = "in_progress"
            else:
                status = "missing"
            rows.append(
                {
                    "key": key,
                    "state": status,
                    "answered": len(specs) if published.get("written") else len(parts),
                    "total": len(specs),
                    "claim": claim,
                    "hash": published.get("hash"),
                    "written_at": published.get("written_at"),
                    "warnings": warnings,
                    "codes": sorted({row["code"] for row in findings}),
                    "accepted": status == "accepted",
                    "accepted_by": (accepted or {}).get("agent") if status == "accepted" else None,
                }
            )
        counts = {}
        for row in rows:
            counts[row["state"]] = counts.get(row["state"], 0) + 1
        return {
            "model": self.model.model_id,
            "as_of": day.isoformat(),
            "keys": rows,
            "counts": counts,
        }

    def audit(self, key=None):
        """The advisory findings of one published answer or of every published one."""
        day = self.context.as_of
        written = inputs.read(self.context.inputs, day)
        if key is not None:
            self._declared(key)
            if key not in written:
                raise ValueError(
                    f"input {key!r} is not published on {day.isoformat()}: nothing to audit"
                )
        keys = [key] if key is not None else [
            name for name in self.model.input_keys(day) if name in written
        ]
        return {
            "model": self.model.model_id,
            "as_of": day.isoformat(),
            "keys": [
                {
                    "key": name,
                    "hash": inputs.hash_of(written[name]),
                    "findings": self.model.audit_input(self.context, name, written[name]),
                }
                for name in keys
            ],
        }

    def accept(self, key, agent, reason):
        """Record that an agent reviewed the warnings of a published answer."""
        day = self.context.as_of
        self._declared(key)
        if not agent:
            raise ValueError("accept needs agent: the name of the reviewing agent")
        if not reason:
            raise ValueError("accept needs reason: why the findings are acceptable")
        written = inputs.read(self.context.inputs, day)
        if key not in written:
            raise ValueError(
                f"input {key!r} is not published on {day.isoformat()}: there is nothing to accept"
            )
        digest = inputs.hash_of(written[key])
        with LOCK:
            record = state.accept(self.folder, key, agent, reason, digest)
            journal.record(
                self.folder, "accept", key=key, agent=agent, digest=digest, reason=reason
            )
        return {
            "model": self.model.model_id,
            "as_of": day.isoformat(),
            "key": key,
            "accepted": record,
        }

    def history(self, key):
        """The versions of one published answer, with the current one marked."""
        self._declared(key)
        published = inputs.read(self.context.inputs, self.context.as_of).get(key)
        current = inputs.hash_of(published) if published is not None else None
        rows = versions.history(self.folder, key)
        if current and all(row["hash"] != current for row in rows):
            rows.append({"hash": current})
        reviews = {
            event["hash"]: event
            for event in journal.tail(self.folder, key, 0)
            if event["action"] == "accept"
        }
        accepted = state.read(self.folder)["accepted"].get(key, {})
        for row in rows:
            digest = row["hash"]
            review = reviews.get(digest, accepted if accepted.get("hash") == digest else {})
            row["current"] = digest == current
            row["accepted"] = bool(review)
            row["reason"] = review.get("reason")
            row["written_at"] = row.get("at")
        return {
            "model": self.model.model_id,
            "as_of": self.context.as_of.isoformat(),
            "key": key,
            "current": current,
            "versions": rows,
        }

    def reopen(self, key, steps, agent, reason):
        """Ask named parts again: keep the published answer as a version and make it missing."""
        day = self.context.as_of
        self._declared(key)
        if not steps:
            raise ValueError("reopen needs steps: the parts of the answer to ask again")
        if not reason:
            raise ValueError("reopen needs reason: what makes the answer unacceptable")
        with LOCK:
            written = inputs.read(self.context.inputs, day)
            if key not in written:
                raise ValueError(
                    f"input {key!r} is not published on {day.isoformat()}: nothing to reopen"
                )
            specs = self._steps(key)
            known = {spec["id"] for spec in specs}
            unknown = [step for step in steps if step not in known]
            if unknown:
                raise ValueError(
                    f"reopen: {key!r} has no steps {', '.join(unknown)}; the ids come from "
                    "the question"
                )
            full = written[key]
            digest = inputs.hash_of(full)
            versions.keep(self.folder, key, digest, full)
            parts = self.model.input_parts(self.context, key, full)
            unsplit = [spec["id"] for spec in specs if spec["id"] not in parts]
            if unsplit:
                raise ValueError(
                    "reopen: the model cannot split its published answer into the steps of "
                    f"its interview ({', '.join(unsplit)} missing); it needs input_parts"
                )
            restored = self.model.assemble_input(self.context, key, parts)
            if inputs.hash_of(restored) != digest:
                raise ValueError("reopen: input_parts does not reconstruct the published answer")
            for step in steps:
                parts.pop(step, None)
            store.write_json(self._parts_path(key), parts)
            known_state = state.read(self.folder)
            known_state["claims"].pop(key, None)
            known_state["accepted"].pop(key, None)
            known_state["findings"].pop(key, None)
            known_state["generations"][key] = known_state["generations"].get(key, 0) + 1
            state.write(self.folder, known_state)
            inputs.path(self.context.inputs, day, key).unlink(missing_ok=True)
            journal.record(
                self.folder, "reopen", key=key, agent=agent, reason=reason,
                digest=digest, steps=list(steps),
            )
            return self.next(key, agent)

    def restart(self, key, agent, minutes=45):
        """Resume the same agent after its run stopped: keep its parts, ask the first missing."""
        self._declared(key)
        if not agent:
            raise ValueError("restart needs agent: the name of the agent being restarted")
        if key in inputs.read(self.context.inputs, self.context.as_of):
            raise ValueError(
                f"input {key!r} is already published: nothing to restart, review it instead"
            )
        with LOCK:
            self._touch(key, agent)
            journal.record(self.folder, "restart", key=key, agent=agent)
            result = self.next(key, agent)
            if key not in inputs.read(self.context.inputs, self.context.as_of):
                result["claim"] = state.claim(self.folder, key, agent, minutes)
            return result

    def next(self, key=None, agent=None):
        """The first missing part of the model, or of one key; no selection survives a call."""
        day = self.context.as_of
        with LOCK:
            completed = inputs.read(self.context.inputs, day)
            declared = self.model.input_keys(day)
            if key is not None:
                self._declared(key)
            # Named workers stay on their assigned key; anonymous legacy callers keep
            # traversing the model in its declared order.
            keys = (key,) if key and agent else (
                (key, *(name for name in declared if name != key)) if key else declared
            )
            published = []
            for current in keys:
                if current in completed:
                    continue
                self._touch(current, agent)
                specs = self._steps(current)
                parts = store.read_json(self._parts_path(current)) or {}
                if all(spec["id"] in parts for spec in specs):
                    # A process may have stopped between saving the last part and publication.
                    full = self.model.check_input(
                        self.context,
                        current,
                        self.model.assemble_input(self.context, current, parts),
                    )
                    published.append(self._publish(current, full, agent))
                    continue
                for spec in specs:
                    if spec["id"] in parts:
                        continue
                    journal.record(
                        self.folder, "next", key=current, agent=agent, step=spec["id"]
                    )
                    result = {
                        "model": self.model.model_id,
                        "as_of": day.isoformat(),
                        "key": current,
                        "step": spec["id"],
                        "text": spec["text"],
                        "schema": spec["schema"],
                        "progress": {"answered": len(parts), "total": len(specs)},
                        "answer_call": {
                            "action": "answer",
                            "ticket": tickets.encode(
                                self.context.inputs,
                                self.model.model_id,
                                day,
                                current,
                                spec["id"],
                                state.read(self.folder)["generations"].get(current, 0),
                            ),
                            **({"agent": agent} if agent else {}),
                        },
                        "instruction": "You are the expert. Answer with your own quantitative "
                        "judgement, not placeholders. Do not ask the human for numbers. Pass "
                        "the answer object as payload with answer_call; the result contains "
                        "the next question.",
                    }
                    result.update(
                        self.model.input_step_context(
                            self.context, current, spec["id"], parts
                        )
                    )
                    if published:
                        result["published"] = published
                    return result
            done = {
                "model": self.model.model_id,
                "as_of": day.isoformat(),
                "complete": True,
                "scope": key or "model",
            }
            if published:
                done["published"] = published
            return done

    def answer(self, ticket, payload, agent=None):
        """Check and save one addressed part; the full answer is published after the last."""
        if not ticket or payload is None:
            raise ValueError("answer requires ticket from next and payload as an object")
        address = tickets.decode(self.context.inputs, ticket)
        if (
            address["model"] != self.model.model_id
            or address["date"] != self.context.as_of.isoformat()
        ):
            raise ValueError("ticket addresses another model or date; call next")
        key, step = address["key"], address["step"]
        if key not in self.model.input_keys(self.context.as_of):
            raise ValueError("ticket names no declared input")
        specs = self._steps(key)
        if step not in {spec["id"] for spec in specs}:
            raise ValueError("ticket names no declared step")
        with LOCK:
            generation = state.read(self.folder)["generations"].get(key, 0)
            if address.get("generation", 0) != generation:
                raise ValueError("ticket is no longer current after reopen; call next")
            self._touch(key, agent)
            result, publication = self._save(key, step, payload, specs, agent)
        if publication:
            result["published"] = [publication]
        return result

    def _publish(self, key, full, agent):
        """Write the full answer, collect its findings and record the review it needs."""
        digest = inputs.write(self.context.inputs, self.context.as_of, key, full)
        found = self.model.audit_input(self.context, key, full)
        state.findings(self.folder, key, digest, found)
        journal.record(
            self.folder, "publish", key=key, agent=agent, digest=digest,
            warnings=len(found),
        )
        if found:
            journal.record(
                self.folder, "warn", key=key, agent=agent, warnings=len(found),
                codes=sorted({row["code"] for row in found}),
            )
        else:
            state.accept(self.folder, key, "auto", "no findings", digest)
            journal.record(
                self.folder, "accept", key=key, agent="auto", digest=digest,
                reason="no findings",
            )
        return {
            "key": key,
            "hash": digest,
            "warnings": found,
            "needs_review": bool(found),
        }

    def _publication(self, key):
        """The review block of an answer already published, recomputed from the stored form."""
        written = inputs.read(self.context.inputs, self.context.as_of).get(key)
        if written is None:
            return None
        found = self.model.audit_input(self.context, key, written)
        return {
            "key": key,
            "hash": inputs.hash_of(written),
            "warnings": found,
            "needs_review": bool(found),
        }

    def _save(self, key, step, payload, specs, agent=None):
        path = self._parts_path(key)
        parts = store.read_json(path) or {}
        normal = self.model.check_input_step(self.context, key, step, payload)
        if step in parts:
            if parts[step] != normal:
                raise ValueError("step already answered; do not overwrite earlier judgements")
            return self.next(key, agent), self._publication(key)
        if key in inputs.read(self.context.inputs, self.context.as_of):
            raise ValueError("input already complete; use write to replace its full answer")
        expected = next(spec["id"] for spec in specs if spec["id"] not in parts)
        if step != expected:
            raise ValueError("ticket is out of order; call next for the current question")
        updated = {**parts, step: normal}
        if len(updated) != len(specs):
            store.write_json(path, updated)
            journal.record(self.folder, "answer", key=key, agent=agent, step=step)
            return self.next(key, agent), None
        full = self.model.check_input(
            self.context,
            key,
            self.model.assemble_input(self.context, key, updated),
        )
        # The full check runs before the last part is kept: a refused answer is asked again.
        store.write_json(path, updated)
        journal.record(self.folder, "answer", key=key, agent=agent, step=step)
        # Keep parts for idempotent retries; an interrupted publication is retried in ``next``.
        publication = self._publish(key, full, agent)
        return self.next(key, agent), publication


def address(root, ticket) -> dict:
    """The model, date, key and step a ticket names, without the saved answers."""
    return tickets.decode(root, ticket)
