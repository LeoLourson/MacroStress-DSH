"""Catalogs of the Moscow Exchange: every share of a board, every bond of the given indices.

``MoexShares`` lists the common and preferred shares of a board and places each under the
sector whose MOEX index holds it (else under the root); a sector is an object with the index's
history. The boards a share traded on before T+2 and its splits come from ISS too, so a new
listing or a split needs no edit. ``MoexBonds`` lists the bonds of corporate bond indices, one
index per rating group, and places each issue under its issuer and the issuer under the group
of its first issue.
"""

from datetime import date
from typing import Literal

import httpx

from mst.contracts import Contract, Identifier
from mst.data.moex import MoexBondSpread, MoexIndex, MoexShare, Split

from ..objects import Name, Object
from .base import Catalog, Discovery
from .base import name as object_name

ISS = "https://iss.moex.com/iss"
SHARE_TYPES = {"1", "2"}
"""SECTYPE of common and preferred shares."""


def _table(http: httpx.Client, path: str, block: str, **params) -> list[dict]:
    """Every row of an ISS block, page by page when the block has a cursor."""
    rows, start = [], 0
    while True:
        response = http.get(f"{ISS}/{path}", params={"iss.meta": "off", "start": start, **params})
        response.raise_for_status()
        body = response.json()
        columns = body[block]["columns"]
        page = [dict(zip(columns, row, strict=True)) for row in body[block]["data"]]
        rows += page
        cursor = body.get(f"{block}.cursor")
        if cursor is None:
            return rows
        index, total, size = cursor["data"][0][:3]
        start = index + size
        if not page or start >= total:
            return rows


def _composition(http: httpx.Client, index: str) -> list[str]:
    path = f"statistics/engines/stock/markets/index/analytics/{index}.json"
    return [row["secids"] for row in _table(http, path, "analytics")]


def _security(http: httpx.Client, secid: str) -> tuple[list[dict], dict]:
    """The boards of a security and its description fields."""
    response = http.get(f"{ISS}/securities/{secid}.json", params={"iss.meta": "off"})
    response.raise_for_status()
    body = response.json()
    boards = [
        dict(zip(body["boards"]["columns"], row, strict=True)) for row in body["boards"]["data"]
    ]
    return boards, {row[0]: row[2] for row in body["description"]["data"]}


def _boards(rows: list[dict], market: str, allowed: list[str]) -> list[str]:
    """The allowed boards a security has history on, oldest first."""
    found = [
        row
        for row in rows
        if row["market"] == market and row["boardid"] in allowed and row["history_from"]
    ]
    return [row["boardid"] for row in sorted(found, key=lambda row: row["history_from"])]


def _emitter(http: httpx.Client, emitter: str, cache: dict) -> dict:
    """``{"inn": ..., "title": ...}`` of an ISS emitter, read once."""
    if emitter not in cache:
        rows = _table(http, f"emitters/{emitter}.json", "emitter")
        row = rows[0] if rows else {}
        cache[emitter] = {
            "inn": row.get("INN"),
            "title": row.get("SHORT_TITLE") or row.get("TITLE"),
        }
    return cache[emitter]


def _codes(emitter: str | None, details: dict, **codes) -> dict:
    out = {kind: [value] for kind, value in codes.items() if value}
    if emitter:
        out["emitter"] = [str(emitter)]
    if details.get("inn"):
        out["inn"] = [details["inn"]]
    return out


class Sector(Contract):
    id: Name
    label: Identifier
    index: str
    """The MOEX sector index whose composition and history stand for the sector."""


class MoexShares(Catalog):
    loader: Literal["moex_shares"]
    board: str
    """The board whose shares make the catalog, TQBR."""
    boards: list[str]
    """Boards whose days count, the main trading modes before and after T+2."""
    root: Name
    root_label: Identifier
    sectors: list[Sector]

    def discover(self, http: httpx.Client) -> Discovery:
        found = Discovery()
        found.objects.append(Object.model_validate({"id": self.root, "class": "market",
                                                    "label": self.root_label}))  # fmt: skip
        sector_of = {}
        for sector in self.sectors:
            found.objects.append(
                Object.model_validate(
                    {
                        "id": sector.id,
                        "class": "sector",
                        "label": sector.label,
                        "parent": self.root,
                        "codes": {"secid": [sector.index]},
                    }  # fmt: skip
                )
            )
            found.drivers[sector.id] = MoexIndex(
                loader="moex_index", secid=sector.index, lag_days=self.lag_days,
                revised=self.revised,
            )  # fmt: skip
            for secid in _composition(http, sector.index):
                sector_of.setdefault(secid, sector.id)
        splits: dict[str, list[Split]] = {}
        for row in _table(http, "statistics/engines/stock/splits.json", "splits"):
            splits.setdefault(row["secid"], []).append(
                Split(day=date.fromisoformat(row["tradedate"]), ratio=row["after"] / row["before"])
            )
        path = f"engines/stock/markets/shares/boards/{self.board}/securities.json"
        emitters: dict = {}
        for row in _table(http, path, "securities", **{"iss.only": "securities"}):
            secid = row["SECID"]
            if str(row["SECTYPE"]) not in SHARE_TYPES:
                continue
            rows, fields = _security(http, secid)
            boards = _boards(rows, "shares", self.boards)
            if not boards:
                found.skipped[secid] = "no history on the main boards"
                continue
            emitter = fields.get("EMITTER_ID")
            details = _emitter(http, emitter, emitters) if emitter else {}
            object_id = object_name(secid)
            found.objects.append(
                Object.model_validate(
                    {
                        "id": object_id,
                        "class": "company",
                        "label": row["SHORTNAME"],
                        "parent": sector_of.get(secid, self.root),
                        "codes": _codes(emitter, details, secid=secid, isin=row["ISIN"]),
                    }  # fmt: skip
                )
            )
            found.drivers[object_id] = MoexShare(
                loader="moex_share", secid=secid, boards=boards, splits=splits.get(secid, []),
                lag_days=self.lag_days, revised=self.revised,
            )  # fmt: skip
        return found


class Group(Contract):
    id: Name
    label: Identifier
    index: str
    """The bond index of the group, e.g. RUCBCPAANS for AA of the national scale."""


class MoexBonds(Catalog):
    loader: Literal["moex_bonds"]
    root: Name
    root_label: Identifier
    groups: list[Group]

    def discover(self, http: httpx.Client) -> Discovery:
        found = Discovery()
        found.objects.append(Object.model_validate({"id": self.root, "class": "bonds",
                                                    "label": self.root_label}))  # fmt: skip
        issuers: dict[str, Object] = {}
        issues: list[Object] = []
        emitters: dict = {}
        for group in self.groups:
            found.objects.append(
                Object.model_validate(
                    {
                        "id": group.id,
                        "class": "rating_group",
                        "label": group.label,
                        "parent": self.root,
                    }
                )  # fmt: skip
            )
            for isin in _composition(http, group.index):
                object_id = object_name(isin)
                if any(issue.id == object_id for issue in issues):
                    continue
                _, fields = _security(http, isin)
                emitter = fields.get("EMITTER_ID")
                if not emitter:
                    found.skipped[isin] = "no issuer in ISS"
                    continue
                issuer_id = f"issuer_{emitter}"
                name = fields.get("NAME") or fields.get("SHORTNAME") or isin
                if issuer_id not in issuers:
                    details = _emitter(http, emitter, emitters)
                    issuers[issuer_id] = Object.model_validate(
                        {
                            "id": issuer_id,
                            "class": "issuer",
                            "label": details.get("title") or name.rsplit(" ", 1)[0],
                            "parent": group.id,
                            "codes": _codes(emitter, details),
                        }  # fmt: skip
                    )
                issues.append(
                    Object.model_validate(
                        {
                            "id": object_id,
                            "class": "issue",
                            "label": name,
                            "parent": issuer_id,
                            "codes": {"isin": [isin]},
                        }  # fmt: skip
                    )
                )
                found.drivers[object_id] = MoexBondSpread(
                    loader="moex_bond_spread", secid=isin, lag_days=self.lag_days,
                    revised=self.revised,
                )  # fmt: skip
        found.objects += list(issuers.values()) + issues
        return found
