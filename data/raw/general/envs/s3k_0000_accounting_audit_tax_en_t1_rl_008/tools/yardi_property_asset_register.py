"""MCP tools for the Yardi Voyager property and fixed asset register."""
import sqlite3
from pathlib import Path


def _con():
    connection = sqlite3.connect(
        Path(__file__).resolve().parent.parent / "system" / "yardi_property_asset_register" / "state.db"
    )
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def _validate_date(value, name):
    if value is None:
        return
    try:
        from datetime import date
        date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{name} must use YYYY-MM-DD format and be a valid calendar date") from exc


def _validate_limit(limit):
    if not 1 <= limit <= 100:
        raise ValueError("limit must be between 1 and 100")


def _decode_cursor(cursor, name):
    if cursor is None:
        return None
    parts = cursor.split("|", 1)
    if len(parts) != 2 or not all(parts):
        raise ValueError(f"{name} cursor is invalid")
    return parts


def _asset_dict(row):
    item = dict(row)
    item["adjusted_basis_usd"] = item["original_cost_usd"] - item["accumulated_depreciation_usd"]
    item["approved_flag"] = bool(item["approved_flag"])
    return item


def _ledger_dict(row):
    item = dict(row)
    item["current_flag"] = bool(item["current_flag"])
    item["approved_flag"] = bool(item["approved_flag"])
    return item


def search_property_assets(
    property_id: str,
    case_id: str = None,
    asset_class: str = None,
    placed_in_service_from: str = None,
    placed_in_service_to: str = None,
    depreciation_through_date: str = None,
    tax_basis_status: str = None,
    status: str = None,
    approved_only: bool = True,
    query: str = None,
    limit: int = 50,
    cursor: str = None,
):
    """Search asset records for one property with optional filters and keyset pagination.

    Args:
        property_id: Required stable Yardi property identifier; display names are not used for joins.
        case_id: Optional disposition or review case identifier.
        asset_class: Optional class: LAND, BUILDING, TENANT_IMROVEMENT, SITE_IMPROVEMENT, EQUIPMENT, or DISPOSITION_COST.
        placed_in_service_from: Optional inclusive lower placed-in-service date in YYYY-MM-DD format.
        placed_in_service_to: Optional inclusive upper placed-in-service date in YYYY-MM-DD format.
        depreciation_through_date: Optional exact stored depreciation-through date in YYYY-MM-DD format.
        tax_basis_status: Optional status: INCLUDED, EXCLUDED, or UNREVIEWED.
        status: Optional lifecycle status: ACTIVE, DISPOSED, RETIRED, or VOID.
        approved_only: When true, return only approved asset records.
        query: Optional case-insensitive text match against asset ID or asset name.
        limit: Maximum records to return, from 1 through 100.
        cursor: Optional keyset cursor returned by a prior call.

    Returns:
        A dictionary with asset items, a next_cursor value, and a has_more flag.
    """
    if not property_id or not property_id.strip():
        raise ValueError("property_id is required")
    _validate_limit(limit)
    for value, name in ((placed_in_service_from, "placed_in_service_from"), (placed_in_service_to, "placed_in_service_to"), (depreciation_through_date, "depreciation_through_date")):
        _validate_date(value, name)
    if placed_in_service_from and placed_in_service_to and placed_in_service_from > placed_in_service_to:
        raise ValueError("placed_in_service_from cannot be after placed_in_service_to")
    allowed = {
        "asset_class": {"LAND", "BUILDING", "TENANT_IMROVEMENT", "SITE_IMPROVEMENT", "EQUIPMENT", "DISPOSITION_COST"},
        "tax_basis_status": {"INCLUDED", "EXCLUDED", "UNREVIEWED"},
        "status": {"ACTIVE", "DISPOSED", "RETIRED", "VOID"},
    }
    for name, value in (("asset_class", asset_class), ("tax_basis_status", tax_basis_status), ("status", status)):
        if value is not None and value not in allowed[name]:
            raise ValueError(f"{name} must be one of {', '.join(sorted(allowed[name]))}")
    cursor_parts = _decode_cursor(cursor, "asset")
    clauses, params = ["property_id = ?"], [property_id]
    for column, value, operator in (
        ("case_id", case_id, "="), ("asset_class", asset_class, "="),
        ("placed_in_service_date", placed_in_service_from, ">="), ("placed_in_service_date", placed_in_service_to, "<="),
        ("depreciation_through_date", depreciation_through_date, "="), ("tax_basis_status", tax_basis_status, "="),
        ("status", status, "="),
    ):
        if value is not None:
            clauses.append(f"{column} {operator} ?")
            params.append(value)
    if approved_only:
        clauses.append("approved_flag = 1")
    if query and query.strip():
        clauses.append("(LOWER(asset_id) LIKE ? OR LOWER(asset_name) LIKE ?)")
        term = f"%{query.strip().lower()}%"
        params.extend((term, term))
    if cursor_parts:
        clauses.append("(placed_in_service_date > ? OR (placed_in_service_date = ? AND asset_id > ?))")
        params.extend((cursor_parts[0], cursor_parts[0], cursor_parts[1]))
    sql = f"""SELECT asset_id, property_id, case_id, property_display_name, asset_name, asset_class,
        placed_in_service_date, original_cost_usd, accumulated_depreciation_usd, tax_character_code,
        tax_basis_status, status, approved_flag
        FROM property_assets WHERE {' AND '.join(clauses)}
        ORDER BY placed_in_service_date, asset_id LIMIT ?"""
    params.append(limit + 1)
    with _con() as connection:
        rows = connection.execute(sql, params).fetchall()
    has_more = len(rows) > limit
    rows = rows[:limit]
    items = [_asset_dict(row) for row in rows]
    next_cursor = f"{rows[-1]['placed_in_service_date']}|{rows[-1]['asset_id']}" if has_more else None
    return {"items": items, "next_cursor": next_cursor, "has_more": has_more}


def get_asset_depreciation_detail(
    asset_ids: list,
    as_of_date: str = "2025-09-30",
    approved_only: bool = True,
):
    """Return stored depreciation and tax-character detail for explicit asset IDs.

    Args:
        asset_ids: One through 50 unique asset identifiers to retrieve.
        as_of_date: Reporting date in YYYY-MM-DD format; later stored depreciation is not projected backward.
        approved_only: When true, unapproved assets are reported as not found.

    Returns:
        A dictionary containing depreciation items and not_found_asset_ids in request order.
    """
    if not asset_ids or len(asset_ids) > 50:
        raise ValueError("asset_ids must contain between 1 and 50 identifiers")
    if len(asset_ids) != len(set(asset_ids)):
        raise ValueError("asset_ids cannot contain duplicates")
    _validate_date(as_of_date, "as_of_date")
    placeholders = ",".join("?" for _ in asset_ids)
    clause = " AND approved_flag = 1" if approved_only else ""
    sql = f"""SELECT asset_id, asset_name, asset_class, placed_in_service_date, depreciation_method,
        original_cost_usd, accumulated_depreciation_usd, final_period_depreciation_usd,
        tax_character_code, depreciation_through_date FROM property_assets
        WHERE asset_id IN ({placeholders}){clause}"""
    with _con() as connection:
        found = {row["asset_id"]: row for row in connection.execute(sql, asset_ids)}
    items = []
    not_found = []
    for asset_id in asset_ids:
        row = found.get(asset_id)
        if row is None or row["placed_in_service_date"] > as_of_date:
            not_found.append(asset_id)
            continue
        accumulated = row["accumulated_depreciation_usd"]
        final_period = row["final_period_depreciation_usd"]
        depreciation_through = row["depreciation_through_date"]
        if depreciation_through and depreciation_through > as_of_date:
            accumulated = max(0, accumulated - final_period)
            final_period = 0
            depreciation_through = as_of_date
        items.append({
            "asset_id": row["asset_id"], "asset_name": row["asset_name"], "asset_class": row["asset_class"],
            "placed_in_service_date": row["placed_in_service_date"], "as_of_date": as_of_date,
            "depreciation_method": row["depreciation_method"], "original_cost_usd": row["original_cost_usd"],
            "accumulated_depreciation_usd": accumulated, "final_period_depreciation_usd": final_period,
            "adjusted_basis_usd": row["original_cost_usd"] - accumulated,
            "tax_character_code": row["tax_character_code"], "depreciation_through_date": depreciation_through,
        })
    return {"items": items, "not_found_asset_ids": not_found}


def get_ownership_ledger(
    property_id: str,
    case_id: str = None,
    entity_id: str = None,
    partner_id: str = None,
    ledger_date_from: str = None,
    ledger_date_to: str = None,
    entry_types: list = None,
    current_only: bool = False,
    approved_only: bool = True,
    limit: int = 50,
    cursor: str = None,
):
    """Retrieve ownership events and snapshots for one property with keyset pagination.

    Args:
        property_id: Required stable Yardi property identifier.
        case_id: Optional disposition or review case identifier.
        entity_id: Optional owning entity identifier.
        partner_id: Optional partner or counterparty entity identifier.
        ledger_date_from: Optional inclusive lower ledger date in YYYY-MM-DD format.
        ledger_date_to: Optional inclusive upper ledger date in YYYY-MM-DD format.
        entry_types: Optional list of allowed ownership ledger entry types.
        current_only: When true, return only current snapshot records.
        approved_only: When true, return only approved ledger records.
        limit: Maximum records to return, from 1 through 100.
        cursor: Optional keyset cursor returned by a prior call.

    Returns:
        A dictionary with ledger items, a next_cursor value, and a has_more flag.
    """
    if not property_id or not property_id.strip():
        raise ValueError("property_id is required")
    _validate_limit(limit)
    _validate_date(ledger_date_from, "ledger_date_from")
    _validate_date(ledger_date_to, "ledger_date_to")
    if ledger_date_from and ledger_date_to and ledger_date_from > ledger_date_to:
        raise ValueError("ledger_date_from cannot be after ledger_date_to")
    valid_types = {"OPENING_SNAPSHOT", "CAPITAL_CONTRIBUTION", "DISTRIBUTION", "BASIS_ADJUSTMENT", "DEBT_ALLOCATION", "LIABILITY_RELIEF_ESTIMATE", "PARTNER_SNAPSHOT", "GP_INTEREST_SNAPSHOT"}
    if entry_types is not None:
        if not entry_types:
            raise ValueError("entry_types cannot be empty when provided")
        invalid = sorted(set(entry_types) - valid_types)
        if invalid:
            raise ValueError(f"unsupported entry_types: {', '.join(invalid)}")
    cursor_parts = _decode_cursor(cursor, "ledger")
    clauses, params = ["property_id = ?"], [property_id]
    for column, value, operator in (("case_id", case_id, "="), ("entity_id", entity_id, "="), ("partner_id", partner_id, "="), ("ledger_date", ledger_date_from, ">="), ("ledger_date", ledger_date_to, "<=")):
        if value is not None:
            clauses.append(f"{column} {operator} ?")
            params.append(value)
    if entry_types is not None:
        clauses.append(f"entry_type IN ({','.join('?' for _ in entry_types)})")
        params.extend(entry_types)
    if current_only:
        clauses.append("current_flag = 1")
    if approved_only:
        clauses.append("approved_flag = 1")
    if cursor_parts:
        clauses.append("(ledger_date > ? OR (ledger_date = ? AND ledger_entry_id > ?))")
        params.extend((cursor_parts[0], cursor_parts[0], cursor_parts[1]))
    sql = f"""SELECT ledger_entry_id, property_id, case_id, entity_id, partner_id, ledger_date,
        outside_basis_usd, capital_account_usd, liability_share_usd, entry_amount_usd, entry_type,
        source_asset_id, current_flag, approved_flag FROM ownership_ledger
        WHERE {' AND '.join(clauses)} ORDER BY ledger_date, ledger_entry_id LIMIT ?"""
    params.append(limit + 1)
    with _con() as connection:
        rows = connection.execute(sql, params).fetchall()
    has_more = len(rows) > limit
    rows = rows[:limit]
    next_cursor = f"{rows[-1]['ledger_date']}|{rows[-1]['ledger_entry_id']}" if has_more else None
    return {"items": [_ledger_dict(row) for row in rows], "next_cursor": next_cursor, "has_more": has_more}


def reconcile_property_tax_basis(
    property_id: str,
    case_id: str,
    as_of_date: str,
    approved_only: bool = True,
    include_excluded_assets: bool = False,
):
    """Reconcile property inside tax basis to current ownership basis and liabilities.

    Args:
        property_id: Required stable Yardi property identifier.
        case_id: Required disposition or review case identifier.
        as_of_date: Reconciliation date in YYYY-MM-DD format.
        approved_only: When true, include only approved assets and ownership snapshots.
        include_excluded_assets: When true, include EXCLUDED assets in the basis totals rather than listing them as reconciling items.

    Returns:
        A tax-basis reconciliation dictionary with totals, tax-character detail, exclusions, and status.
    """
    if not property_id or not property_id.strip() or not case_id or not case_id.strip():
        raise ValueError("property_id and case_id are required")
    _validate_date(as_of_date, "as_of_date")
    asset_clauses = ["property_id = ?", "case_id = ?", "placed_in_service_date <= ?"]
    asset_params = [property_id, case_id, as_of_date]
    if approved_only:
        asset_clauses.append("approved_flag = 1")
    with _con() as connection:
        assets = connection.execute(
            f"SELECT asset_id, tax_basis_status, tax_character_code, original_cost_usd, accumulated_depreciation_usd, final_period_depreciation_usd FROM property_assets WHERE {' AND '.join(asset_clauses)}",
            asset_params,
        ).fetchall()
        ledger_clauses = ["property_id = ?", "case_id = ?", "current_flag = 1", "ledger_date <= ?"]
        ledger_params = [property_id, case_id, as_of_date]
        if approved_only:
            ledger_clauses.append("approved_flag = 1")
        snapshots = connection.execute(
            f"SELECT outside_basis_usd, liability_share_usd FROM ownership_ledger WHERE {' AND '.join(ledger_clauses)}",
            ledger_params,
        ).fetchall()
    included, excluded = [], []
    for row in assets:
        if row["tax_basis_status"] == "INCLUDED" or (include_excluded_assets and row["tax_basis_status"] == "EXCLUDED"):
            included.append(row)
        else:
            reason = "Excluded from tax basis" if row["tax_basis_status"] == "EXCLUDED" else "Tax basis review incomplete"
            excluded.append({"asset_id": row["asset_id"], "reason": reason, "amount_usd": row["original_cost_usd"] - row["accumulated_depreciation_usd"]})
    original = sum(row["original_cost_usd"] for row in included)
    accumulated = sum(row["accumulated_depreciation_usd"] for row in included)
    final_period = sum(row["final_period_depreciation_usd"] for row in included)
    inside = original - accumulated
    by_code = {}
    for row in included:
        bucket = by_code.setdefault(row["tax_character_code"], {"tax_character_code": row["tax_character_code"], "asset_count": 0, "original_cost_usd": 0, "accumulated_depreciation_usd": 0, "adjusted_basis_usd": 0})
        bucket["asset_count"] += 1
        bucket["original_cost_usd"] += row["original_cost_usd"]
        bucket["accumulated_depreciation_usd"] += row["accumulated_depreciation_usd"]
        bucket["adjusted_basis_usd"] += row["original_cost_usd"] - row["accumulated_depreciation_usd"]
    outside = sum(row["outside_basis_usd"] for row in snapshots)
    liabilities = sum(row["liability_share_usd"] for row in snapshots)
    if not included or not snapshots or any(item["reason"] == "Tax basis review incomplete" for item in excluded):
        status = "INCOMPLETE"
    elif inside == outside:
        status = "RECONCILED"
    else:
        status = "CONTROLLED_DIFF"
    return {
        "property_id": property_id, "case_id": case_id, "as_of_date": as_of_date,
        "asset_count": len(included), "original_cost_usd": original,
        "accumulated_depreciation_usd": accumulated, "final_period_depreciation_usd": final_period,
        "adjusted_inside_basis_usd": inside, "outside_basis_usd": outside,
        "liability_share_usd": liabilities, "by_tax_character": [by_code[key] for key in sorted(by_code)],
        "excluded_assets": excluded, "reconciliation_status": status,
    }


def update_asset_tax_basis_status(
    asset_id: str,
    tax_basis_status: str,
    approved_flag: bool,
    approved_by_entity_id: str,
    expected_row_version: int,
):
    """Update one asset's tax-basis review state using optimistic concurrency.

    Args:
        asset_id: Asset identifier to update.
        tax_basis_status: New status: INCLUDED, EXCLUDED, or UNREVIEWED.
        approved_flag: Whether the updated review state is approved.
        approved_by_entity_id: Reviewer entity identifier; required when approved_flag is true.
        expected_row_version: Current row version expected by the caller.

    Returns:
        A dictionary containing ok, the updated asset summary, or a structured error.
    """
    if tax_basis_status not in {"INCLUDED", "EXCLUDED", "UNREVIEWED"}:
        raise ValueError("tax_basis_status must be INCLUDED, EXCLUDED, or UNREVIEWED")
    if expected_row_version < 1:
        raise ValueError("expected_row_version must be at least 1")
    reviewer = approved_by_entity_id.strip() if approved_by_entity_id else ""
    if approved_flag and not reviewer:
        return {"ok": False, "asset": None, "error": {"code": "VALIDATION_ERROR", "message": "approved_by_entity_id is required when approved_flag is true", "current_row_version": None}}
    from datetime import datetime, timezone
    approved_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ") if approved_flag else None
    reviewer_value = reviewer or None
    with _con() as connection:
        current = connection.execute("SELECT row_version FROM property_assets WHERE asset_id = ?", (asset_id,)).fetchone()
        if current is None:
            return {"ok": False, "asset": None, "error": {"code": "NOT_FOUND", "message": f"Asset {asset_id} was not found", "current_row_version": None}}
        if current["row_version"] != expected_row_version:
            return {"ok": False, "asset": None, "error": {"code": "VERSION_CONFLICT", "message": "The asset was updated by another user", "current_row_version": current["row_version"]}}
        connection.execute(
            "UPDATE property_assets SET tax_basis_status = ?, approved_flag = ?, approved_by_entity_id = ?, approved_at = ?, row_version = row_version + 1 WHERE asset_id = ? AND row_version = ?",
            (tax_basis_status, int(approved_flag), reviewer_value, approved_at, asset_id, expected_row_version),
        )
        row = connection.execute("SELECT asset_id, tax_basis_status, approved_flag, approved_by_entity_id, approved_at, row_version FROM property_assets WHERE asset_id = ?", (asset_id,)).fetchone()
    asset = dict(row)
    asset["approved_flag"] = bool(asset["approved_flag"])
    return {"ok": True, "asset": asset, "error": None}
