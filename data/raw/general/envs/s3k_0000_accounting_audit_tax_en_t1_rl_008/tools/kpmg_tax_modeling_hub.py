"""Query and calculation tools for the KPMG Tax Modeling Hub."""
import sqlite3
from pathlib import Path

_DB_PATH = Path(__file__).resolve().parent.parent / "system" / "kpmg_tax_modeling_hub" / "state.db"
_COMPONENT_TYPES = {
    "AMOUNT_REALIZED", "LIABILITY_RELIEF", "ADJUSTED_TAX_BASIS", "SELLING_COST",
    "ORIGINAL_COST", "ACCUMULATED_DEPRECIATION", "INSIDE_BASIS_ADJUSTMENT",
    "OUTSIDE_EQUITY_BASIS",
}
_CHARACTER_CODES = {
    "IRC_1245", "IRC_291", "IRC_1250_UNRECAPTURED", "IRC_1231", "IRC_751",
    "IRC_741", "ELIGIBLE_GAIN",
}
_STRUCTURE_CODES = {"DIRECT", "EQUITY", "INSTALLMENT"}
_STATUSES = {"MODELED", "APPROVED", "ARCHIVED"}


def _con():
    con = sqlite3.connect(_DB_PATH)
    con.row_factory = sqlite3.Row
    return con


def _validate_page(page_size, page_token):
    if not 1 <= page_size <= 200:
        raise ValueError("page_size must be between 1 and 200")
    if page_token is None:
        return 0
    try:
        offset = int(page_token)
    except ValueError as exc:
        raise ValueError("page_token must be a nonnegative integer offset") from exc
    if offset < 0 or str(offset) != page_token:
        raise ValueError("page_token must be a nonnegative integer offset")
    return offset


def _validate_date(value, name):
    if value is None:
        return
    try:
        parsed = __import__("datetime").date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{name} must use YYYY-MM-DD format") from exc
    if parsed.isoformat() != value:
        raise ValueError(f"{name} must use YYYY-MM-DD format")


def _case_row(con, case_id):
    row = con.execute("SELECT * FROM disposal_cases WHERE case_id = ?", (case_id,)).fetchone()
    if row is None:
        raise ValueError(f"Unknown case_id: {case_id}")
    return row


def _approved_basis(con, case_id):
    return con.execute(
        """SELECT * FROM tax_basis_components
           WHERE case_id = ? AND approved_flag = 1 AND superseded_flag = 0""",
        (case_id,),
    ).fetchall()


def _approved_schedule(con, case_id):
    return con.execute(
        """SELECT * FROM recognition_schedules
           WHERE case_id = ? AND approved_flag = 1 AND superseded_flag = 0""",
        (case_id,),
    ).fetchall()


def _tax_calculation(con, case_id, as_of_date, include_character_detail):
    case = _case_row(con, case_id)
    basis_rows = _approved_basis(con, case_id)
    schedule_rows = _approved_schedule(con, case_id)
    if not basis_rows:
        raise ValueError(f"No approved, current tax-basis detail exists for case_id: {case_id}")
    amounts = {component: 0 for component in _COMPONENT_TYPES}
    for row in basis_rows:
        amounts[row["component_type"]] += row["amount_usd_cents"]
    basis_deduction = (
        amounts["OUTSIDE_EQUITY_BASIS"]
        if case["structure_code"] == "EQUITY" and amounts["OUTSIDE_EQUITY_BASIS"]
        else amounts["ADJUSTED_TAX_BASIS"]
    )
    total_gain = amounts["AMOUNT_REALIZED"] + amounts["LIABILITY_RELIEF"] - basis_deduction - amounts["SELLING_COST"]
    selected = [row for row in schedule_rows if as_of_date is None or row["payment_date"] <= as_of_date]
    recognized = sum(row["recognized_gain_usd_cents"] for row in selected)
    deferred = sum(row["deferred_gain_usd_cents"] for row in schedule_rows if row["deferred_gain_flag"] == 1)
    by_character = []
    if include_character_detail:
        characters = sorted({row["character_code"] for row in schedule_rows})
        for character in characters:
            character_rows = [row for row in schedule_rows if row["character_code"] == character]
            by_character.append({
                "character_code": character,
                "recognized_gain_usd_cents": sum(
                    row["recognized_gain_usd_cents"]
                    for row in character_rows
                    if as_of_date is None or row["payment_date"] <= as_of_date
                ),
                "deferred_gain_usd_cents": sum(row["deferred_gain_usd_cents"] for row in character_rows),
            })
    scheduled_total = sum(row["recognized_gain_usd_cents"] + row["deferred_gain_usd_cents"] for row in schedule_rows)
    status = "RECONCILED" if scheduled_total == total_gain else "DETAIL_VARIANCE"
    return {
        "case_id": case["case_id"],
        "amount_realized_usd_cents": amounts["AMOUNT_REALIZED"],
        "liability_relief_usd_cents": amounts["LIABILITY_RELIEF"],
        "adjusted_tax_basis_usd_cents": amounts["ADJUSTED_TAX_BASIS"],
        "selling_cost_usd_cents": amounts["SELLING_COST"],
        "total_taxable_gain_usd_cents": total_gain,
        "recognized_gain_usd_cents": recognized,
        "deferred_gain_usd_cents": deferred,
        "by_character": by_character,
        "reconciliation_status": status,
    }


def search_disposal_cases(
    taxpayer_name: str = None,
    property_id: str = None,
    closing_date_from: str = None,
    closing_date_to: str = None,
    structure_code: str = None,
    status: str = None,
    page_size: int = 50,
    page_token: str = None,
) -> dict:
    """Search disposal case headers using exact business and date filters.

    Args:
        taxpayer_name: Exact legal taxpayer name to match.
        property_id: Exact property identifier to match.
        closing_date_from: Earliest closing date, inclusive, in YYYY-MM-DD format.
        closing_date_to: Latest closing date, inclusive, in YYYY-MM-DD format.
        structure_code: Optional DIRECT, EQUITY, or INSTALLMENT structure filter.
        status: Optional MODELED, APPROVED, or ARCHIVED workflow status.
        page_size: Maximum records to return, from 1 through 200.
        page_token: Offset token returned by a prior call.

    Returns:
        A dictionary containing items, next_page_token, and total_count.
    """
    if not any([taxpayer_name, property_id, closing_date_from, closing_date_to, structure_code, status]):
        raise ValueError("Provide at least one case search filter")
    _validate_date(closing_date_from, "closing_date_from")
    _validate_date(closing_date_to, "closing_date_to")
    if closing_date_from and closing_date_to and closing_date_from > closing_date_to:
        raise ValueError("closing_date_from cannot be later than closing_date_to")
    if structure_code is not None and structure_code not in _STRUCTURE_CODES:
        raise ValueError("structure_code must be DIRECT, EQUITY, or INSTALLMENT")
    if status is not None and status not in _STATUSES:
        raise ValueError("status must be MODELED, APPROVED, or ARCHIVED")
    offset = _validate_page(page_size, page_token)
    clauses, params = [], []
    for column, value in (("taxpayer_name", taxpayer_name), ("property_id", property_id), ("structure_code", structure_code), ("status", status)):
        if value is not None:
            clauses.append(f"{column} = ?")
            params.append(value)
    if closing_date_from is not None:
        clauses.append("closing_date >= ?")
        params.append(closing_date_from)
    if closing_date_to is not None:
        clauses.append("closing_date <= ?")
        params.append(closing_date_to)
    where = " AND ".join(clauses)
    columns = "case_id, entity_id, property_id, taxpayer_name, property_name, structure_code, closing_date, tax_year, status"
    with _con() as con:
        total = con.execute(f"SELECT COUNT(*) FROM disposal_cases WHERE {where}", params).fetchone()[0]
        rows = con.execute(
            f"SELECT {columns} FROM disposal_cases WHERE {where} ORDER BY closing_date DESC, case_id LIMIT ? OFFSET ?",
            [*params, page_size, offset],
        ).fetchall()
    next_token = str(offset + len(rows)) if offset + len(rows) < total else None
    return {"items": [dict(row) for row in rows], "next_page_token": next_token, "total_count": total}


def get_tax_basis_detail(
    case_id: str,
    component_types: list = None,
    approved_only: bool = True,
    include_superseded: bool = False,
    page_size: int = 100,
    page_token: str = None,
) -> dict:
    """Return paginated tax-basis components for a disposal case.

    Args:
        case_id: Disposal case identifier.
        component_types: Optional list of component types to include.
        approved_only: When true, include only approved component rows.
        include_superseded: When false, exclude superseded imports.
        page_size: Maximum records to return, from 1 through 200.
        page_token: Offset token returned by a prior call.

    Returns:
        A dictionary containing case_id, property_id, items, and next_page_token.
    """
    if not case_id:
        raise ValueError("case_id is required")
    if component_types is not None:
        invalid = sorted(set(component_types) - _COMPONENT_TYPES)
        if invalid:
            raise ValueError(f"Unsupported component_types: {', '.join(invalid)}")
    offset = _validate_page(page_size, page_token)
    clauses, params = ["case_id = ?"], [case_id]
    if component_types:
        clauses.append(f"component_type IN ({','.join('?' for _ in component_types)})")
        params.extend(component_types)
    if approved_only:
        clauses.append("approved_flag = 1")
    if not include_superseded:
        clauses.append("superseded_flag = 0")
    where = " AND ".join(clauses)
    columns = "basis_component_id, component_type, amount_usd_cents, tax_character_code, source_period, source_system, approved_flag, superseded_flag, description"
    with _con() as con:
        case = _case_row(con, case_id)
        total = con.execute(f"SELECT COUNT(*) FROM tax_basis_components WHERE {where}", params).fetchone()[0]
        rows = con.execute(
            f"SELECT {columns} FROM tax_basis_components WHERE {where} ORDER BY source_period, basis_component_id LIMIT ? OFFSET ?",
            [*params, page_size, offset],
        ).fetchall()
    next_token = str(offset + len(rows)) if offset + len(rows) < total else None
    return {"case_id": case_id, "property_id": case["property_id"], "items": [dict(row) for row in rows], "next_page_token": next_token}


def get_recognition_schedule(
    case_id: str,
    payment_date_from: str = None,
    payment_date_to: str = None,
    character_code: str = None,
    approved_only: bool = True,
    include_superseded: bool = False,
    page_size: int = 100,
    page_token: str = None,
) -> dict:
    """Return payment-level tax recognition rows for one disposal case.

    Args:
        case_id: Disposal case identifier.
        payment_date_from: Earliest payment date, inclusive, in YYYY-MM-DD format.
        payment_date_to: Latest payment date, inclusive, in YYYY-MM-DD format.
        character_code: Optional IRC character classification to match.
        approved_only: When true, include only approved schedule rows.
        include_superseded: When false, exclude superseded imports.
        page_size: Maximum records to return, from 1 through 200.
        page_token: Offset token returned by a prior call.

    Returns:
        A dictionary containing case_id, items, and next_page_token.
    """
    if not case_id:
        raise ValueError("case_id is required")
    _validate_date(payment_date_from, "payment_date_from")
    _validate_date(payment_date_to, "payment_date_to")
    if payment_date_from and payment_date_to and payment_date_from > payment_date_to:
        raise ValueError("payment_date_from cannot be later than payment_date_to")
    if character_code is not None and character_code not in _CHARACTER_CODES:
        raise ValueError("character_code is not supported")
    offset = _validate_page(page_size, page_token)
    clauses, params = ["case_id = ?"], [case_id]
    if payment_date_from is not None:
        clauses.append("payment_date >= ?")
        params.append(payment_date_from)
    if payment_date_to is not None:
        clauses.append("payment_date <= ?")
        params.append(payment_date_to)
    if character_code is not None:
        clauses.append("character_code = ?")
        params.append(character_code)
    if approved_only:
        clauses.append("approved_flag = 1")
    if not include_superseded:
        clauses.append("superseded_flag = 0")
    where = " AND ".join(clauses)
    columns = "schedule_line_id, payment_reference, payment_date, principal_usd_cents, gross_profit_pct_bps, recognized_gain_usd_cents, deferred_gain_usd_cents, character_code, recapture_exempt, deferred_gain_flag, approved_flag, superseded_flag"
    with _con() as con:
        _case_row(con, case_id)
        total = con.execute(f"SELECT COUNT(*) FROM recognition_schedules WHERE {where}", params).fetchone()[0]
        rows = con.execute(
            f"SELECT {columns} FROM recognition_schedules WHERE {where} ORDER BY payment_date, schedule_line_id LIMIT ? OFFSET ?",
            [*params, page_size, offset],
        ).fetchall()
    next_token = str(offset + len(rows)) if offset + len(rows) < total else None
    return {"case_id": case_id, "items": [dict(row) for row in rows], "next_page_token": next_token}


def recompute_disposal_tax(
    case_id: str,
    as_of_date: str = None,
    include_character_detail: bool = True,
) -> dict:
    """Recompute taxable, recognized, and deferred gain from approved detail.

    Args:
        case_id: Disposal case identifier to recompute.
        as_of_date: Optional recognition cutoff date in YYYY-MM-DD format.
        include_character_detail: Whether to return IRC character subtotals.

    Returns:
        A dictionary with basis inputs, gain totals, character detail, and reconciliation status.
    """
    if not case_id:
        raise ValueError("case_id is required")
    _validate_date(as_of_date, "as_of_date")
    with _con() as con:
        return _tax_calculation(con, case_id, as_of_date, include_character_detail)


def compare_disposal_structures(
    case_id: str,
    comparison_case_ids: list = None,
    tax_year: int = 2025,
) -> dict:
    """Compare disposal structures sharing a common property perimeter.

    Args:
        case_id: Base disposal case identifier.
        comparison_case_ids: Optional case identifiers to compare with the base case.
        tax_year: Tax year used to classify current-year recognized gain.

    Returns:
        A dictionary containing structure comparisons and common-perimeter reconciliation.
    """
    if not case_id:
        raise ValueError("case_id is required")
    if not 1900 <= tax_year <= 2200:
        raise ValueError("tax_year must be between 1900 and 2200")
    with _con() as con:
        base = _case_row(con, case_id)
        if comparison_case_ids is None:
            rows = con.execute(
                "SELECT case_id FROM disposal_cases WHERE property_id = ? AND case_id <> ? ORDER BY structure_code, case_id",
                (base["property_id"], case_id),
            ).fetchall()
            selected_ids = [case_id, *[row["case_id"] for row in rows]]
        else:
            selected_ids = [case_id, *[value for value in comparison_case_ids if value != case_id]]
        comparisons = []
        property_ids = set()
        for selected_id in selected_ids:
            selected_case = _case_row(con, selected_id)
            property_ids.add(selected_case["property_id"])
            schedule = _approved_schedule(con, selected_id)
            basis = _approved_basis(con, selected_id)
            total_gain = sum(row["recognized_gain_usd_cents"] + row["deferred_gain_usd_cents"] for row in schedule)
            current_year = sum(row["recognized_gain_usd_cents"] for row in schedule if row["payment_date"][:4] == str(tax_year))
            deferred = sum(row["deferred_gain_usd_cents"] for row in schedule if row["deferred_gain_flag"] == 1)
            by_character = [
                {
                    "character_code": character,
                    "amount_usd_cents": sum(
                        row["recognized_gain_usd_cents"] + row["deferred_gain_usd_cents"]
                        for row in schedule if row["character_code"] == character
                    ),
                }
                for character in sorted({row["character_code"] for row in schedule})
            ]
            basis_map = {component: 0 for component in _COMPONENT_TYPES}
            for row in basis:
                basis_map[row["component_type"]] += row["amount_usd_cents"]
            assumptions = []
            if selected_case["executive_gain_usd_cents"] != total_gain:
                assumptions.append("EXECUTIVE_SUMMARY_VARIANCE")
            if selected_case["structure_code"] == "INSTALLMENT":
                assumptions.extend(["RECAPTURE_RECOGNIZED_IN_SALE_YEAR", "ELIGIBLE_GAIN_FOLLOWS_PAYMENT_TIMING"])
            if selected_case["structure_code"] == "EQUITY":
                assumptions.append("SECTION_751_HOT_ASSET_ALLOCATION")
            transaction_inputs = {
                "amount_realized_usd_cents": basis_map["AMOUNT_REALIZED"],
                "liability_relief_usd_cents": basis_map["LIABILITY_RELIEF"],
                "adjusted_tax_basis_usd_cents": basis_map["ADJUSTED_TAX_BASIS"],
                "selling_cost_usd_cents": basis_map["SELLING_COST"],
                "inside_basis_adjustment_usd_cents": basis_map["INSIDE_BASIS_ADJUSTMENT"],
                "outside_equity_basis_usd_cents": basis_map["OUTSIDE_EQUITY_BASIS"],
            }
            comparisons.append({
                "case_id": selected_id,
                "structure_code": selected_case["structure_code"],
                "total_gain_usd_cents": total_gain,
                "current_year_recognized_gain_usd_cents": current_year,
                "deferred_gain_usd_cents": deferred,
                "by_character": by_character,
                "transaction_tax_inputs": transaction_inputs,
                "assumption_flags": assumptions,
            })
    reconciliation = {
        "property_id": base["property_id"] if len(property_ids) == 1 else None,
        "all_cases_share_property": len(property_ids) == 1,
        "structures_present": sorted({item["structure_code"] for item in comparisons}),
        "comparison_count": len(comparisons),
    }
    return {"base_case_id": case_id, "comparisons": comparisons, "common_perimeter_reconciliation": reconciliation}
