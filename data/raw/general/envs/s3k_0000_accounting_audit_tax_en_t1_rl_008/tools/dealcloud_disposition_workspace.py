"""Query tools for the DealCloud Real Estate Disposition Workspace."""

import sqlite3
from pathlib import Path


def _con():
    connection = sqlite3.connect(
        Path(__file__).resolve().parent.parent
        / "system"
        / "dealcloud_disposition_workspace"
        / "state.db"
    )
    connection.row_factory = sqlite3.Row
    return connection


def _rows(cursor):
    return [dict(row) for row in cursor.fetchall()]


def _require_text(value, field_name):
    if value is None or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty value.")


def _validate_iso_date(value, field_name):
    if value is None:
        return
    try:
        __import__("datetime").date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{field_name} must use YYYY-MM-DD format.") from exc


def search_disposition_transactions(
    property_id: str = None,
    structure_code: str = None,
    buyer_entity_id: str = None,
    closing_date_from: str = None,
    closing_date_to: str = None,
    transaction_status: str = None,
    page_size: int = 50,
    page_token: str = None,
):
    """Search disposition transactions using business filters and offset pagination.

    Args:
        property_id: Exact property identifier to match.
        structure_code: Transaction structure: direct_sale, equity_transfer, or installment_sale.
        buyer_entity_id: Exact buyer entity identifier to match.
        closing_date_from: Inclusive closing-date lower bound in YYYY-MM-DD format.
        closing_date_to: Inclusive closing-date upper bound in YYYY-MM-DD format.
        transaction_status: Status to match: active, recommended, superseded, closed, or draft.
        page_size: Maximum records to return, from 1 through 200.
        page_token: Opaque numeric offset returned by a prior call.

    Returns:
        A dictionary containing items, next_page_token, and total_count.
    """
    if not any((property_id, structure_code, buyer_entity_id, closing_date_from, closing_date_to, transaction_status)):
        raise ValueError("Provide at least one transaction filter.")
    if structure_code not in (None, "direct_sale", "equity_transfer", "installment_sale"):
        raise ValueError("structure_code must be direct_sale, equity_transfer, or installment_sale.")
    if transaction_status not in (None, "active", "recommended", "superseded", "closed", "draft"):
        raise ValueError("transaction_status must be active, recommended, superseded, closed, or draft.")
    if not 1 <= page_size <= 200:
        raise ValueError("page_size must be between 1 and 200.")
    _validate_iso_date(closing_date_from, "closing_date_from")
    _validate_iso_date(closing_date_to, "closing_date_to")
    if closing_date_from and closing_date_to and closing_date_from > closing_date_to:
        raise ValueError("closing_date_from cannot be after closing_date_to.")
    if page_token is None:
        offset = 0
    else:
        try:
            offset = int(page_token)
        except ValueError as exc:
            raise ValueError("page_token must be a nonnegative offset returned by this tool.") from exc
        if offset < 0 or str(offset) != page_token.strip():
            raise ValueError("page_token must be a nonnegative offset returned by this tool.")

    clauses = []
    parameters = []
    for column, value in (
        ("property_id", property_id),
        ("structure_code", structure_code),
        ("buyer_entity_id", buyer_entity_id),
        ("transaction_status", transaction_status),
    ):
        if value is not None:
            clauses.append(f"{column} = ?")
            parameters.append(value)
    if closing_date_from is not None:
        clauses.append("closing_date >= ?")
        parameters.append(closing_date_from)
    if closing_date_to is not None:
        clauses.append("closing_date <= ?")
        parameters.append(closing_date_to)
    where_sql = " AND ".join(clauses)
    with _con() as connection:
        total_count = connection.execute(
            f"SELECT COUNT(*) FROM disposition_transactions WHERE {where_sql}", parameters
        ).fetchone()[0]
        cursor = connection.execute(
            f"""SELECT transaction_id, case_id, property_id, buyer_entity_id,
                       structure_code, gross_consideration_usd, closing_date,
                       transaction_status
                FROM disposition_transactions
                WHERE {where_sql}
                ORDER BY closing_date DESC, transaction_id
                LIMIT ? OFFSET ?""",
            [*parameters, page_size, offset],
        )
        items = _rows(cursor)
    next_offset = offset + len(items)
    return {
        "items": items,
        "next_page_token": str(next_offset) if next_offset < total_count else None,
        "total_count": total_count,
    }


def get_consideration_terms(transaction_id: str):
    """Return the complete consideration and seller-financing terms for one transaction.

    Args:
        transaction_id: Exact DealCloud disposition transaction identifier.

    Returns:
        A transaction terms dictionary, or an error dictionary when no transaction exists.
    """
    _require_text(transaction_id, "transaction_id")
    with _con() as connection:
        row = connection.execute(
            """SELECT transaction_id, case_id, structure_code,
                      declared_tax_consideration_usd, cash_consideration_usd,
                      closing_principal_usd, secured_seller_note_principal_usd,
                      seller_note_interest_rate_bps, seller_note_term_months,
                      liability_relief_usd, closing_date
               FROM disposition_transactions WHERE transaction_id = ?""",
            (transaction_id,),
        ).fetchone()
    if row is None:
        return {
            "error": {
                "code": "NOT_FOUND",
                "message": f"Disposition transaction {transaction_id} was not found.",
            }
        }
    return dict(row)


def get_transfer_tax_requirements(
    transaction_id: str,
    include_inactive: bool = False,
    requirement_types: list = None,
):
    """List transfer-tax requirements for a transaction with optional type filtering.

    Args:
        transaction_id: Exact DealCloud disposition transaction identifier.
        include_inactive: Whether to include requirements whose active_flag is zero.
        requirement_types: Optional list of exact requirement_type values to include.

    Returns:
        A dictionary containing transaction_id, filtered items, and total_count.
    """
    _require_text(transaction_id, "transaction_id")
    if requirement_types is not None:
        if not requirement_types or any(not str(value).strip() for value in requirement_types):
            raise ValueError("requirement_types must contain at least one non-empty requirement type.")
    clauses = ["transaction_id = ?", "requirement_type LIKE 'documentary_transfer_tax_%'"]
    parameters = [transaction_id]
    if not include_inactive:
        clauses.append("active_flag = 1")
    if requirement_types is not None:
        placeholders = ",".join("?" for _ in requirement_types)
        clauses.append(f"requirement_type IN ({placeholders})")
        parameters.extend(requirement_types)
    with _con() as connection:
        cursor = connection.execute(
            f"""SELECT requirement_id, requirement_type, status, amount_usd,
                       due_date, active_flag, qualifying_instrument_flag,
                       declared_consideration_usd, tax_component_code,
                       rate_version, evidence_reference
                FROM closing_requirements
                WHERE {' AND '.join(clauses)}
                ORDER BY due_date, requirement_id""",
            parameters,
        )
        items = _rows(cursor)
    return {"transaction_id": transaction_id, "items": items, "total_count": len(items)}


def get_withholding_and_closing_status(
    case_id: str,
    transaction_id: str = None,
    active_only: bool = True,
):
    """Return withholding, affidavit, solvency, collateral, and covenant status for a case.

    Args:
        case_id: Exact disposition case identifier.
        transaction_id: Optional transaction identifier within the case.
        active_only: Whether to exclude inactive closing requirements.

    Returns:
        A dictionary with case and transaction scope, requirements, and latest activities.
    """
    _require_text(case_id, "case_id")
    if transaction_id is not None:
        _require_text(transaction_id, "transaction_id")
    requirement_types = (
        "california_withholding_clearance",
        "federal_nonforeign_affidavit",
        "buyer_solvency_diligence",
        "seller_note_collateral",
        "payment_covenant",
    )
    activity_types = (
        "withholding_clearance",
        "federal_nonforeign_affidavit",
        "buyer_solvency_diligence",
        "collateral_review",
        "payment_covenant_approval",
        "structure_superseded",
    )
    req_clauses = ["case_id = ?", f"requirement_type IN ({','.join('?' for _ in requirement_types)})"]
    req_parameters = [case_id, *requirement_types]
    act_clauses = ["case_id = ?", f"activity_type IN ({','.join('?' for _ in activity_types)})"]
    act_parameters = [case_id, *activity_types]
    if transaction_id is not None:
        req_clauses.append("transaction_id = ?")
        req_parameters.append(transaction_id)
        act_clauses.append("transaction_id = ?")
        act_parameters.append(transaction_id)
    if active_only:
        req_clauses.append("active_flag = 1")
    with _con() as connection:
        requirements = _rows(
            connection.execute(
                f"""SELECT requirement_type, status, due_date, active_flag, evidence_reference
                    FROM closing_requirements
                    WHERE {' AND '.join(req_clauses)}
                    ORDER BY due_date, requirement_id""",
                req_parameters,
            )
        )
        latest_activities = _rows(
            connection.execute(
                f"""SELECT activity_type, activity_status, activity_timestamp, source_reference
                    FROM execution_activity_log
                    WHERE {' AND '.join(act_clauses)}
                    ORDER BY activity_timestamp DESC, activity_id DESC
                    LIMIT 20""",
                act_parameters,
            )
        )
    return {
        "case_id": case_id,
        "transaction_id": transaction_id,
        "requirements": requirements,
        "latest_activities": latest_activities,
    }


def recompute_documentary_transfer_tax(
    case_id: str,
    transaction_ids: list = None,
    rate_version: str = "LA-2025-v1",
    include_qualifying_instruments: bool = False,
):
    """Recompute approved county and city documentary transfer tax components.

    Args:
        case_id: Exact disposition case identifier.
        transaction_ids: Optional list limiting the calculation to specific case transactions.
        rate_version: Exact stored transfer-tax rate version to apply.
        include_qualifying_instruments: Whether qualifying instruments may enter the calculation.

    Returns:
        A reconciliation dictionary with approved base, component taxes, included and excluded IDs.
    """
    _require_text(case_id, "case_id")
    _require_text(rate_version, "rate_version")
    if transaction_ids is not None and (not transaction_ids or any(not str(value).strip() for value in transaction_ids)):
        raise ValueError("transaction_ids must contain at least one non-empty transaction identifier.")

    scope_clauses = ["t.case_id = ?"]
    scope_parameters = [case_id]
    if transaction_ids is not None:
        placeholders = ",".join("?" for _ in transaction_ids)
        scope_clauses.append(f"t.transaction_id IN ({placeholders})")
        scope_parameters.extend(transaction_ids)
    qualifying_filter = "" if include_qualifying_instruments else "AND r.qualifying_instrument_flag = 0"
    with _con() as connection:
        rows = connection.execute(
            f"""SELECT t.transaction_id, r.requirement_type, r.amount_usd,
                       r.declared_consideration_usd, r.qualifying_instrument_flag
                FROM disposition_transactions t
                JOIN closing_requirements r ON r.transaction_id = t.transaction_id
                WHERE {' AND '.join(scope_clauses)}
                  AND r.requirement_type IN (
                      'documentary_transfer_tax_county',
                      'documentary_transfer_tax_city'
                  )
                  AND r.status = 'approved'
                  AND r.active_flag = 1
                  AND r.rate_version = ?
                  {qualifying_filter}
                ORDER BY t.transaction_id, r.requirement_id""",
            [*scope_parameters, rate_version],
        ).fetchall()
        excluded_rows = connection.execute(
            f"""SELECT DISTINCT t.transaction_id
                FROM disposition_transactions t
                JOIN closing_requirements r ON r.transaction_id = t.transaction_id
                WHERE {' AND '.join(scope_clauses)}
                  AND r.requirement_type = 'documentary_transfer_tax_equity_instrument'
                  AND r.qualifying_instrument_flag = 1
                  AND r.active_flag = 1
                ORDER BY t.transaction_id""",
            scope_parameters,
        ).fetchall()

    included_ids = sorted({row["transaction_id"] for row in rows})
    approved_base_usd = sum(
        max(row["declared_consideration_usd"] or 0 for row in rows if row["transaction_id"] == transaction_id)
        for transaction_id in included_ids
    )
    county_tax_usd = sum(
        row["amount_usd"] or 0
        for row in rows
        if row["requirement_type"] == "documentary_transfer_tax_county"
    )
    city_tax_usd = sum(
        row["amount_usd"] or 0
        for row in rows
        if row["requirement_type"] == "documentary_transfer_tax_city"
    )
    component_sets = {
        transaction_id: {row["requirement_type"] for row in rows if row["transaction_id"] == transaction_id}
        for transaction_id in included_ids
    }
    reconciliation_flag = all(
        components == {"documentary_transfer_tax_county", "documentary_transfer_tax_city"}
        for components in component_sets.values()
    )
    return {
        "case_id": case_id,
        "rate_version": rate_version,
        "approved_base_usd": approved_base_usd,
        "county_tax_usd": county_tax_usd,
        "city_tax_usd": city_tax_usd,
        "combined_tax_usd": county_tax_usd + city_tax_usd,
        "included_transaction_ids": included_ids,
        "excluded_qualifying_transaction_ids": [row["transaction_id"] for row in excluded_rows],
        "reconciliation_flag": reconciliation_flag,
    }
