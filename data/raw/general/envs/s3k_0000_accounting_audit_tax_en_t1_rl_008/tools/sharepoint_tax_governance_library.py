"""SQLite-backed tools for the Microsoft SharePoint Tax Governance Library."""

import sqlite3
from datetime import date
from pathlib import Path


_DOCUMENT_TYPES = {
    "memorandum", "presentation", "workbook", "assumption_schedule",
    "irs_publication", "california_ftb_guidance", "transfer_tax_form",
    "public_company_reference",
}
_DOCUMENT_STATUSES = {"draft", "in_review", "approved", "retired"}
_DECISION_STATUSES = {"draft", "pending_approval", "approved", "retired"}
_SOURCE_TYPES = {
    "workbook_cell", "source_document", "assumption", "transaction_record",
    "execution_requirement",
}
_REVIEW_STATUSES = {"current", "approved", "superseded", "rejected"}
_STANDARD_REQUIREMENTS = {
    "collateral", "payment_covenant", "buyer_solvency_diligence",
    "california_withholding", "federal_nonforeign_affidavit",
}


def _con() -> sqlite3.Connection:
    con = sqlite3.connect(
        Path(__file__).resolve().parent.parent / "system" / "sharepoint_tax_governance_library" / "state.db"
    )
    con.row_factory = sqlite3.Row
    return con


def _page_offset(page_token: str | None) -> int:
    if page_token is None:
        return 0
    try:
        offset = int(page_token)
    except ValueError as error:
        raise ValueError("page_token must be a non-negative integer offset") from error
    if offset < 0:
        raise ValueError("page_token must be a non-negative integer offset")
    return offset


def _validate_page_size(page_size: int, maximum: int = 100) -> None:
    if not 1 <= page_size <= maximum:
        raise ValueError(f"page_size must be between 1 and {maximum}")


def _validate_iso_date(value: str | None, name: str) -> None:
    if value is None:
        return
    try:
        date.fromisoformat(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be an ISO date in YYYY-MM-DD format") from error


def _row_dicts(rows: list[sqlite3.Row]) -> list[dict]:
    return [dict(row) for row in rows]


def search_tax_documents(
    query: str,
    document_type: str | None = None,
    case_id: str | None = None,
    approval_status: str = "approved",
    effective_from: str | None = None,
    effective_to: str | None = None,
    page_size: int = 25,
    page_token: str | None = None,
) -> dict:
    """Search controlled tax documents using text and governance filters.

    Args:
        query: Required text matched against document names, authorities, and IDs.
        document_type: Optional controlled document type to restrict the search.
        case_id: Optional transaction or planning case identifier.
        approval_status: Document lifecycle status; defaults to approved documents.
        effective_from: Optional inclusive ISO-8601 effective-date lower bound.
        effective_to: Optional inclusive ISO-8601 effective-date upper bound.
        page_size: Number of records to return, from 1 through 100.
        page_token: Optional non-negative offset returned by a prior page.

    Returns:
        A dictionary containing items, next_page_token, and total_estimate.
    """
    if not query.strip():
        raise ValueError("query must contain search text; this tool does not support full-table exports")
    if document_type is not None and document_type not in _DOCUMENT_TYPES:
        raise ValueError(f"document_type must be one of: {', '.join(sorted(_DOCUMENT_TYPES))}")
    if approval_status not in _DOCUMENT_STATUSES:
        raise ValueError(f"approval_status must be one of: {', '.join(sorted(_DOCUMENT_STATUSES))}")
    if effective_from is not None and effective_to is not None and effective_from > effective_to:
        raise ValueError("effective_from must be on or before effective_to")
    _validate_iso_date(effective_from, "effective_from")
    _validate_iso_date(effective_to, "effective_to")
    _validate_page_size(page_size)
    offset = _page_offset(page_token)
    clauses = ["approval_status = ?", "(document_name LIKE ? OR source_authority LIKE ? OR document_id LIKE ?)"]
    term = f"%{query.strip()}%"
    params: list = [approval_status, term, term, term]
    if document_type is not None:
        clauses.append("document_type = ?")
        params.append(document_type)
    if case_id is not None:
        clauses.append("case_id = ?")
        params.append(case_id)
    if effective_from is not None:
        clauses.append("effective_date >= ?")
        params.append(effective_from)
    if effective_to is not None:
        clauses.append("effective_date <= ?")
        params.append(effective_to)
    where = " WHERE " + " AND ".join(clauses)
    columns = "document_id, document_name, document_type, version, effective_date, prepared_by_id, reviewed_by_id, approval_status, case_id, file_uri"
    con = _con()
    try:
        total = con.execute(f"SELECT COUNT(*) FROM controlled_tax_documents{where}", params).fetchone()[0]
        rows = con.execute(
            f"SELECT {columns} FROM controlled_tax_documents{where} "
            "ORDER BY effective_date DESC, document_id DESC LIMIT ? OFFSET ?",
            [*params, page_size, offset],
        ).fetchall()
    finally:
        con.close()
    next_token = str(offset + page_size) if offset + page_size < total else None
    return {"items": _row_dicts(rows), "next_page_token": next_token, "total_estimate": total}


def get_controlled_assumptions(
    case_id: str,
    as_of_date: str | None = None,
    assumption_name: str | None = None,
    include_sources: bool = True,
) -> dict:
    """Retrieve approved assumption-schedule evidence for one tax planning case.

    Args:
        case_id: Required case identifier for the controlled assumption schedule.
        as_of_date: Optional ISO-8601 date limiting documents and evidence effective on or before it.
        assumption_name: Optional anchor-name text filter for a particular assumption.
        include_sources: Whether to include approved authority documents supporting the case.

    Returns:
        A dictionary with case metadata, the selected assumption document, assumptions, and optional source_documents.
    """
    if not case_id.strip():
        raise ValueError("case_id is required")
    _validate_iso_date(as_of_date, "as_of_date")
    con = _con()
    try:
        document_sql = (
            "SELECT document_id, version, effective_date, prepared_by_id, reviewed_by_id, approval_status "
            "FROM controlled_tax_documents WHERE case_id = ? AND document_type = 'assumption_schedule' "
            "AND approval_status = 'approved'"
        )
        document_params: list = [case_id]
        if as_of_date is not None:
            document_sql += " AND effective_date <= ?"
            document_params.append(as_of_date)
        document_sql += " ORDER BY effective_date DESC, document_id DESC LIMIT 1"
        document = con.execute(document_sql, document_params).fetchone()
        if document is None:
            raise ValueError("no approved assumption schedule was found for the requested case and date")
        evidence_sql = (
            "SELECT anchor_name, observed_value_text AS value_text, observed_amount_cents AS amount_cents, "
            "source_reference, review_status, effective_date "
            "FROM document_evidence_register WHERE document_id = ? AND case_id = ? "
            "AND source_type = 'assumption' AND review_status IN ('approved', 'current')"
        )
        evidence_params: list = [document["document_id"], case_id]
        if as_of_date is not None:
            evidence_sql += " AND effective_date <= ?"
            evidence_params.append(as_of_date)
        if assumption_name is not None:
            evidence_sql += " AND anchor_name LIKE ?"
            evidence_params.append(f"%{assumption_name.strip()}%")
        assumptions = _row_dicts(con.execute(evidence_sql + " ORDER BY effective_date, anchor_name", evidence_params).fetchall())
        source_documents = None
        if include_sources:
            source_documents = _row_dicts(con.execute(
                "SELECT document_id, document_name, source_authority, version "
                "FROM controlled_tax_documents WHERE case_id = ? AND approval_status = 'approved' "
                "AND document_type IN ('irs_publication', 'california_ftb_guidance') "
                "ORDER BY effective_date, document_id",
                [case_id],
            ).fetchall())
    finally:
        con.close()
    return {
        "case_id": case_id,
        "as_of_date": as_of_date,
        "assumption_document": dict(document),
        "assumptions": assumptions,
        "source_documents": source_documents,
    }


def get_decision_record(
    decision_id: str | None = None,
    case_id: str | None = None,
    status: str = "approved",
    latest_only: bool = True,
) -> dict:
    """Retrieve governed tax decisions by decision identifier or case identifier.

    Args:
        decision_id: Optional unique decision record identifier.
        case_id: Optional transaction or planning case identifier.
        status: Lifecycle status to return; defaults to approved decisions.
        latest_only: When true, return only the newest matching record by approval and creation timestamp.

    Returns:
        A dictionary containing decision items and their count, including lineage and retirement flags.
    """
    if decision_id is None and case_id is None:
        raise ValueError("provide decision_id or case_id; unfiltered decision exports are not supported")
    if status not in _DECISION_STATUSES:
        raise ValueError(f"status must be one of: {', '.join(sorted(_DECISION_STATUSES))}")
    clauses = ["d.status = ?"]
    params: list = [status]
    if decision_id is not None:
        clauses.append("d.decision_id = ?")
        params.append(decision_id)
    if case_id is not None:
        clauses.append("d.case_id = ?")
        params.append(case_id)
    con = _con()
    try:
        sql = (
            "SELECT d.decision_id, d.document_id, d.case_id, d.decision_code, d.recommendation_text, d.fallback_text, "
            "d.reviewer_id, d.reviewer_entity_id, d.status, d.approved_at, d.draft_score, d.current_score, d.decision_version, "
            "CASE WHEN d.status = 'approved' AND d.decision_id = (SELECT d2.decision_id FROM decision_records d2 "
            "WHERE d2.case_id = d.case_id AND d2.status = 'approved' ORDER BY d2.approved_at DESC, d2.created_at DESC, d2.decision_id DESC LIMIT 1) THEN 1 ELSE 0 END AS is_latest_approved, "
            "CASE WHEN d.status = 'retired' OR d.case_id LIKE '%RETIRED%' THEN 1 ELSE 0 END AS is_retired_case, "
            "CASE WHEN d.status = 'approved' AND d.case_id NOT LIKE '%RETIRED%' THEN 1 ELSE 0 END AS is_current_case_lineage "
            "FROM decision_records d WHERE " + " AND ".join(clauses) +
            " ORDER BY COALESCE(d.approved_at, d.created_at) DESC, d.decision_id DESC"
        )
        if latest_only:
            sql += " LIMIT 1"
        rows = con.execute(sql, params).fetchall()
    finally:
        con.close()
    return {"items": _row_dicts(rows), "count": len(rows)}


def get_document_evidence(
    document_id: str | None = None,
    case_id: str | None = None,
    anchor_name: str | None = None,
    source_type: str | None = None,
    review_status: str = "current",
    page_size: int = 50,
    page_token: str | None = None,
) -> dict:
    """Return paginated evidence-register records for a filtered controlled document lineage.

    Args:
        document_id: Optional controlled document identifier.
        case_id: Optional transaction or planning case identifier.
        anchor_name: Optional evidence anchor identifier.
        source_type: Optional evidence source category.
        review_status: Evidence review status to return; defaults to current.
        page_size: Number of evidence rows to return, from 1 through 100.
        page_token: Optional non-negative offset returned by a prior page.

    Returns:
        A dictionary containing matching items and a next_page_token when additional rows exist.
    """
    if document_id is None and case_id is None and anchor_name is None:
        raise ValueError("provide document_id, case_id, or anchor_name; unfiltered evidence exports are not supported")
    if source_type is not None and source_type not in _SOURCE_TYPES:
        raise ValueError(f"source_type must be one of: {', '.join(sorted(_SOURCE_TYPES))}")
    if review_status not in _REVIEW_STATUSES:
        raise ValueError(f"review_status must be one of: {', '.join(sorted(_REVIEW_STATUSES))}")
    _validate_page_size(page_size)
    offset = _page_offset(page_token)
    clauses = ["review_status = ?"]
    params: list = [review_status]
    for field, value in (("document_id", document_id), ("case_id", case_id), ("anchor_name", anchor_name), ("source_type", source_type)):
        if value is not None:
            clauses.append(f"{field} = ?")
            params.append(value)
    where = " WHERE " + " AND ".join(clauses)
    columns = "evidence_id, document_id, case_id, anchor_name, source_type, source_reference, evidence_text, review_status, reviewer_id, reviewer_entity_id, effective_date, requirement_type, observed_value_text, observed_amount_cents"
    con = _con()
    try:
        total = con.execute(f"SELECT COUNT(*) FROM document_evidence_register{where}", params).fetchone()[0]
        rows = con.execute(
            f"SELECT {columns} FROM document_evidence_register{where} "
            "ORDER BY effective_date DESC, evidence_id DESC LIMIT ? OFFSET ?",
            [*params, page_size, offset],
        ).fetchall()
    finally:
        con.close()
    next_token = str(offset + page_size) if offset + page_size < total else None
    return {"items": _row_dicts(rows), "next_page_token": next_token}


def check_document_lineage(
    document_id: str,
    case_id: str | None = None,
    anchor_name: str | None = None,
    include_retired: bool = False,
) -> dict:
    """Assess document-to-decision, evidence, authority, and closing-requirement lineage.

    Args:
        document_id: Required memorandum, presentation, or other controlled document identifier.
        case_id: Optional case constraint; defaults to the document's controlled case.
        anchor_name: Optional evidence anchor constraint for focused lineage review.
        include_retired: Whether retired decisions and evidence may contribute to the lineage result.

    Returns:
        A dictionary with lineage status, node counts, decisions, requirements, missing links, retired references, and approvals.
    """
    if not document_id.strip():
        raise ValueError("document_id is required")
    con = _con()
    try:
        document = con.execute("SELECT * FROM controlled_tax_documents WHERE document_id = ?", [document_id]).fetchone()
        if document is None:
            raise ValueError("document_id was not found")
        selected_case = case_id or document["case_id"]
        if selected_case is None:
            raise ValueError("case_id is required because the controlled document has no case assignment")
        if document["case_id"] != selected_case:
            raise ValueError("case_id does not match the controlled document")
        evidence_clauses = ["case_id = ?"]
        evidence_params: list = [selected_case]
        if anchor_name is not None:
            evidence_clauses.append("anchor_name = ?")
            evidence_params.append(anchor_name)
        if not include_retired:
            evidence_clauses.append("review_status IN ('approved', 'current')")
        evidence_where = " AND ".join(evidence_clauses)
        evidence_rows = con.execute(
            f"SELECT * FROM document_evidence_register WHERE {evidence_where}", evidence_params
        ).fetchall()
        decision_status = "IN ('approved', 'retired')" if include_retired else "= 'approved'"
        decisions = con.execute(
            f"SELECT * FROM decision_records WHERE case_id = ? AND status {decision_status} ORDER BY approved_at, created_at",
            [selected_case],
        ).fetchall()
        related_docs = con.execute(
            "SELECT * FROM controlled_tax_documents WHERE case_id = ? "
            + ("" if include_retired else "AND approval_status = 'approved' ")
            + "ORDER BY effective_date, document_id",
            [selected_case],
        ).fetchall()
        requirements = sorted({row["requirement_type"] for row in evidence_rows if row["requirement_type"] in _STANDARD_REQUIREMENTS})
        missing_links: list[dict] = []
        if not decisions:
            missing_links.append({"from_type": "document", "from_id": document_id, "to_type": "approved_decision", "reason": "No approved decision exists for the case."})
        if not evidence_rows:
            missing_links.append({"from_type": "document", "from_id": document_id, "to_type": "evidence", "reason": "No eligible evidence exists for the requested case and anchor."})
        if not any(row["document_type"] == "workbook" for row in related_docs):
            missing_links.append({"from_type": "case", "from_id": selected_case, "to_type": "workbook", "reason": "No eligible controlled workbook exists for the case."})
        if not any(row["document_type"] == "assumption_schedule" for row in related_docs):
            missing_links.append({"from_type": "case", "from_id": selected_case, "to_type": "assumption_schedule", "reason": "No eligible assumption schedule exists for the case."})
        approvals = []
        for row in decisions:
            if row["approved_at"] is not None:
                approvals.append({"reviewer_id": row["reviewer_id"], "status": row["status"], "approved_at": row["approved_at"]})
        retired_references = _row_dicts(con.execute(
            "SELECT decision_id, document_id, case_id FROM decision_records WHERE case_id = ? AND status = 'retired'", [selected_case]
        ).fetchall())
        counts = {
            "documents": len(related_docs),
            "decisions": len(decisions),
            "workbooks": sum(row["document_type"] == "workbook" for row in related_docs),
            "assumption_schedules": sum(row["document_type"] == "assumption_schedule" for row in related_docs),
            "source_documents": sum(row["document_type"] in {"irs_publication", "california_ftb_guidance"} for row in related_docs),
            "evidence_rows": len(evidence_rows),
        }
    finally:
        con.close()
    return {
        "document_id": document_id,
        "case_id": selected_case,
        "lineage_status": "complete" if not missing_links else "incomplete",
        "node_counts": counts,
        "approved_decision_ids": [row["decision_id"] for row in decisions if row["status"] == "approved"],
        "requirement_types": requirements,
        "missing_requirement_types": sorted(_STANDARD_REQUIREMENTS - set(requirements)),
        "missing_links": missing_links,
        "retired_references": retired_references,
        "review_approvals": approvals,
    }
