CREATE TABLE disposition_transactions (
  transaction_id TEXT PRIMARY KEY,
  case_id TEXT NOT NULL,
  property_id TEXT NOT NULL,
  buyer_entity_id TEXT NOT NULL,
  structure_code TEXT NOT NULL CHECK (structure_code IN ('direct_sale','equity_transfer','installment_sale')),
  transaction_status TEXT NOT NULL CHECK (transaction_status IN ('active','recommended','superseded','closed','draft')),
  gross_consideration_usd INTEGER NOT NULL CHECK (gross_consideration_usd >= 0),
  declared_tax_consideration_usd INTEGER NOT NULL CHECK (declared_tax_consideration_usd >= 0),
  cash_consideration_usd INTEGER NOT NULL CHECK (cash_consideration_usd >= 0),
  closing_principal_usd INTEGER NOT NULL CHECK (closing_principal_usd >= 0),
  secured_seller_note_principal_usd INTEGER NOT NULL CHECK (secured_seller_note_principal_usd >= 0),
  liability_relief_usd INTEGER NOT NULL CHECK (liability_relief_usd >= 0),
  seller_note_interest_rate_bps INTEGER CHECK (seller_note_interest_rate_bps IS NULL OR seller_note_interest_rate_bps >= 0),
  seller_note_term_months INTEGER CHECK (seller_note_term_months IS NULL OR seller_note_term_months > 0),
  closing_date TEXT NOT NULL,
  rate_version TEXT NOT NULL,
  notes TEXT,
  UNIQUE (case_id, structure_code, transaction_id)
);

CREATE TABLE closing_requirements (
  requirement_id TEXT PRIMARY KEY,
  transaction_id TEXT NOT NULL,
  case_id TEXT NOT NULL,
  requirement_type TEXT NOT NULL,
  owner_role TEXT NOT NULL,
  status TEXT NOT NULL CHECK (status IN ('open','in_review','approved','complete','waived','rejected')),
  amount_usd INTEGER CHECK (amount_usd IS NULL OR amount_usd >= 0),
  due_date TEXT NOT NULL,
  active_flag INTEGER NOT NULL CHECK (active_flag IN (0,1)),
  qualifying_instrument_flag INTEGER NOT NULL DEFAULT 0 CHECK (qualifying_instrument_flag IN (0,1)),
  rate_version TEXT,
  tax_component_code TEXT,
  declared_consideration_usd INTEGER CHECK (declared_consideration_usd IS NULL OR declared_consideration_usd >= 0),
  evidence_reference TEXT,
  notes TEXT,
  FOREIGN KEY (transaction_id) REFERENCES disposition_transactions(transaction_id)
);

CREATE TABLE execution_activity_log (
  activity_id TEXT PRIMARY KEY,
  transaction_id TEXT,
  case_id TEXT NOT NULL,
  activity_type TEXT NOT NULL,
  actor_id TEXT NOT NULL,
  activity_timestamp TEXT NOT NULL,
  activity_status TEXT NOT NULL CHECK (activity_status IN ('logged','completed','pending','superseded','rejected')),
  source_reference TEXT NOT NULL,
  activity_text TEXT,
  FOREIGN KEY (transaction_id) REFERENCES disposition_transactions(transaction_id)
);