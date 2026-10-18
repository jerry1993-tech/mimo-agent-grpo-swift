CREATE TABLE disposal_cases (
  case_id TEXT PRIMARY KEY,
  entity_id TEXT NOT NULL,
  property_id TEXT NOT NULL,
  taxpayer_name TEXT NOT NULL,
  property_name TEXT NOT NULL,
  structure_code TEXT NOT NULL CHECK (structure_code IN ('DIRECT','EQUITY','INSTALLMENT')),
  closing_date TEXT NOT NULL,
  tax_year INTEGER NOT NULL,
  status TEXT NOT NULL CHECK (status IN ('MODELED','APPROVED','ARCHIVED')),
  executive_gain_usd_cents INTEGER NOT NULL,
  currency_code TEXT NOT NULL DEFAULT 'USD',
  source_reference TEXT,
  created_at TEXT NOT NULL
);

CREATE TABLE tax_basis_components (
  basis_component_id TEXT PRIMARY KEY,
  case_id TEXT NOT NULL,
  property_id TEXT NOT NULL,
  component_type TEXT NOT NULL CHECK (component_type IN ('AMOUNT_REALIZED','LIABILITY_RELIEF','ADJUSTED_TAX_BASIS','SELLING_COST','ORIGINAL_COST','ACCUMULATED_DEPRECIATION','INSIDE_BASIS_ADJUSTMENT','OUTSIDE_EQUITY_BASIS')),
  amount_usd_cents INTEGER NOT NULL,
  tax_character_code TEXT NOT NULL,
  source_period TEXT NOT NULL,
  source_system TEXT NOT NULL CHECK (source_system IN ('DEALCLOUD','YARDI','KPMG_TAX_MODEL')),
  approved_flag INTEGER NOT NULL CHECK (approved_flag IN (0,1)),
  superseded_flag INTEGER NOT NULL DEFAULT 0 CHECK (superseded_flag IN (0,1)),
  description TEXT,
  FOREIGN KEY (case_id) REFERENCES disposal_cases(case_id)
);

CREATE TABLE recognition_schedules (
  schedule_line_id TEXT PRIMARY KEY,
  case_id TEXT NOT NULL,
  payment_reference TEXT NOT NULL,
  payment_date TEXT NOT NULL,
  principal_usd_cents INTEGER NOT NULL,
  gross_profit_pct_bps INTEGER NOT NULL,
  recognized_gain_usd_cents INTEGER NOT NULL,
  deferred_gain_usd_cents INTEGER NOT NULL,
  character_code TEXT NOT NULL CHECK (character_code IN ('IRC_1245','IRC_291','IRC_1250_UNRECAPTURED','IRC_1231','IRC_751','IRC_741','ELIGIBLE_GAIN')),
  recapture_exempt INTEGER NOT NULL CHECK (recapture_exempt IN (0,1)),
  deferred_gain_flag INTEGER NOT NULL CHECK (deferred_gain_flag IN (0,1)),
  approved_flag INTEGER NOT NULL CHECK (approved_flag IN (0,1)),
  superseded_flag INTEGER NOT NULL DEFAULT 0 CHECK (superseded_flag IN (0,1)),
  import_batch TEXT NOT NULL,
  FOREIGN KEY (case_id) REFERENCES disposal_cases(case_id)
);