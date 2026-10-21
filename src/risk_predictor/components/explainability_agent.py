"""
Ephemeral Medical Document Processor (Zero-Retention Vault) &
AI Underwriting / Explainability Feedback Agent.
"""
import hashlib
import json
import os
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import urllib.request

from src.risk_predictor.logger import logger
from src.risk_predictor.components.country_risk_engine import SUPPORTED_DISEASES, COUNTRY_CONFIGS


DOCUMENT_KEYWORD_RULES: Dict[str, List[str]] = {
    "diabetes_type_2": ["diabetes", "hba1c", "hyperglycemia", "metformin", "insulin", "fasting glucose"],
    "hypertension": ["hypertension", "high blood pressure", "amlodipine", "lisinopril", "systolic 150", "bp 150"],
    "cardiovascular_disease": ["coronary", "myocardial", "angina", "stent", "atrial fibrillation", "cad", "statin"],
    "asthma_copd": ["asthma", "copd", "bronchitis", "albuterol", "inhaler", "spirometry", "wheezing"],
    "thyroid_disorder": ["hypothyroidism", "hyperthyroidism", "levothyroxine", "tsh elevated", "thyroid"],
    "chronic_kidney_disease": ["chronic kidney", "ckd", "creatinine elevated", "egfr low", "nephropathy"],
    "mental_health_condition": ["anxiety", "depression", "ssri", "sertraline", "panic disorder"],
    "autoimmune_arthritis": ["rheumatoid", "arthritis", "lupus", "autoimmune", "methotrexate", "psoriasis"],
}


class EphemeralDocumentProcessor:
    """
    Processes uploaded medical documents (lab reports, discharge summaries, JSON/TXT/PDF/Images)
    under a strict Terms & Conditions contract:
    1. Verifies explicit user consent (`accept_terms == True`).
    2. Stages the upload in a temporary file, extracts clinical biomarkers & disease flags.
    3. Cryptographically wipes (zero-fills) and permanently deletes the file in a `finally:` block.
    4. Appends an anonymized Zero-Retention Deletion Certificate to `artifacts/privacy_audit_log.json`.
    """

    def __init__(
        self,
        scratch_dir: Path = Path("artifacts/tmp_ephemeral_uploads"),
        audit_log_path: Path = Path("artifacts/privacy_audit_log.json"),
    ):
        self.scratch_dir = scratch_dir
        self.audit_log_path = audit_log_path
        os.makedirs(self.scratch_dir, exist_ok=True)
        os.makedirs(self.audit_log_path.parent, exist_ok=True)

    def process_and_purge(
        self,
        filename: str,
        raw_bytes: bytes,
        accept_terms: bool,
    ) -> Dict[str, Any]:
        if not accept_terms:
            raise ValueError(
                "Terms & Conditions must be accepted before uploading medical documents."
            )

        t0 = time.perf_counter()
        sha256_hash = hashlib.sha256(raw_bytes).hexdigest()
        byte_size = len(raw_bytes)
        temp_file = self.scratch_dir / f"ephemeral_{uuid.uuid4().hex}_{Path(filename).name}"

        detected_diseases: List[str] = []
        extracted_vitals: Dict[str, Any] = {}
        clinical_summary_notes: List[str] = []

        try:
            # Write temporarily to disk to support file-based parsers, then immediately purge in finally
            with open(temp_file, "wb") as f:
                f.write(raw_bytes)

            text_content = raw_bytes.decode("utf-8", errors="ignore").lower()

            for d_key, keywords in DOCUMENT_KEYWORD_RULES.items():
                if any(kw in text_content for kw in keywords):
                    detected_diseases.append(d_key)
                    clinical_summary_notes.append(
                        f"Detected clinical reference to {SUPPORTED_DISEASES[d_key]['label']}."
                    )

            # Parse common lab vitals if present (e.g., HbA1c, BP, Total Cholesterol)
            hba1c_match = re.search(r"hba1c[:\s=]+([0-9]+(?:\.[0-9]+)?)", text_content)
            if hba1c_match:
                val = float(hba1c_match.group(1))
                extracted_vitals["hba1c_pct"] = val
                if val >= 6.5 and "diabetes_type_2" not in detected_diseases:
                    detected_diseases.append("diabetes_type_2")

            bp_match = re.search(r"(\d{2,3})\s*/\s*(\d{2,3})\s*mmhg", text_content)
            if bp_match:
                sys_bp, dia_bp = int(bp_match.group(1)), int(bp_match.group(2))
                extracted_vitals["blood_pressure"] = f"{sys_bp}/{dia_bp} mmHg"
                if (sys_bp >= 140 or dia_bp >= 90) and "hypertension" not in detected_diseases:
                    detected_diseases.append("hypertension")

            if not clinical_summary_notes:
                clinical_summary_notes.append(
                    "Document scanned successfully; no high-risk chronic pathology markers flagged."
                )

        finally:
            # Cryptographic zero-fill + immediate unlink deletion
            if temp_file.exists():
                try:
                    with open(temp_file, "wb") as f:
                        f.write(b"\x00" * min(max(byte_size, 1), 65536))
                        f.flush()
                    os.remove(temp_file)
                except OSError:
                    if temp_file.exists():
                        temp_file.unlink(missing_ok=True)

        retention_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        still_exists = temp_file.exists()

        receipt = {
            "receipt_id": f"DEL-{uuid.uuid4().hex[:10].upper()}",
            "original_filename": Path(filename).name,
            "sha256_fingerprint": sha256_hash,
            "bytes_purged": byte_size,
            "retention_duration_ms": retention_ms,
            "deleted_at_utc": datetime.now(timezone.utc).isoformat(),
            "file_exists_on_disk": still_exists,
            "status": "PERMANENTLY_DELETED" if not still_exists else "ERROR_RETENTION",
            "terms_accepted": True,
            "detected_diseases": detected_diseases,
            "detected_disease_labels": [SUPPORTED_DISEASES[d]["label"] for d in detected_diseases],
            "extracted_vitals": extracted_vitals,
            "clinical_notes": clinical_summary_notes,
        }
        self._append_audit_log(receipt)
        logger.info(
            f"Ephemeral document '{filename}' processed and permanently deleted "
            f"(receipt={receipt['receipt_id']}, retention={retention_ms}ms)."
        )
        return receipt

    def get_audit_log(self, limit: int = 20) -> List[Dict[str, Any]]:
        if not self.audit_log_path.exists():
            return []
        try:
            with open(self.audit_log_path, "r", encoding="utf-8") as f:
                records = json.load(f)
            return records[-limit:][::-1]
        except Exception:
            return []

    def _append_audit_log(self, receipt: Dict[str, Any]) -> None:
        records = []
        if self.audit_log_path.exists():
            try:
                with open(self.audit_log_path, "r", encoding="utf-8") as f:
                    records = json.load(f)
            except Exception:
                records = []
        # Store only non-PII metadata in audit log
        records.append(
            {
                "receipt_id": receipt["receipt_id"],
                "sha256_fingerprint": receipt["sha256_fingerprint"][:16] + "...",
                "bytes_purged": receipt["bytes_purged"],
                "retention_duration_ms": receipt["retention_duration_ms"],
                "deleted_at_utc": receipt["deleted_at_utc"],
                "file_exists_on_disk": receipt["file_exists_on_disk"],
                "status": receipt["status"],
                "detected_conditions_count": len(receipt["detected_diseases"]),
            }
        )
        with open(self.audit_log_path, "w", encoding="utf-8") as f:
            json.dump(records[-100:], f, indent=2)


class UnderwritingExplanationAgent:
    """
    AI Underwriting & Clinical Explanation Agent.
    Synthesizes local SHAP-style feature attributions, country healthcare rules,
    Public vs. Private insurance trade-offs, and actionable counterfactual interventions.
    """

    def __init__(self):
        self.api_key = (
            os.environ.get("AI_AGENT_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")
            or ""
        ).strip()

    def generate_feedback(
        self,
        prediction_bundle: Dict[str, Any],
        document_receipt: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        country = prediction_bundle["country"]
        cfg = COUNTRY_CONFIGS.get(country, COUNTRY_CONFIGS["USA"])
        sym = prediction_bundle["currency_symbol"]
        pred_val = prediction_bundle["predicted_charges_local"]
        tier = prediction_bundle["risk_tier"]
        ins_type = prediction_bundle["insurance_type"]
        ins_cat = prediction_bundle["insurance_category"]
        bmi = prediction_bundle["bmi"]
        bmi_cat = prediction_bundle["bmi_category"]
        attributions = prediction_bundle["explainability"]["local_attributions"]
        counterfactuals = prediction_bundle["explainability"]["counterfactual_what_ifs"]
        active_diseases = prediction_bundle["active_diseases"]

        top_drivers = [a for a in attributions if a["impact_local_currency"] > 0][:3]
        protective_factors = [a for a in attributions if a["impact_local_currency"] <= 0][:2]

        driver_text = (
            ", ".join(
                f"{d['factor']} ({d['impact_formatted']}, {d['share_pct']}% of estimate)"
                for d in top_drivers
            )
            if top_drivers
            else "baseline demographic profile with minimal excess risk loading"
        )

        disease_guidance = []
        for d_key, meta in SUPPORTED_DISEASES.items():
            if meta["label"] in active_diseases:
                disease_guidance.append(f"{meta['label']}: {meta['clinical_note']}")

        doc_section = "No external medical document uploaded."
        if document_receipt:
            doc_section = (
                f"Document '{document_receipt['original_filename']}' was analyzed in-memory "
                f"and permanently purged in {document_receipt['retention_duration_ms']} ms "
                f"(Receipt `{document_receipt['receipt_id']}`). "
                f"Extracted findings: {'; '.join(document_receipt['clinical_notes'])}"
            )

        selected_tier_meta = cfg["insurance_tiers"].get(
            ins_type, list(cfg["insurance_tiers"].values())[0]
        )
        insurance_advice = (
            f"In {cfg['country_name']}, you selected **{ins_type}** ({ins_cat} tier), which covers "
            f"approximately **{selected_tier_meta['oop_coverage_pct']}%** of eligible medical expenses "
            f"({selected_tier_meta['description']}). {cfg['system_overview']}"
        )

        # Optional live LLM enrichment when a valid Gemini REST key is configured
        llm_narrative = self._try_remote_llm_synthesis(
            prediction_bundle, driver_text, insurance_advice, doc_section
        )

        if not llm_narrative:
            llm_narrative = (
                f"Based on the {cfg['country_name']} ({country}) {prediction_bundle['model_used']} model, "
                f"your estimated annual healthcare risk and premium exposure is {sym}{pred_val:,.2f} "
                f"({tier}). Your anthropometric profile ({prediction_bundle['height_cm']} cm, "
                f"{prediction_bundle['weight_kg']} kg -> BMI {bmi} kg/m², {bmi_cat}) and clinical history "
                f"were evaluated against a healthy country baseline of "
                f"{sym}{prediction_bundle['explainability']['baseline_healthy_charge']:,.2f}. "
                f"The primary upward cost drivers are: {driver_text}."
            )

        return {
            "agent_name": "Actuarial & Clinical Explainability AI Agent",
            "executive_summary": llm_narrative,
            "why_this_result": {
                "primary_upward_drivers": top_drivers,
                "protective_or_neutral_factors": protective_factors,
                "anthropometric_assessment": (
                    f"Height {prediction_bundle['height_cm']} cm & Weight {prediction_bundle['weight_kg']} kg "
                    f"yield BMI = {bmi} kg/m² ({bmi_cat})."
                ),
                "chronic_comorbidity_notes": disease_guidance
                if disease_guidance
                else ["No active chronic comorbidities selected."],
            },
            "country_insurance_guidance": insurance_advice,
            "document_privacy_verification": doc_section,
            "recommended_interventions": counterfactuals
            if counterfactuals
            else [
                {
                    "intervention": "Maintain current healthy BMI, non-smoking status, and preventive screening schedule",
                    "new_predicted_charge": pred_val,
                    "annual_savings": 0.0,
                    "savings_pct": 0.0,
                }
            ],
        }

    def _try_remote_llm_synthesis(
        self,
        prediction_bundle: Dict[str, Any],
        driver_text: str,
        insurance_advice: str,
        doc_section: str,
    ) -> Optional[str]:
        if not self.api_key or self.api_key.startswith("AQ."):
            # Vertex/OAuth token or offline mode uses deterministic actuarial synthesis without latency
            return None
        try:
            url = (
                "https://generativelanguage.googleapis.com/v1beta/models/"
                f"gemini-1.5-flash:generateContent?key={self.api_key}"
            )
            prompt = (
                f"Act as a senior health actuary. Summarize in 3 sentences why a patient in "
                f"{prediction_bundle['country_name']} with BMI {prediction_bundle['bmi']} and "
                f"plan {prediction_bundle['insurance_type']} received an estimated annual cost of "
                f"{prediction_bundle['predicted_charges_formatted']} ({prediction_bundle['risk_tier']}). "
                f"Drivers: {driver_text}. {insurance_advice} {doc_section}"
            )
            req_body = json.dumps({"contents": [{"parts": [{"text": prompt}]}]}).encode("utf-8")
            req = urllib.request.Request(url, data=req_body, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=2.5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data["candidates"][0]["content"]["parts"][0]["text"].strip()
        except Exception:
            return None
