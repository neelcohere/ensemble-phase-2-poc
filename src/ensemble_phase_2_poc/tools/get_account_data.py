from typing import List, Dict, Any
from pydantic import BaseModel
from ensemble_phase_2_poc.tools.base_tool import Tool


class GetAccountDataInput(BaseModel):
    """Input schema for GetAccountData - no args from LLM, uses bound values"""


class GetAccountData(Tool):
    """Get account data - injected args are stored as instance attributes"""

    name: str = "get_account_data"
    description: str = Tool.get_tool_description(name)
    args_schema: type = GetAccountDataInput
    include_in_scorer_check: bool = False

    # Injected values (set at instantiation)
    account_number: str = ""
    client_name: str = ""
    facility_prefix: str = ""
    lob: str = ""

    def _execute(self) -> List[Dict[str, Any]]:
        """Sample output - uses self.account_number, etc."""
        self.logger.info(f"Fetching account data for account: {self.account_number}")
        self.logger.info(f"Client: {self.client_name}, Facility: {self.facility_prefix}, LOB: {self.lob}")
        return [
            {
                "account_number": "ACC-12345",
                "client_name": "Acme Healthcare",
                "facility_prefix": "FAC",
                "lob": "Commercial",
                "patient": {
                    "patient_id": "PAT-78901",
                    "first_name": "Maria",
                    "last_name": "Rodriguez",
                    "dob": "1985-03-15",
                    "ssn_last_four": "4521",
                    "address": {
                        "street": "1234 Oak Lane",
                        "city": "Austin",
                        "state": "TX",
                        "zip": "78701",
                    },
                    "phone": "512-555-0147",
                    "email": "m.rodriguez@email.com",
                },
                "insurance": {
                    "primary": {
                        "payer_id": "BCBS-TX-001",
                        "payer_name": "Blue Cross Blue Shield of Texas",
                        "plan_type": "PPO",
                        "member_id": "XYZ123456789",
                        "group_number": "GRP-55012",
                        "effective_date": "2024-01-01",
                        "copay": 25.00,
                        "deductible": 1500.00,
                        "deductible_met": 875.00,
                    },
                    "secondary": None,
                },
                "claims": [
                    {
                        "claim_id": "CLM-2025-098765",
                        "date_of_service": "2025-01-10",
                        "date_submitted": "2025-01-12",
                        "provider": {
                            "npi": "1234567890",
                            "name": "Dr. Sarah Chen",
                            "facility": "Austin Medical Center",
                            "tax_id": "74-1234567",
                        },
                        "diagnosis_codes": ["E11.9", "I10"],
                        "procedure_codes": [
                            {
                                "cpt": "99214",
                                "description": "Office visit, established patient",
                                "units": 1,
                                "charge": 185.00,
                            },
                            {
                                "cpt": "36415",
                                "description": "Venipuncture",
                                "units": 1,
                                "charge": 25.00,
                            },
                            {
                                "cpt": "80053",
                                "description": "Comprehensive metabolic panel",
                                "units": 1,
                                "charge": 95.00,
                            },
                        ],
                        "total_charges": 300.00,
                        "insurance_paid": 100.00,
                        "patient_responsibility": 0.00,
                        "adjustments": 200.00,
                        "status": "partially_paid",
                        "remittance_date": "2025-01-22",
                    }
                ],
                "balance": {
                    "total_outstanding": 49.00,
                    "insurance_pending": 0.00,
                    "patient_balance": 49.00,
                    "days_in_ar": 16,
                    "aging_bucket": "0-30",
                },
                "notes": [
                    {
                        "date": "2025-01-22",
                        "user": "jsmith",
                        "text": "Need to post a contractual adjustment at transaction ID 1300 for $200 to clear account.",
                    },
                ],
            }
        ]
