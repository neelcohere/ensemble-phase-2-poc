from typing import List, Dict, Any
from pydantic import BaseModel, Field
from ensemble_phase_2_poc.tools.base_tool import Tool


class PostContractualAdjustmentInput(BaseModel):
    """Input schema for PostContractualAdjustment - only transaction_id from LLM"""

    transaction_id: str = Field(
        description="The transaction ID to post the adjustment against"
    )


class PostContractualAdjustment(Tool):
    """Post a contractual adjustment - injected args are stored as instance attributes"""

    name: str = "post_contractual_adjustment"
    description: str = Tool.get_tool_description(name)
    args_schema: type = PostContractualAdjustmentInput

    # Injected values (set at instantiation)
    account_number: str = ""
    client_name: str = ""
    facility_prefix: str = ""
    lob: str = ""

    def _run(self, transaction_id: str) -> List[Dict[str, Any]]:
        """Post adjustment - uses self.account_number, etc. + transaction_id from LLM"""
        self.logger.info(f"Posting contractual adjustment for account: {self.account_number}, transaction: {transaction_id}")
        return [
            {
                "status": "success",
                "account_number": self.account_number,
                "transaction_id": transaction_id,
            }
        ]
