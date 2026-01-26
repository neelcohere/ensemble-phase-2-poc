from pydantic import BaseModel, Field


class GetAccountDataInput(BaseModel):
    """Input schema for GetAccountData - no args from LLM, uses bound values"""


class PostContractualAdjustmentInput(BaseModel):
    """Input schema for PostContractualAdjustment - only transaction_id from LLM"""

    transaction_id: str = Field(
        description="The transaction ID to post the adjustment against"
    )


class PostAccountNoteInput(BaseModel):
    """Input schema for PostAccountNote - description from LLM"""

    description: str = Field(
        description="A detailed description of the actions taken on the account"
    )
