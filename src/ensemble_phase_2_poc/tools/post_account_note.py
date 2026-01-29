from typing import List, Dict, Any
from pydantic import BaseModel, Field
from ensemble_phase_2_poc.tools.base_tool import Tool


class PostAccountNoteInput(BaseModel):
    """Input schema for PostAccountNote - description from LLM"""

    description: str = Field(
        description="A detailed description of the actions taken on the account"
    )


class PostAccountNote(Tool):
    """Post a note on the account - injected args are stored as instance attributes"""

    name: str = "post_account_note"
    description: str = Tool.get_tool_description(name)
    args_schema: type = PostAccountNoteInput

    # Injected values (set at instantiation)
    account_number: str = ""
    client_name: str = ""
    facility_prefix: str = ""
    lob: str = ""

    def _run(self, description: str) -> List[Dict[str, Any]]:
        """Post note - uses self.account_number, etc. + description from LLM"""
        self.logger.info(f"Posting account note for account: {self.account_number}")
        self.logger.debug(f"Note content: {description[:100]}..." if len(description) > 100 else f"Note content: {description}")
        return [
            {
                "status": "success",
                "account_number": self.account_number,
                "note": description,
            }
        ]
