# Tools Module

This module provides a set of tools that agents can use to interact with account management systems. Each tool is self-contained with its own schema definition and implementation.

## Architecture

Each tool is implemented as a separate file:

```
tools/
├── __init__.py                     # Centralized exports
├── base_tool.py                    # Implements the tool base class and utility functions.
├── descriptions.yaml               # Tool descriptions. Included as context when supplied to agents.
├── get_account_data.py             # GetAccountData tool + GetAccountDataInput schema
├── ...additional tool implementations
```

### Design Principles

- **Colocated Schemas**: Each tool file contains both the tool class and its corresponding Pydantic input schema definition
- **Centralized tool descriptions**: All tool descriptions, which will be seen as context by agents at inference time, are in `descriptions.yaml`
- **Self-Contained**: Tools are independent and can be imported individually
- **Extensible**: New tools can be added by creating a new file following the established pattern

## Tools

### GetAccountData

**File**: `get_account_data.py`

**Purpose**: Retrieves comprehensive account data including patient information, insurance details, claims history, and account balance.

**Schema**: `GetAccountDataInput` (no parameters - uses injected account context)

**Injected Parameters**:
- `account_number`: The account identifier
- `client_name`: The client/organization name
- `facility_prefix`: Facility identifier prefix
- `lob`: Line of business

**Returns**: List containing account data with nested patient, insurance, claims, balance, and notes information.

### PostContractualAdjustment

**File**: `post_contractual_adjustment.py`

**Purpose**: Posts a contractual adjustment to a specific transaction on an account.

**Schema**: `PostContractualAdjustmentInput`
- `transaction_id` (str): The transaction ID to post the adjustment against

**Injected Parameters**:
- `account_number`: The account identifier
- `client_name`: The client/organization name
- `facility_prefix`: Facility identifier prefix
- `lob`: Line of business

**Returns**: List containing success status, account number, and transaction ID.

### PostAccountNote

**File**: `post_account_note.py`

**Purpose**: Posts a note to the account documenting actions taken or relevant information.

**Schema**: `PostAccountNoteInput`
- `description` (str): A detailed description of the actions taken on the account

**Injected Parameters**:
- `account_number`: The account identifier
- `client_name`: The client/organization name
- `facility_prefix`: Facility identifier prefix
- `lob`: Line of business

**Returns**: List containing success status, account number, and the posted note.

## How to Contribute a New Tool

### Step 1: Create a New Tool File

Create a new Python file in the `tools/` directory following the naming convention: `{tool_name_snake_case}.py`

Example: `dispute_claim.py`

```python
# filepath: ensemble-phase-2-poc/src/ensemble_phase_2_poc/tools/dispute_claim.py
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from langchain.tools import BaseTool
from ensemble_phase_2_poc.tools.utils import get_tool_description


class DisputeClaimInput(BaseModel):
    """Input schema for DisputeClaim"""

    claim_id: str = Field(description="The claim ID to dispute")
    reason: str = Field(description="The reason for the dispute")


class DisputeClaim(BaseTool):
    """Dispute a claim - injected args are stored as instance attributes"""

    name: str = "dispute_claim"
    description: str = get_tool_description(name)
    args_schema: type = DisputeClaimInput

    # Injected values (set at instantiation)
    account_number: str = ""
    client_name: str = ""
    facility_prefix: str = ""
    lob: str = ""

    def _run(self, claim_id: str, reason: str) -> List[Dict[str, Any]]:
        """Dispute claim implementation"""
        return [
            {
                "status": "success",
                "account_number": self.account_number,
                "claim_id": claim_id,
                "reason": reason,
            }
        ]
```

### Step 2: Update `descriptions.yaml`

Add your tool's description to `descriptions.yaml`:

```yaml
dispute_claim: "Dispute a claim on the account"
```

### Step 3: Update `__init__.py`

Add your new tool and its schema to the exports in `__init__.py`:

```python
# ...existing code...
from ensemble_phase_2_poc.tools.dispute_claim import DisputeClaim

__all__ = [
    # ...existing code...
    "DisputeClaim",
]
```

## Usage

Import and use tools directly:

```python
from ensemble_phase_2_poc.tools import DisputeClaim

dispute_claim_tool = DisputeClaim(
    account_number=state["account_number"],
    client_name=state["client_name"],
    facility_prefix=state["facility_prefix"],
    lob=state["lob"],
)

agent = self.build_agent(
    name=self.node_id,
    model="command-a-03-2025",
    tools=[dispute_claim_tool], # supply newly created tool to your agent
)
```