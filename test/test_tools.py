"""Tests for ensemble_phase_2_poc.tools module."""

import pytest
from unittest.mock import patch, mock_open
from ensemble_phase_2_poc.tools import GetAccountData, PostAccountNote, PostContractualAdjustment
from ensemble_phase_2_poc.tools.base_tool import Tool
from ensemble_phase_2_poc.tools.get_account_data import GetAccountDataInput
from ensemble_phase_2_poc.tools.post_account_note import PostAccountNoteInput
from ensemble_phase_2_poc.tools.post_contractual_adjustment import PostContractualAdjustmentInput


MOCK_DESCRIPTIONS_YAML = """
get_account_data: "Retrieve account data for an account"
post_contractual_adjustment: "Post an adjustment"
post_account_note: "Post a note on the account summarizing all actions taken"
"""


class TestToolBase:
    """Test base Tool class."""

    @patch("builtins.open", mock_open(read_data=MOCK_DESCRIPTIONS_YAML))
    def test_get_tool_description_returns_description(self):
        """get_tool_description returns the correct description for a known tool."""
        desc = Tool.get_tool_description("get_account_data")
        assert "Retrieve account data for an account" in desc

    @patch("builtins.open", mock_open(read_data=MOCK_DESCRIPTIONS_YAML))
    def test_get_tool_description_raises_for_unknown(self):
        """get_tool_description raises ValueError for tools not registered in a descriptions file"""
        with pytest.raises(ValueError, match="not found, please include the tool description in"):
            Tool.get_tool_description("unknown_tool")


class TestGetAccountData:
    """Test GetAccountData tool."""

    def test_instantiation_injected_params(self):
        """GetAccountData accepts injected params when instantiated"""
        tool = GetAccountData(
            account_number="ACC-123",
            client_name="Acme",
            facility_prefix="FAC",
            lob="Acute",
        )
        assert tool.account_number == "ACC-123"
        assert tool.client_name == "Acme"
        assert tool.facility_prefix == "FAC"
        assert tool.lob == "Acute"


    # NOTE: the current implementation of GetAccountData.run assumes that there's always data to return.
    # For now, given that the tool is mocked, it's ok, but this test will need to be refactored in the future
    def test_run_returns_list_of_dicts(self):
        """tool._run should return a list of account data dicts."""
        tool = GetAccountData(
            account_number="ACC-123",
            client_name="Acme",
            facility_prefix="FAC",
            lob="Acute",
        )
        result = tool._run()
        assert isinstance(result, list)
        assert isinstance(result[0], dict)
        assert "account_number" in result[0]
        assert "patient" in result[0]
        assert "claims" in result[0]

    # NOTE: the current implementation of GetAccountData.run returns the same packet regardless of
    # account_number, client_name, etc. Future implementation should assert that account_number,
    # client_name, etc. match. Given that _run is not dynamic rn, we only test for expected data format
    def test_run_output_structure(self):
        """_run output has expected top-level keys."""
        tool = GetAccountData(
            account_number="ACC-12345",
            client_name="Acme Healthcare",
            facility_prefix="FAC",
            lob="Commercial",
        )
        result = tool._run()
        data = result[0]
        assert "balance" in data
        assert "notes" in data
        assert "insurance" in data
        assert "patient" in data
        assert "claims" in data
        assert "account_number" in data
        assert "client_name" in data
        assert "facility_prefix" in data
        assert "lob" in data


class TestPostAccountNote:
    """Test PostAccountNote tool."""

    def test_instantiation(self):
        """PostAccountNote accepts injected params at instantiation"""
        tool = PostAccountNote(
            account_number="ACC-456",
            client_name="Client",
            facility_prefix="FAC",
            lob="Acute",
        )
        assert tool.account_number == "ACC-456"
        assert tool.facility_prefix == "FAC"
        assert tool.client_name == "Client"
        assert tool.lob == "Acute"

    def test_run_returns_success_structure(self):
        """
        PostAccountNote._run should:
        * set the note field to provided description
        * set status to "success"
        * set account_number to the injected account number
        """
        tool = PostAccountNote(
            account_number="ACC-456",
            client_name="Client",
            facility_prefix="FAC",
            lob="Acute",
        )
        result = tool._run(description="Resolved billing issue.")
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["status"] == "success"
        assert result[0]["account_number"] == "ACC-456"
        assert result[0]["note"] == "Resolved billing issue."


class TestPostContractualAdjustment:
    """Test PostContractualAdjustment tool."""

    def test_instantiation(self):
        """PostContractualAdjustment accepts injected params."""
        tool = PostContractualAdjustment(
            account_number="ACC-789",
            client_name="Client",
            facility_prefix="FAC",
            lob="Acute",
        )
        assert tool.account_number == "ACC-789"
        assert tool.client_name == "Client"
        assert tool.facility_prefix == "FAC"
        assert tool.lob == "Acute"

    def test_run_returns_success_structure(self):
        """
        PostContractualAdjustment._run should:
        * set the transaction_id field to the provided id
        * set the account_number to the injected account_number
        * set status to "success"
        """
        tool = PostContractualAdjustment(
            account_number="ACC-789",
            client_name="Client",
            facility_prefix="FAC",
            lob="Acute",
        )
        result = tool._run(transaction_id="TXN-100")
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["status"] == "success"
        assert result[0]["account_number"] == "ACC-789"
        assert result[0]["transaction_id"] == "TXN-100"