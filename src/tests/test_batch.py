from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from google.genai import types

from gemini_batch_api.batch import get_inlined_responses

RequestFactory = SimpleNamespace


class TestGetInlinedResponses:
    """get_inlined_responsesのテスト"""

    @pytest.fixture
    @classmethod
    def client(cls) -> MagicMock:
        return MagicMock()

    @classmethod
    @patch("gemini_batch_api.batch.time.sleep")
    def test_returns_success(cls, sleep: MagicMock, client: MagicMock) -> None:
        """成功のテスト"""
        # Arrange
        responses = [object(), object()]
        pending_job = RequestFactory(state=types.JobState.JOB_STATE_RUNNING, dest=None)
        succeeded_job = RequestFactory(
            state=types.JobState.JOB_STATE_SUCCEEDED,
            dest=SimpleNamespace(inlined_responses=responses),
        )
        client.batches.get.side_effect = [pending_job, succeeded_job]

        # Act
        result = get_inlined_responses(client, batch_name="batch-1", max_retries=5)

        # Assert
        assert result == responses
        assert client.batches.get.call_count == 2
        client.batches.get.assert_called_with(name="batch-1")
        sleep.assert_called_once_with(30)

    @classmethod
    @patch("gemini_batch_api.batch.time.sleep")
    def test_returns_none_when_job_ends_failed(cls, sleep: MagicMock, client: MagicMock) -> None:
        """失敗のテスト"""
        # Arrange
        failed_job = RequestFactory(state=types.JobState.JOB_STATE_FAILED, dest=None)
        client.batches.get.return_value = failed_job

        # Act
        result = get_inlined_responses(client, batch_name="batch-2")

        # Assert
        assert result is None
        client.batches.get.assert_called_once_with(name="batch-2")
        sleep.assert_not_called()
