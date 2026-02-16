from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from google.genai import types

from gemini_batch_api.batch import (
    Request,
    create_batch,
    get_inline_request,
    get_inlined_responses,
)


class TestGetInlineRequest:
    """get_inline_requestのテスト"""

    @pytest.fixture
    @classmethod
    def client(cls) -> MagicMock:
        return MagicMock()

    @classmethod
    def test_without_file(cls, client: MagicMock) -> None:
        """file未指定時はテキストのみのインラインリクエストを返すことを確認"""
        # Arrange
        request = Request(query="hello")

        # Act
        result = get_inline_request(client=client, request=request)

        # Assert
        assert len(result.contents.parts) == 1
        assert result.contents.parts[0].text == "hello"
        assert result.config is None
        client.files.upload.assert_not_called()

    @classmethod
    def test_with_file_without_upload(cls, client: MagicMock, tmp_path: Path) -> None:
        """upload=False時はファイルをbytesとしてインラインに含めることを確認"""
        # Arrange
        file_path = tmp_path / "doc.txt"
        file_path.write_text("abc", encoding="utf-8")
        request = Request(query="q", file=file_path, upload=False)

        # Act
        result = get_inline_request(client=client, request=request)

        # Assert
        assert len(result.contents.parts) == 2  # noqa: PLR2004
        assert result.contents.parts[0].text == "q"
        assert result.contents.parts[1].inline_data.data == b"abc"
        assert result.contents.parts[1].inline_data.mime_type == "text/plain"
        client.files.upload.assert_not_called()

    @classmethod
    def test_raises_when_upload_returns_empty_uri(cls, client: MagicMock, tmp_path: Path) -> None:
        """upload=Trueでuriが空の場合はValueErrorを送出することを確認"""
        # Arrange
        file_path = tmp_path / "image.bin"
        file_path.write_bytes(b"\x00\x01")
        request = Request(query="q", file=file_path, upload=True)
        client.files.upload.return_value = SimpleNamespace(uri=None)

        # Act
        with pytest.raises(ValueError, match="Failed to upload file"):
            get_inline_request(client=client, request=request)

        # Assert
        config = types.UploadFileConfig(mime_type="application/octet-stream")
        client.files.upload.assert_called_once_with(file=file_path, config=config)

    @classmethod
    def test_with_schema(cls, client: MagicMock) -> None:
        """schema指定時はJSON構造化出力用のconfigを設定することを確認"""
        # Arrange
        request = Request(query="hello", schema=dict)

        # Act
        result = get_inline_request(client=client, request=request)

        # Assert
        assert result.config is not None
        assert result.config.response_schema is dict
        assert result.config.response_mime_type == "application/json"


class TestCreateBatch:
    """create_batchのテスト"""

    @pytest.fixture
    @classmethod
    def client(cls) -> MagicMock:
        return MagicMock()

    @classmethod
    @patch("gemini_batch_api.batch.get_inline_request")
    def test_without_display_name(cls, get_inline_request_mock: MagicMock, client: MagicMock) -> None:
        """display_name未指定時はconfigなしでバッチ作成することを確認"""
        # Arrange
        requests = [Request(query="q1"), Request(query="q2")]
        inline_requests = [object(), object()]
        get_inline_request_mock.side_effect = inline_requests
        client.batches.create.return_value = SimpleNamespace(name="batch-1")

        # Act
        result = create_batch(client=client, model="gemini-2.0", requests=requests)

        # Assert
        assert result == "batch-1"
        assert get_inline_request_mock.call_count == len(requests)
        get_inline_request_mock.assert_any_call(client, requests[0])
        get_inline_request_mock.assert_any_call(client, requests[1])
        client.batches.create.assert_called_once_with(model="gemini-2.0", src=inline_requests, config=None)

    @classmethod
    @patch("gemini_batch_api.batch.get_inline_request")
    def test_with_display_name(cls, get_inline_request_mock: MagicMock, client: MagicMock) -> None:
        """display_name指定時はCreateBatchJobConfigを付与することを確認"""
        # Arrange
        requests = [Request(query="q")]
        inline_request = object()
        get_inline_request_mock.return_value = inline_request
        client.batches.create.return_value = SimpleNamespace(name="batch-2")

        # Act
        result = create_batch(client=client, model="gemini-2.0", requests=requests, display_name="my-batch")

        # Assert
        assert result == "batch-2"
        get_inline_request_mock.assert_called_once_with(client, requests[0])
        _, kwargs = client.batches.create.call_args
        assert kwargs["model"] == "gemini-2.0"
        assert kwargs["src"] == [inline_request]
        assert isinstance(kwargs["config"], types.CreateBatchJobConfig)
        assert kwargs["config"].display_name == "my-batch"


class TestGetInlinedResponses:
    """get_inlined_responsesのテスト"""

    @pytest.fixture
    @classmethod
    def client(cls) -> MagicMock:
        return MagicMock()

    @classmethod
    @patch("gemini_batch_api.batch.time.sleep")
    def test_returns_success(cls, sleep_mock: MagicMock, client: MagicMock) -> None:
        """成功のテスト"""
        # Arrange
        responses = [object(), object()]
        pending_job = SimpleNamespace(state=types.JobState.JOB_STATE_RUNNING, dest=None)
        succeeded_job = SimpleNamespace(
            state=types.JobState.JOB_STATE_SUCCEEDED,
            dest=SimpleNamespace(inlined_responses=responses),
        )
        client.batches.get.side_effect = [pending_job, succeeded_job]

        # Act
        result = get_inlined_responses(client, batch_name="batch-1", max_retries=5)

        # Assert
        assert result == responses
        assert client.batches.get.call_count == len(responses)
        client.batches.get.assert_called_with(name="batch-1")
        sleep_mock.assert_called_once_with(30)

    @classmethod
    @patch("gemini_batch_api.batch.time.sleep")
    def test_returns_none_when_job_ends_failed(cls, sleep_mock: MagicMock, client: MagicMock) -> None:
        """失敗のテスト"""
        # Arrange
        failed_job = SimpleNamespace(state=types.JobState.JOB_STATE_FAILED, dest=None)
        client.batches.get.return_value = failed_job

        # Act
        result = get_inlined_responses(client, batch_name="batch-2")

        # Assert
        assert result is None
        client.batches.get.assert_called_once_with(name="batch-2")
        sleep_mock.assert_not_called()
