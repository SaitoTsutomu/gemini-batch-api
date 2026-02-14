import mimetypes
import time
from dataclasses import dataclass
from pathlib import Path

from google import genai
from google.genai import types


@dataclass
class Request:
    """リクエスト"""

    query: str  # クエリ
    file: Path | None = None  # ファイル
    upload: bool = False  # アップロードするかどうか
    schema: type | None = None  # 構造化出力


def get_inline_request(client: genai.Client, request: Request) -> types.InlinedRequest:
    """インラインリクエストを取得

    :param client: クライアント
    :param request: Requestオブジェクト
    :return: InlinedRequestオブジェクト
    """
    parts = [types.Part.from_text(text=request.query)]
    if request.file:
        mime_type = mimetypes.guess_type(request.file.name)[0] or "application/octet-stream"
        if request.upload:
            uploaded_file = client.files.upload(file=request.file)
            if not uploaded_file.uri:
                msg = "Failed to upload file"
                raise ValueError(msg)
            parts.append(
                types.Part.from_uri(file_uri=uploaded_file.uri, mime_type=mime_type),
            )
        else:
            data = request.file.read_bytes()
            parts.append(
                types.Part.from_bytes(data=data, mime_type=mime_type),
            )
    contents = types.Content(role="user", parts=parts)
    if not request.schema:
        return types.InlinedRequest(contents=contents)
    return types.InlinedRequest(
        contents=contents,
        config=types.GenerateContentConfig(
            response_schema=request.schema,
            response_mime_type="application/json",
        ),
    )


def create_batch(client: genai.Client, model: str, requests: list[Request], display_name: str = "") -> str | None:
    """バッチ作成

    :param client: クライアント
    :param model: モデル名
    :param requests: インラインリクエスト
    :param display_name: 表示名
    :return: バッチ名
    """
    inline_requests = [get_inline_request(client, request) for request in requests]
    config = types.CreateBatchJobConfig(display_name=display_name) if display_name else None
    batch = client.batches.create(model=model, src=inline_requests, config=config)
    return batch.name


def get_inlined_responses(
    client: genai.Client, batch_name: str, max_retries: int = 5760
) -> list[types.InlinedResponse] | None:
    """インラインリクエストのレスポンスを取得

    :param client: クライアント
    :param batch_name: バッチ名
    :param max_retries: 最大リトライ回数
    :return: レスポンス
    """
    for _ in range(max_retries):
        job = client.batches.get(name=batch_name)
        if job.state in types.JOB_STATES_ENDED_VERTEX:
            break
        time.sleep(30)
    if job.state != types.JobState.JOB_STATE_SUCCEEDED or not job.dest:
        return None
    return job.dest.inlined_responses
