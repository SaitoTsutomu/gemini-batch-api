import mimetypes
import os
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
    """インラインリクエストを取得"""
    parts = [types.Part.from_text(text=request.query)]
    if request.file:
        mime_type = mimetypes.guess_type(request.file)[0] or ""
        if request.upload:
            uploaded_file = client.files.upload(file=request.file)
            parts.append(
                types.Part.from_uri(file_uri=uploaded_file.uri or "", mime_type=mime_type),
            )
        else:
            data = Path(request.file).read_bytes()
            parts.append(
                types.Part.from_bytes(data=data, mime_type=mime_type),
            )
    contents = types.Content(role="user", parts=parts)
    if not request.schema:
        return types.InlinedRequest(contents=contents)
    return types.InlinedRequest(
        contents=contents,
        config=types.GenerateContentConfig(
            responseSchema=request.schema,  # type: ignore[unexpected-keyword]
            responseMimeType="application/json",  # type: ignore[unexpected-keyword]
        ),
    )


def create_batch(model: str, requests: list[Request], display_name: str = "", api_key: str = "") -> str:
    """バッチ作成

    :param model: モデル名
    :param requests: インラインリクエスト
    :param display_name: 表示名
    :param api_key: APIキー
    :return: バッチ名
    """
    api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
    client = genai.Client(api_key=api_key)
    inline_requests = [get_inline_request(client, request) for request in requests]
    config = types.CreateBatchJobConfig(display_name=display_name) if display_name else None
    batch = client.batches.create(model=model, src=inline_requests, config=config)
    return batch.name or ""


def get_inlined_responses(batch_name: str, api_key: str = "") -> list[types.InlinedResponse] | None:
    """インラインリクエストのレスポンスを取得

    :param batch_name: バッチ名
    :param api_key: APIキー
    :return: レスポンス
    """
    api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
    client = genai.Client(api_key=api_key)
    while True:
        job = client.batches.get(name=batch_name)
        if job.state in types.JOB_STATES_ENDED_VERTEX:
            break
        time.sleep(30)
    if job.state != types.JobState.JOB_STATE_SUCCEEDED or not job.dest:
        return None
    return job.dest.inlined_responses
