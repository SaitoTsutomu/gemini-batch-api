# ruff: noqa: T201
from datetime import datetime
from pathlib import Path

import fire
from pydantic import BaseModel, Field

from gemini_batch_api import Request, create_batch, get_inlined_responses


class FileInfo(BaseModel):
    """ファイル情報"""

    text: str = Field(description="テキスト")


def main(*, upload: bool = False, structured: bool = False) -> None:
    """画像からテキスト抽出"""
    request = Request(
        query="文字抽出",
        file=Path(__file__).parent / "sample.png",
        upload=upload,
        schema=FileInfo if structured else None,
    )
    print(f"開始 {datetime.now():%H:%M:%S}")
    batch_name = create_batch("gemini-flash-lite-latest", [request])
    inlined_responses = get_inlined_responses(batch_name)
    if inlined_responses:
        for response in inlined_responses:
            if response.response:
                print(response.response.text)
                metadata = response.response.usage_metadata
                if metadata:
                    print("入力トークン数", metadata.prompt_token_count)
                    print("出力トークン数", metadata.candidates_token_count)
    print(f"終了 {datetime.now():%H:%M:%S}")


if __name__ == "__main__":
    fire.Fire(main)
