# Gemini Batch API Utility

`gemini-batch-api` は、GoogleのGemini Batch APIを簡単に利用するためのユーティリティライブラリです。
大量のリクエストを効率的に一括処理できます。

## 特徴

- **簡単なリクエスト定義**: テキスト、ローカルファイル、アップロードファイルの指定が容易です。
- **構造化出力のサポート**: Pydanticモデルを使用して、JSON形式の結果を直接取得できます。
- **一括モニタリング**: バッチジョブの完了を待機し、結果をリストとして取得します。

## 使い方の例

```python
from pathlib import Path
from gemini_batch_api import Request, create_batch, get_inlined_responses

import os

# 1. クライアントを初期化
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

# 2. リクエストのリストを作成
requests = [
    Request(query="富士山について教えてください。"),
    Request(query="画像の内容を説明してください。", file=Path("sample.png")),
]

# 3. バッチジョブを作成
batch_name = create_batch(
    client=client,
    model="gemini-flash-lite-latest",
    requests=requests,
    display_name="my-first-batch"
)
print(f"Batch job created: {batch_name}")

# 4. 結果の取得（完了するまでポーリングし、結果をリストで返します）
responses = get_inlined_responses(client, batch_name)

if responses:
    for res in responses:
        if res.response:
            print(res.response.text)
```

上記を`main.py`に保存して、以下のコマンドで実行できます（結果取得まで時間がかかることがあります）。

```bash
GEMINI_API_KEY="AI..." uv run --with gemini-batch-api main.py
```

### 構造化出力 (Structured Output)

Pydanticモデルを使用して、出力を構造化できます。

```python
from pydantic import BaseModel, Field

class ItemInfo(BaseModel):
    name: str = Field(description="商品名")
    price: int = Field(description="価格")

requests = [
    Request(
        query="このレシートの内容を抽出して",
        file=Path("receipt.jpg"),
        schema=ItemInfo  # Pydanticモデルを指定
    ),
]

# あとは同様に create_batch / get_inlined_responses を実行します
```

### ファイルアップロード

ファイルをアップロードする場合は、`Request.upload`にTrueを指定してください。
