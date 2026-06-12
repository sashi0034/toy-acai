# Python シミュレーション用メモ

`simulate_random.py` は `toy_acai_core` という C++/nanobind 拡張モジュールを使います。
これは Python の `.py` ファイルではなく、ビルドで生成される共有ライブラリです。
Python バインド版のシミュレーション更新間隔 (1/60 秒) とレンダリング間隔は異なります。
生成される場所はだいたいここです:

```text
linux-python/build/toy_acai_core*.so
```

`./local-scripts/sim-random.sh` が次のエラーで落ちる場合:

```text
ModuleNotFoundError: No module named 'toy_acai_core'
```

原因は `linux-python/build` に `toy_acai_core*.so` が無いことです。

先にこれで Python 拡張をビルドします:

```bash
./linux/setup-apptainer.sh
BUILD_PARALLELISM=1 ./linux-python/build-apptainer.sh
```

その後でランダムシミュレーションを実行します:

```bash
./local-scripts/sim-random.sh
```

PPO 学習を実行する場合は、まず PyTorch 入りの Apptainer image を作り直してから Python 拡張をビルドします:

```bash
./linux/setup-apptainer.sh
BUILD_PARALLELISM=1 ./linux-python/build-apptainer.sh
```

学習は次のように実行できます:

```bash
# smoke run の場合、以下のような設定が必要です
# TOY_ACAI_EPISODES=2 TOY_ACAI_STEPS=20 TOY_ACAI_RENDER_EVERY=1 \
# TOY_ACAI_ROLLOUT_STEPS=8 TOY_ACAI_BATCH_SIZE=8 \

./local-scripts/train-ppo.sh
```

Slack 投稿が多すぎる場合は、まず `TOY_ACAI_RENDER_EVERY` を大きくしてください。
これは「何エピソードごとに GIF を作って Slack 送信用にスプールするか」を決めます。
例えば `TOY_ACAI_RENDER_EVERY=100` なら 100 エピソードごとに投稿候補が作られます。

`TOY_ACAI_SLACK_POLL_SECONDS` や `./local-scripts/slack-uploader.sh --poll-seconds 300` は uploader が pending を見に行く間隔です。
生成済み GIF の数自体を減らしたい場合は `TOY_ACAI_RENDER_EVERY` を変更してください。

学習中に作られた GIF は `outputs/rl/default/slack/pending/*.json` として Slack 送信用にスプールされます。
Slack の設定はリポジトリ直下の `.env` に置けます。まず `.env.example` をコピーして、ログインノードで実際の値を入れてください:

```bash
cp .env.example .env
$EDITOR .env
```

Slack app の Bot Token Scopes には `files:write` が必要です。
また、Bot user を `SLACK_CHANNEL_ID` のチャンネルに参加させておいてください。

計算ノードからはネット通信せず、ログインノードで次を起動してください:

```bash
./local-scripts/slack-uploader.sh
```

注意:

- Python モジュールは `linux-python/build-apptainer.sh` で作ります。`linux-cli/build-apptainer.sh` は CLI 実行ファイル専用です。
- ビルドは Apptainer 環境内で行う想定です。ホスト側で直接 CMake すると Boost などの依存バージョンが違って失敗することがあります。
- 実行スクリプトは `linux-python/build` を Python のモジュール探索パスに足して、生成済みの `.so` を import します。
