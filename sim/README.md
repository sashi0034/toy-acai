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
# TOY_ACAI_EPISODES=2 TOY_ACAI_STEPS=20 TOY_ACAI_RENDER_EVERY=0 \
# TOY_ACAI_ROLLOUT_STEPS=8 TOY_ACAI_BATCH_SIZE=8 \

./local-scripts/train-ppo.sh
```

学習を実行するたびに、`TOY_ACAI_OUTPUT_DIR` (既定では `outputs/rl`) の下に `run_<timestamp>` ディレクトリが新規作成されます。
checkpoint や metrics、GIF はその run 専用ディレクトリに書き込まれるため、過去の結果が上書きされることはありません。
最新の run ディレクトリには `outputs/rl/latest` という symlink が貼られます。
Slack uploader は既定で `outputs/rl/run_*/slack` を巡回するため、複数の学習 run を同時に動かしても各 run の投稿候補を拾えます。

チェックポイントから続ける場合は、例えば次のようにします:

```bash
TOY_ACAI_RESUME_CHECKPOINT=outputs/rl/latest/checkpoints/ppo_002000.pt \
./local-scripts/train-ppo.sh
```

再開時も新しい run ディレクトリへ書き出すため、元の run の checkpoint は保持されます。

観測特徴量や報酬設計を変えた後は、新規学習してください。
現在の学習はエピソード終端の `red_alive`、`blue_alive`、終了ステップから計算する勝敗スコアを主報酬にします。
加えて各 Blue 機体に、場外に出ている間の本人ペナルティ、敵ミサイルが自機へ向いている間の距離依存ペナルティ、最近傍 Red へ向いている間の本人補助報酬、最近傍 Red から向かれている間の同量ペナルティ、直近 1 秒の移動距離が 100px 以下の本人ペナルティ、直前観測で射界内に生存 Red がいる状態で有効にミサイルを発射したときの本人補助報酬、自機のミサイルで Red 機体を撃墜したときの本人補助報酬、自機が撃墜されたときの本人ペナルティを与えます。
実験用に、学習対象の Blue 機数は `TOY_ACAI_LEARNER_COUNT` で変更できます。現在の既定は 1 機です。敵 Red は 4 機のままです。
`outputs/rl/latest/checkpoints/ppo_latest.pt` は checkpoint 保存ごとにも更新されるため、途中終了後も直近の保存済み方策を参照できます。

Slack 投稿が多すぎる場合は、まず `TOY_ACAI_RENDER_EVERY` を大きくしてください。
これは「何エピソードごとに GIF を作って Slack 送信用にスプールするか」を決めます。
例えば `TOY_ACAI_RENDER_EVERY=100` なら 100 エピソードごとに投稿候補が作られます。

`TOY_ACAI_ROLLOUT_STEPS` は小さめにすると PPO の更新頻度が上がります。

生成済み GIF の数自体を減らしたい場合は `TOY_ACAI_RENDER_EVERY` を変更してください。

学習中に作られた GIF は各 run の `slack/pending/*.json` として Slack 送信用にスプールされます。
学習開始時に uploader が run ごとに `docs/rl_model_overview.md` を添付した Slack 親メッセージを投稿し、以降の GIF 投稿はその run のスレッドにまとまります。
GIF 投稿 10 件ごとに、横軸 episode、縦軸 reward の推移 PNG も同じスレッドへ投稿されます。
Slack の設定はリポジトリ直下の `.env` に置けます。まず `.env.example` をコピーして、ログインノードで実際の値を入れてください:

```bash
cp .env.example .env
$EDITOR .env
```

Slack app の Bot Token Scopes には `files:write`、`files:read`、`chat:write` が必要です。
また、Bot user を `SLACK_CHANNEL_ID` のチャンネルに参加させておいてください。

計算ノードからはネット通信せず、ログインノードで次を起動してください:

```bash
./local-scripts/slack-uploader.sh
```

このスクリプトは `TOY_ACAI_OUTPUT_DIR` (既定 `outputs/rl`) の下にある `run_*/slack` を全て監視します。
特定の run だけを投稿したい場合は `sim/slack_uploader.py --spool outputs/rl/run_<timestamp>/slack` を指定してください。

注意:

- Python モジュールは `linux-python/build-apptainer.sh` で作ります。`linux-cli/build-apptainer.sh` は CLI 実行ファイル専用です。
- ビルドは Apptainer 環境内で行う想定です。ホスト側で直接 CMake すると Boost などの依存バージョンが違って失敗することがあります。
- 実行スクリプトは `linux-python/build` を Python のモジュール探索パスに足して、生成済みの `.so` を import します。
