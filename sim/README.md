# Python シミュレーション用メモ

`simulate_random.py` は `toy_acai_core` という C++/nanobind 拡張モジュールを使います。
これは Python の `.py` ファイルではなく、ビルドで生成される共有ライブラリです。
Python バインド版のシミュレーション更新間隔 (1/60 秒) とレンダリング間隔は異なります。
生成される場所はだいたいここです:

```text
linux-python/build/toy_acai_core*.so
```

`./local-scripts/run-sim.sh` が次のエラーで落ちる場合:

```text
ModuleNotFoundError: No module named 'toy_acai_core'
```

原因は `linux-python/build` に `toy_acai_core*.so` が無いことです。

先にこれで Python 拡張をビルドします:

```bash
./linux/setup-apptainer.sh
BUILD_PARALLELISM=1 ./linux-python/build-apptainer.sh
```

その後で実行します:

```bash
./local-scripts/run-sim.sh
```

注意:

- Python モジュールは `linux-python/build-apptainer.sh` で作ります。`linux-cli/build-apptainer.sh` は CLI 実行ファイル専用です。
- ビルドは Apptainer 環境内で行う想定です。ホスト側で直接 CMake すると Boost などの依存バージョンが違って失敗することがあります。
- 実行スクリプトは `linux-python/build` を Python のモジュール探索パスに足して、生成済みの `.so` を import します。
