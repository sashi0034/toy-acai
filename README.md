
仮想環境を起動

```
source <仮想環境のパス>/bin/activate
```

Python バインディングのビルド

```bash
cd linux-python
mkdir -p build
cd build
cmake .. -DSIV3D_ROOT="<siv6 のパス>" -DPython_EXECUTABLE="<Python インタプリタのパス>"
cd ../
cmake --build build --parallel 16
```

学習開始

```
python -m sim.train
```
