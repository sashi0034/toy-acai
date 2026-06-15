# Python 強化学習モデル概要

このドキュメントは、`sim/` 以下で実装している toy-acai の強化学習モデルの概要です。
現在の Python 学習コードは、C++ シミュレータ `toy_acai_core` を Python から呼び出し、Blue チーム 4 機を PPO で学習させます。Red チームは固定のルールベース AI です。

主な実装ファイルは次の通りです。

- `sim/toy_acai_rl/env.py`: C++ シミュレータを PPO 用に包む環境ラッパー、観測ベクトル、報酬設計
- `sim/toy_acai_rl/ppo.py`: Actor-Critic モデル、rollout buffer、PPO 更新
- `sim/train_ppo.py`: 学習ループ、評価、checkpoint 保存、ログ出力
- `src/PythonBindings.cpp`: C++ の戦場状態を Python の `dict` / NumPy 配列として返すバインディング

## 全体の流れ

学習は以下のサイクルで進みます。

1. `ToyAcaiPPOEnv.reset()` で C++ シミュレータを初期化する。
2. C++ の生状態 `fighters` / `missiles` / `hit_events` を、`build_agent_observations()` でニューラルネット入力用の固定長ベクトルに変換する。
3. `PPOTrainer.act()` が Blue 4 機それぞれの行動を出す。
4. Red 4 機は `RuleBasedOpponent` が最近傍の生存 Blue 機へ向かって旋回・射撃する。
5. `ToyAcaiPPOEnv.step()` が Blue の行動と Red の行動をまとめて C++ シミュレータへ渡す。
6. 次状態、各 Blue 機の報酬、終了判定を受け取る。
7. 観測、行動、行動確率、報酬、価値推定を `RolloutBuffer` に蓄積する。
8. `rollout_steps` 以上たまったら、GAE で advantage / return を計算し、PPO でネットワークを更新する。

## 学習対象と対戦相手

Blue チームが学習対象です。コード上では `TEAM_LEARN = 0` です。
Red チームは `TEAM_RULE = 1` で、現在は学習せず、単純なルールベース AI として動きます。

`RuleBasedOpponent` は各 Red 機について次のように行動します。

- 生存している最近傍の Blue 機を探す。
- スロットル相当の `acceleration` は `0.55` に固定する。
- 目標方向との角度差をもとに `turn` を `[-1, 1]` へクリップして出す。
- 角度差が `0.35 rad` 未満なら `fire = 1.0` にする。
- 目標がいない場合や端に寄った場合は、戦場中心へ戻る向きに旋回する。

つまり現状の学習は「固定ルールの Red に対して、Blue がどう動けば勝てるか」を学ぶ self-play ではない PPO です。

## シミュレータ状態

C++ 側の `BattlefieldEnv` は、各 step 後に Python へ次の情報を返します。

### `fighters`

形状はおおむね `[8, 9]` です。Blue 4 機 + Red 4 機の状態が並びます。

各行の意味は次の通りです。

| index | 意味 |
| --- | --- |
| 0 | `teamId`。Blue は 0、Red は 1 |
| 1 | `memberId`。チーム内 ID |
| 2 | x 座標 |
| 3 | y 座標 |
| 4 | yaw。機体の向き |
| 5 | speed |
| 6 | health。`0` 以下なら撃墜扱い |
| 7 | missileCooldown |
| 8 | outOfBoundsTime。戦場外に出ている時間 |

### `missiles`

形状は `[ミサイル数, 8]` です。

| index | 意味 |
| --- | --- |
| 0 | x 座標 |
| 1 | y 座標 |
| 2 | yaw |
| 3 | speed |
| 4 | age |
| 5 | lockLostTime |
| 6 | teamId。どちらのチームが撃ったか |
| 7 | targetFighterIndex |

### `hit_events`

形状は `[命中イベント数, 4]` です。

| index | 意味 |
| --- | --- |
| 0 | shooterFighterIndex |
| 1 | shooterTeam |
| 2 | targetFighterIndex |
| 3 | targetTeam |

この情報は主に撃墜報酬の計算に使われます。

## 行動空間

ニューラルネットが各 Blue 機ごとに出す行動は 3 要素です。

| index | 名前 | 型 | 環境に渡す値 |
| --- | --- | --- | --- |
| 0 | acceleration | 連続値 | `tanh(raw)` により `[-1, 1]` |
| 1 | turn | 連続値 | `tanh(raw)` により `[-1, 1]` |
| 2 | fire | 離散値 | `0` または `1` |

C++ 側では `acceleration` と `turn` は `[-1, 1]` に clamp され、`fire` は `0.5` 以上なら発射入力として扱われます。

連続行動は正規分布からサンプルし、PPO の log probability 計算には `tanh` 前の `raw` 値を保存します。
環境へ渡すときだけ `tanh` で範囲内に収めます。

## 観測入力

ニューラルネットへの入力は、`build_agent_observations()` が作る固定長ベクトルです。
Blue 4 機それぞれについて 1 本ずつ作るので、出力形状は `[4, obs_dim]` になります。

現在の標準構成では 1 機あたりの観測次元は次の計算です。

```text
self features       13
other fighters       7 * 13 = 91
tracked missiles     8 * 14 = 112
--------------------------------
total               216
```

座標や速度は、絶対座標をそのまま入れるのではなく、なるべく自機から見た相対量として表現しています。
これは「自分の前方に敵がいる」「右側からミサイルが来ている」のような判断を、マップ上の絶対位置に依存せず学習しやすくするためです。

### 自機特徴量 13 次元

最初の 13 次元は自機自身の状態です。

| 個数 | 内容 |
| --- | --- |
| 8 | 自機から見た 8 方向の境界までの距離 |
| 1 | `speed / MAX_SPEED` |
| 1 | `health` |
| 1 | `missileCooldown` |
| 1 | `missileCooldown <= 0` なら 1。射撃可能フラグ |
| 1 | `outOfBoundsTime / OUT_OF_BOUNDS_DEATH_TIME` |

境界距離は、前、右前、右、右後、後、左後、左、左前の 8 方向です。
戦場外へ出ると 3 秒で撃墜されるため、壁や境界に近い方向を観測に入れています。

### 他機特徴量 7 機 x 13 次元

自分以外の 7 機について、敵を先、味方を後に並べて入れます。
各機の特徴量は 13 次元です。

| index | 内容 |
| --- | --- |
| 0 | 自機前方向への相対位置成分 / 戦場対角長 |
| 1 | 自機右方向への相対位置成分 / 戦場対角長 |
| 2 | 距離 / 戦場対角長 |
| 3 | 相手への方位差の cos |
| 4 | 相手への方位差の sin |
| 5 | 相手 yaw と自機 yaw の差の cos |
| 6 | 相手 yaw と自機 yaw の差の sin |
| 7 | 相手速度 / `MAX_SPEED` |
| 8 | 相手 health |
| 9 | 味方なら `1`、敵なら `-1` |
| 10 | closing。接近しているか離れているか |
| 11 | 自機の射撃可能角内にいるなら `1` |
| 12 | 相手の missileCooldown が 0 以下なら `1` |

角度をそのまま入れず `cos` / `sin` にしているのは、`pi` と `-pi` の境界で値が急に飛ぶ問題を避けるためです。

### ミサイル特徴量 最大 8 発 x 14 次元

ミサイルは数が変動するため、近い順に最大 `MAX_TRACKED_MISSILES = 8` 発だけを観測します。
8 発未満の場合は 0 で埋めます。

各ミサイルの特徴量は 14 次元です。

| index | 内容 |
| --- | --- |
| 0 | 自機前方向への相対位置成分 / 戦場対角長 |
| 1 | 自機右方向への相対位置成分 / 戦場対角長 |
| 2 | 距離 / 戦場対角長 |
| 3 | missile_closing。自分に近づいているか |
| 4 | ミサイル方位差の cos |
| 5 | ミサイル方位差の sin |
| 6 | ミサイル yaw と自機 yaw の差の cos |
| 7 | ミサイル yaw と自機 yaw の差の sin |
| 8 | ミサイル速度 / `MAX_SPEED` |
| 9 | ミサイル age / 6.0 |
| 10 | lockLostTime / 1.1 |
| 11 | 味方ミサイルなら `1`、敵ミサイルなら `-1` |
| 12 | 自機を target にしているなら `1` |
| 13 | incoming_alignment。自分へ向いているほど大きい |

近いミサイルを優先しているのは、回避判断では遠くのミサイルより近いミサイルのほうが重要になりやすいからです。

## ニューラルネット構造

モデルは `ActorCritic` です。
方策 actor と価値関数 critic が、観測を処理する backbone を共有します。

デフォルト構成は次の通りです。

```text
input: obs_dim
  -> Linear(obs_dim, hidden_dim)
  -> Tanh
  -> Linear(hidden_dim, hidden_dim)
  -> Tanh

actor continuous head:
  -> Linear(hidden_dim, 2)
  -> acceleration / turn の正規分布平均

actor fire head:
  -> Linear(hidden_dim, 1)
  -> fire の Bernoulli logit

critic head:
  -> Linear(hidden_dim, 1)
  -> 状態価値 V(s)
```

`hidden_dim` のデフォルトは `256` です。
checkpoint 再開時は、保存済み checkpoint の `hidden_dim` を優先します。

連続行動の標準偏差は観測ごとに出すのではなく、`log_std` という学習可能パラメータとして持ちます。
`log_std` は `[-2.5, 0.0]` に clamp されます。

`fire` は Bernoulli 分布です。
初期状態でまったく撃たない方策になりにくいよう、`fire_logits.bias` はデフォルト `0.4` に初期化されます。

## 共有方策と個別方策

デフォルトでは Blue 4 機は同じモデルを共有します。

```text
Blue 0 observation -> shared ActorCritic -> Blue 0 action
Blue 1 observation -> shared ActorCritic -> Blue 1 action
Blue 2 observation -> shared ActorCritic -> Blue 2 action
Blue 3 observation -> shared ActorCritic -> Blue 3 action
```

共有方策では、4 機ぶんの経験をまとめて 1 つのモデルを更新します。
サンプル効率が良く、同じ操縦ルールを複数機の経験から学べます。

`--separate-policies` を指定すると、各 Blue 機が別々の Actor-Critic を持ちます。
役割分担を学べる可能性はありますが、各モデルが受け取る経験量は減るため、学習は難しくなりやすいです。

## 報酬設計

報酬は大きく分けて、エピソード終了時の終端スコアと、毎ステップの補助報酬で構成されます。

### 終端スコア

`terminal_score()` が計算します。
どちらかが全滅した、または最大 step に到達したときだけ加算されます。

Red が全滅した場合は勝利として、次のスコアです。

```text
2.0 + 0.5 * blue_alive_ratio + 0.05 * time_bonus
```

意味は次の通りです。

- 勝利そのものを大きく評価する。
- 味方が多く残っているほど加点する。
- 同じ生存数なら、早く勝ったほうを少しだけ加点する。

Red が残っている場合は非勝利として、次のスコアです。

```text
-2.0 * red_alive_ratio + 0.2 * blue_alive_ratio
```

これは「勝てなかった場合でも、Red をどれだけ減らしたかを強く見る」設計です。
Blue の生存数も少し見ますが、優先度は Red の残数のほうが高いです。

### 毎ステップの補助報酬

終端報酬だけだと、どの行動が良かったのかがエピソード最後まで分かりません。
そこで `auxiliary_agent_rewards()` が毎 step の小さな手がかりを足しています。

| 項目 | デフォルト | 内容 |
| --- | ---: | --- |
| 生存報酬 | `0.0003` / step | 生きている Blue 機に加点 |
| 戦力差報酬 | `0.002 * (blue_alive - red_alive) / team_size` / step | 生存数で優勢なら加点、劣勢なら減点 |
| 移動報酬 | `0.03 * 移動距離 / 戦場対角長` | 動かず停滞する行動を避けるための小さな加点 |
| 撃墜報酬 | `1.0` | Red を撃墜した Blue 機本人へ加点 |
| チーム撃墜報酬 | `0.2` | 味方が Red を撃墜したとき、生存 Blue 全員へ加点 |
| 自機損失ペナルティ | `-1.0` | 前 step 生存、今 step 非生存になった Blue 本人へ減点 |
| チーム損失ペナルティ | `-0.2` | Blue が失われたとき、Blue 全員へ減点 |

終端スコアが勝敗を教え、補助報酬が「生き残る」「撃墜する」「味方を失わない」「停滞しない」という中間目標を教える構造です。

## PPO の学習方法

PPO の実装は `sim/toy_acai_rl/ppo.py` にあります。

### rollout buffer

各 step で次を保存します。

- observation
- PPO 確率計算用の raw action
- old log probability
- reward
- done
- value estimate

複数機を扱うため、`RolloutBuffer` は機体ごとの `AgentRolloutBuffer` を持ちます。
共有方策では、更新時に全機の buffer を結合して 1 つのモデルを更新します。

### GAE

更新前に GAE で advantage を計算します。

```text
delta_t = reward_t + gamma * value_{t+1} * (1 - done_t) - value_t
advantage_t = delta_t + gamma * lambda * (1 - done_t) * advantage_{t+1}
return_t = advantage_t + value_t
```

デフォルトは `gamma = 0.995`、`gae_lambda = 0.95` です。
計算した advantage は平均 0、標準偏差 1 に正規化してから使います。

### PPO 更新

保存しておいた古い方策の log probability と、現在の方策で同じ行動を出す log probability を比較します。

```text
ratio = exp(new_log_prob - old_log_prob)
```

方策 loss は clipped objective です。

```text
unclipped = ratio * advantage
clipped = clip(ratio, 1 - clip, 1 + clip) * advantage
policy_loss = -mean(min(unclipped, clipped))
```

critic は return に value を近づけます。

```text
value_loss = 0.5 * mean((return - value)^2)
```

最終的な loss は次の形です。

```text
loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
```

デフォルト値は次の通りです。

| パラメータ | デフォルト |
| --- | ---: |
| `clip` | `0.2` |
| `lr` | `3e-4` |
| `update_epochs` | `4` |
| `batch_size` | `128` in `train_ppo.py` |
| `rollout_steps` | `512` in `train_ppo.py` |
| `value_coef` | `0.5` |
| `entropy_coef` | `0.003` |
| `max_grad_norm` | `0.5` |

`entropy` は探索を促すための項です。
これがあることで、学習の早い段階で「同じ行動ばかり選ぶ」状態に寄りすぎるのを抑えます。

## 評価方法

学習中の評価は `evaluate()` が行います。
評価時は `deterministic=True` にして、連続行動は分布からサンプルせず平均値を使います。
`fire` は `sigmoid(logit) >= eval_fire_threshold` なら発射します。
デフォルトの `eval_fire_threshold` は `0.15` です。

評価では GIF を出力できます。
学習時の探索ノイズを切ることで、その時点の方策が安定してどれくらい勝てるかを見ます。

ログには主に次の指標が出ます。

- `reward`: 1 エピソードの平均報酬和
- `blue_alive` / `red_alive`: 終了時の生存数
- `outcome`: 勝ちなら `1`、負けなら `-1`、時間切れなどは `0`
- `terminal_score`: 終端スコア
- `fire_input_rate`: `fire` を出した割合
- `mean_accel` / `mean_turn` / `mean_abs_turn`: 行動の平均
- `policy_loss` / `value_loss` / `entropy` / `approx_kl` / `clip_fraction`: PPO 更新の統計

## checkpoint と再開

`PPOTrainer.save()` は次を checkpoint に保存します。

- 各 agent の model state dict
- agent 数
- 共有方策かどうか
- PPO config
- episode
- obs_dim

再開時は `--resume-checkpoint` を指定します。
観測設計を変えると `obs_dim` が変わるため、古い checkpoint はそのまま読み込めません。
その場合は新規学習が必要です。

共有方策で、個別方策 checkpoint を読み込む場合は、保存されている各機の重みを平均して 1 つの共有モデルへ読み込みます。

## 現在の設計の特徴

- 入力は固定長で、ニューラルネットが扱いやすい。
- 位置関係は自機基準の前後左右成分で表すため、操縦判断に直結しやすい。
- 角度は `cos` / `sin` で表すため、角度境界の不連続を避けている。
- ミサイルは近い順に最大 8 発だけ見るため、危険度の高い対象に集中しやすい。
- Blue 4 機はデフォルトで方策を共有するため、経験を効率よく使える。
- 勝敗を終端スコアで強く教えつつ、補助報酬で中間的な行動改善を促している。

## 注意点

観測特徴量や報酬設計を変えると、学習済み checkpoint との互換性や方策の意味が変わります。
特に観測次元が変わる変更では、既存 checkpoint は読み込めないため新規学習してください。

また、現在の Red は固定ルールです。
学習済み Blue がこの Red には強くなっても、別の Red 方策や人間操作に対して同じ強さを発揮するとは限りません。
self-play や複数タイプの opponent を混ぜる場合は、環境ラッパーと評価方法も合わせて見直す必要があります。
