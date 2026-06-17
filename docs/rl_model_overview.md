# Python 強化学習モデル概要

このドキュメントは、`sim/` 以下で実装している toy-acai の強化学習モデルの概要です。
現在の Python 学習コードは、C++ シミュレータ `toy_acai_core` を Python から呼び出し、Blue チームを PPO で学習させます。Red チームは固定のルールベース AI です。

主な実装ファイルは次の通りです。

- `sim/toy_acai_rl/env.py`: C++ シミュレータを PPO 用に包む環境ラッパー、観測ベクトル、報酬設計
- `sim/toy_acai_rl/ppo.py`: Actor-Critic モデル、rollout buffer、PPO 更新
- `sim/train_ppo.py`: 学習ループ、評価、checkpoint 保存、ログ出力
- `src/PythonBindings.cpp`: C++ の戦場状態を Python の `dict` / NumPy 配列として返すバインディング

## 全体の流れ

学習は以下のサイクルで進みます。

1. `ToyAcaiPPOEnv.reset()` で C++ シミュレータを初期化する。
2. 学習開始位置をランダム化し、Blue は左側、Red は右側の範囲内で `x` / `y` / `yaw` を毎エピソード変える。
3. C++ の生状態 `fighters` / `missiles` / `hit_events` を、`build_agent_observations()` でニューラルネット入力用の固定長ベクトルに変換する。
4. `PPOTrainer.act()` が学習対象の Blue 機それぞれの行動を出す。
5. 現在のカリキュラム段階で有効な Red 機は `RuleBasedOpponent` が最近傍の生存 Blue 機へ向かって旋回・射撃する。
6. `ToyAcaiPPOEnv.step()` が Blue の行動と Red の行動をまとめて C++ シミュレータへ渡す。
7. 次状態、各 Blue 機の報酬、終了判定を受け取る。
8. 観測、行動、行動確率、報酬、価値推定を `RolloutBuffer` に蓄積する。
9. `rollout_steps` 以上たまったら、GAE で advantage / return を計算し、PPO でネットワークを更新する。

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

## 開始位置ランダム化

学習では、エピソードごとに `ToyAcaiPPOEnv.reset()` が C++ シミュレータを固定配置へ reset した直後、Python 側で active な Blue / Red 機の開始 pose をランダムに上書きします。

Blue は戦場左側、Red は戦場右側の範囲から `x` を選びます。`y` は上下端を避けた範囲を active 機数ぶんのスロットに分け、同じチーム内の機体が開始直後に重なりにくいようにしています。`yaw` は Blue が概ね右向き、Red が概ね左向きになるようにしつつ、小さな jitter を入れます。

## カリキュラム学習

学習は `sim/toy_acai_rl/curriculum.py` の設定に従い、Red の有効機数を段階的に増やすカリキュラムで進みます。
現在のステージは次の 4 段階です。

```text
Red 1 機 -> Red 2 機 -> Red 3 機 -> Red 4 機
```

Blue 側の学習対象は `CURRICULUM_LEARNER_COUNT = 1` で、通常は先頭の Blue 1 機だけを学習します。
C++ シミュレータ上の最大機数は Blue 4 機 + Red 4 機のままですが、`ToyAcaiPPOEnv` に渡す `learner_count` / `opponent_count` で有効機数を絞ります。

ステージ昇格判定は次の通りです。

- `CURRICULUM_EVAL_EVERY = 200` エピソードごとに、現在ステージの Red 機数で決定論的評価を行う。
- 評価は `CURRICULUM_PROMOTION_EVALS = 20` 回実行する。
- `CURRICULUM_PROMOTION_WINS = 14` 勝以上なら、次のステージへ進む。
- 勝率条件を満たさなくても、同じステージで `CURRICULUM_STAGE_MAX_EPISODES = 10000` エピソードに達したら強制的に次へ進む。
- 最終ステージの Red 4 機では、それ以上の昇格は行わない。

昇格すると rollout buffer をクリアし、学習環境を新しい `opponent_count` で作り直します。
評価グラフ用の履歴もステージごとにリセットされます。

ログには `curriculum_stage`、`stage_episode`、`opponent_count` が入り、昇格判定は `curriculum_metrics.jsonl`、昇格イベントは `curriculum_events.jsonl` に出力されます。
checkpoint には現在の `curriculum_stage`、`opponent_count`、`stage_episode` も保存されるため、`--resume-checkpoint` で再開すると同じステージから続きます。

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

形状は `[ミサイル数, 9]` です。

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
| 8 | missileId。C++ 側で発射ごとに採番される一意 ID |

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

学習時の `acceleration` / `turn` の探索ノイズは、毎 step 独立ではなく時間相関を持つようにしています。
各 step で観測を取り直して方策の平均と価値は再計算しますが、正規化ノイズ `z_t` だけを `z_t = rho * z_{t-1} + sqrt(1 - rho^2) * eps_t` で更新し、`raw = mean + std * z_t` として使います。
これにより action repeat は使わず、数フレーム同じ方向のランダムな加速・旋回入力が残りやすくなります。
`rho` は `0.99` 固定です。
episode の先頭ではノイズ状態をリセットし、評価時 (`deterministic=True`) はこの探索ノイズを使いません。

## 観測入力

ニューラルネットへの入力は、`build_agent_observations()` が作る固定長ベクトルです。
学習対象の Blue 機それぞれについて 1 本ずつ作るので、出力形状は `[learner_count, obs_dim]` になります。
現在の学習スクリプト既定では `learner_count=1` です。C++ シミュレータの最大 Blue 機数は 4 のままで、実験用に先頭 1 機だけを有効化しています。

現在の標準構成では 1 機あたりの観測次元は次の計算です。

```text
self features        5
other fighters       7 * 11 = 77
tracked missiles     8 * 7 = 56
--------------------------------
total               138
```

座標や速度は、絶対座標をそのまま入れるのではなく、なるべく自機から見た距離・方位・相対量として表現しています。
これは「自分の前方に敵がいる」「右側からミサイルが来ている」のような判断を、マップ上の絶対位置に依存せず学習しやすくするためです。

### 自機特徴量 5 次元

最初の 5 次元は自機自身の状態です。

| 個数 | 内容 |
| --- | --- |
| 3 | 自機から見た前・左・右方向の境界までの距離 |
| 1 | `speed / MAX_SPEED` |
| 1 | `alive_mask`。生存なら `1`、撃墜済みなら `0` |

境界距離は、前、左、右の 3 方向です。
壁や境界に近い方向を観測に入れることで、戦場外へ出る前に旋回する手がかりになります。

### 他機特徴量 7 機 x 11 次元

自分以外の 7 機について、敵を先、味方を後にし、それぞれ生存機を距離順、撃墜済み機体を後ろに並べて入れます。
各機の特徴量は 11 次元です。

| index | 内容 |
| --- | --- |
| 0 | 距離 / 戦場対角長 |
| 1 | 相手への方位差の cos |
| 2 | 相手への方位差の sin |
| 3 | 相手 yaw と自機 yaw の差の cos |
| 4 | 相手 yaw と自機 yaw の差の sin |
| 5 | 相手速度 / `MAX_SPEED` |
| 6 | 相手 `alive_mask` |
| 7 | 味方なら `1`、敵なら `-1` |
| 8 | closing。接近しているか離れているか |
| 9 | 自機の射撃可能角内にいるなら `1` |
| 10 | 相手の missileCooldown が 0 以下なら `1` |

角度をそのまま入れず `cos` / `sin` にしているのは、`pi` と `-pi` の境界で値が急に飛ぶ問題を避けるためです。
自機前方向・右方向への相対位置成分は、`距離 * cos/sin(方位差)` で復元できるため入れていません。

### 敵ミサイル特徴量 最大 8 発 x 7 次元

敵ミサイルは数が変動するため、近い順に最大 `MAX_TRACKED_MISSILES = 8` 発だけを観測します。
味方ミサイルは観測から除外します。
8 発未満の場合は 0 で埋めます。

各ミサイルの特徴量は 7 次元です。

| index | 内容 |
| --- | --- |
| 0 | 距離 / 戦場対角長 |
| 1 | missile_closing。自分に近づいているか |
| 2 | ミサイル方位差の cos |
| 3 | ミサイル方位差の sin |
| 4 | ミサイル yaw と自機 yaw の差の cos |
| 5 | ミサイル yaw と自機 yaw の差の sin |
| 6 | incoming_alignment。自分へ向いているほど大きい |

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
初期状態でまったく撃たない方策になりにくいよう、`fire_logits.bias` はデフォルト `1.0` に初期化されます。

## 個別方策

学習対象の Blue 機はそれぞれ別々の Actor-Critic モデルを持ちます。

```text
Blue 0 observation -> ActorCritic 0 -> Blue 0 action
...
```

rollout buffer も機体ごとに分かれており、PPO 更新時は各モデルを自分の経験で更新します。
共有方策の切り替えや互換読み込みは持たせていません。

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
| 生存報酬 | 無効 | 実験用に一旦コメントアウト |
| 場外ペナルティ | `-0.03` / step | 戦場外に出ている Blue 機本人へ減点 |
| ミサイル回避報酬 | `+1.0` / 回避 1 発 | 自分を追跡していた敵ミサイルが消滅したフレームに、対象の Blue 本人へ加点 |
| ミサイル発射報酬 | `0.03` | 直前観測で射界内に生存 Red がいる状態で、有効にミサイルを発射できた Blue 機本人へ加点 |
| 撃墜報酬 | `10.0` | Red を撃墜した Blue 機本人へ加点 |
| 自機損失ペナルティ | `-20.0` | 前 step 生存、今 step 非生存になった Blue 本人へ減点 |

終端スコアがチーム勝敗を教え、補助報酬が「場外へ出続けない」「敵ミサイルを振り切る」「敵に向けて撃つ・撃墜する」「自分が撃墜されない」という中間目標を教える構造です。

撃墜報酬 `10.0` と自機損失ペナルティ `-20.0` は、敵撃破をミサイル回避より明確に大きくしつつ、自機損失をさらに重く見ます。

ミサイル発射報酬は、発射前の観測で自機 yaw から `0.85 rad` 以内に生存 Red がいる場合だけ加点します。命中までは要求しないため、敵へ向けて撃つ行動を早めに教えつつ、敵が後方にいる状態で cooldown だけ増えたような発射イベントには加点しません。

ミサイル回避報酬は、毎ステップの減点ではなく「自分を追跡していた敵ミサイル(`targetFighterIndex == 自機 fighter index` かつ `teamId != 学習チーム`)が次の step で消えたフレーム」だけに `+1.0` を与えます。判定の流れは次の通りです。

1. 前ステップのミサイル一覧と現ステップのミサイル一覧を `missileId` で突き合わせる。
2. 前ステップで自分追跡していたミサイルのうち、現ステップで対応が見つからないもの = この step で消滅したものを抽出する。
3. それぞれについて、消滅直前のフレームでの `missile_closing` を計算する。`missile_closing > 0`(まだ近づき続けていた)の場合は、機体を止めていても lifetime 切れで運良く消えただけのケースとみなして報酬を出さない。`missile_closing <= 0`(既に離脱方向に転じていた)場合だけ「ロックを切らせた / 振り切った」とみなして加点する。
4. 自機が現ステップで非生存になった場合は加点しない(別途 `自機損失ペナルティ` が入る)。

この設計は、「危険なミサイルが近づいている間ずっと減点する」連続シグナルの代わりに、「実際に振り切れた瞬間」というイベント単位のシグナルを与える形で、機動を保たせる(止まったまま遠距離からの追跡が lifetime 切れするのを待つ)ような縮退ポリシーが報酬を稼げないようにしています。

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
更新時は各機の buffer で対応する個別モデルを更新します。

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

GIF の各フレームには、Slack で見たときに方策の動きを把握しやすいように、
青チーム各機の critic 値とそのフレーム時点までの累計報酬を `B0 critic=+1.234 reward=+0.789` の形でオーバーレイ表示します。
critic はその step で観測した状態 \(V(s)\) の推定値、reward は `auxiliary_agent_rewards()` と終端スコアを合算した step 増分をエピソード開始から該当フレームまで足し上げたエージェント毎の累計値です。

ログには主に次の指標が出ます。

- `reward`: 1 エピソードの平均報酬和
- `blue_alive` / `red_alive`: 終了時の生存数
- `outcome`: 勝ちなら `1`、負けなら `-1`、時間切れなどは `0`
- `terminal_score`: 終端スコア
- `fire_input_rate`: `fire` を出した割合
- `mean_accel` / `mean_turn` / `mean_abs_turn`: 行動の平均
- `policy_loss` / `value_loss` / `entropy` / `approx_kl` / `clip_fraction`: PPO 更新の統計

### 補助報酬成分のエピソード累計

`auxiliary_agent_rewards()` は step 単位で `evasion_reward` / `kill_reward` / `missile_fire_reward` / `movement_reward` / `out_of_bounds_penalty` / `death_penalty` / `blue_kills` / `blue_losses` / `hit_events` / `survival_reward` を返します。
ログ用の集計はこれらを step 毎に積み上げ、`train_metrics.jsonl` / `eval_metrics.jsonl` ではエピソード合計として記録します。
`mean_movement_distance` だけは step 平均として 1 エピソード内の平均値を出します。
そのため例えば「ミサイル回避報酬がエピソード内で何度発火したか」は `evasion_reward / AUX_EVASION_REWARD` で確認できます。

実装上は `train_ppo.EpisodeInfoAggregator` がこの集計を担当し、`run_episode()` が step ごとに `add(info)` を呼び、ループ終了後に `apply(last_info)` で最終 step 由来の値(`terminal_score` / `outcome` / `blue_alive` / `red_alive`)と合体させています。
したがって最終 step だけに依存するキー(終端スコアや勝敗)はそのまま最後の値を、step 増分のキーはエピソード累計をログに残します。

## checkpoint と再開

`PPOTrainer.save()` は次を checkpoint に保存します。

- 各 agent の model state dict
- agent 数
- checkpoint 形式 `individual_v1`
- PPO config
- episode
- obs_dim
- curriculum_stage / opponent_count / stage_episode

学習を起動するたびに、`--out-dir` (既定 `outputs/rl`) の下に `run_<timestamp>` ディレクトリが新規に作成され、その run 専用の `checkpoints/`、`media/`、`slack/`、`*.jsonl` がそこへ書き込まれます。
これにより、過去の checkpoint や metrics は上書きされません。
加えて、`outputs/rl/latest` の symlink が最新の run ディレクトリへ向け直されます。
Slack uploader は既定で `outputs/rl/run_*/slack` を巡回するため、同時に複数の学習 run を動かしても各 run の Slack 投稿候補を拾えます。

再開時は `--resume-checkpoint` を指定します (例: `outputs/rl/latest/checkpoints/ppo_latest.pt`)。
再開ジョブの出力も新しい `run_<timestamp>` に書き出されるため、再開元の checkpoint やログは壊れません。
観測設計を変えると `obs_dim` が変わるため、古い checkpoint はそのまま読み込めません。
その場合は新規学習が必要です。

## 現在の設計の特徴

- 入力は固定長で、ニューラルネットが扱いやすい。
- 位置関係は自機基準の距離・方位で表すため、操縦判断に直結しやすい。
- 角度は `cos` / `sin` で表すため、角度境界の不連続を避けている。
- ミサイルは近い順に最大 8 発だけ見るため、危険度の高い対象に集中しやすい。
- 学習対象機は個別方策を持つため、複数機に戻した場合は機体ごとの役割分担を学べる余地がある。
- 勝敗を終端スコアで強く教えつつ、補助報酬で中間的な行動改善を促している。

## 注意点

観測特徴量や報酬設計を変えると、学習済み checkpoint との互換性や方策の意味が変わります。
特に観測次元が変わる変更では、既存 checkpoint は読み込めないため新規学習してください。

また、現在の Red は固定ルールです。
学習済み Blue がこの Red には強くなっても、別の Red 方策や人間操作に対して同じ強さを発揮するとは限りません。
self-play や複数タイプの opponent を混ぜる場合は、環境ラッパーと評価方法も合わせて見直す必要があります。
