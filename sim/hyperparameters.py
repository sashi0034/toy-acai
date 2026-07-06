# hyperparameters.py

# 隠れ層の次元数
HIDDEN_DIM = 128

# 学習率
LEARNING_RATE = 3e-4

# 学習全体で実行する更新回数
NUM_UPDATES = 5000

# 1 回のパラメータ更新に使うエピソード数
EPISODES_PER_UPDATE = 16

# 教師データを用いた更新の頻度
TEACHER_UPDATE_INTERVAL = 5

# ステップごとの報酬割引率
REWARD_DISCOUNT = 0.99

# GAE のトレース減衰率 λ
GAE_LAMBDA = 0.95

# 最大シミュレーション時間 (秒)
MAX_SIMULATION_SECONDS = 15.0

# 観測特徴量の距離を正規化するための係数
DISTANCE_NORMALIZATION_FACTOR = 1.0 / 500.0

# 観測特徴量の速度を正規化するための係数
SPEED_NORMALIZATION_FACTOR = 1.0 / 500.0
