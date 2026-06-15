# config_2d.py - 2D项目全局配置文件
import numpy as np

# 数据控制变量
shuju = 0  # 0: 使用默认数据, 1: 使用读取数据

# 网络结构参数
NETWORK_DEPTH = 8  # 网络深度（隐藏层数量，从10层改回8层）
HIDDEN_SIZE = 64  # 隐藏层神经元数量（保持64个不变）
ACTIVATION = 'tanh'  # 激活函数类型

# 计算域几何参数（2D - m 单位，统一为国际单位制）
# 基于实际数据范围：x=5025-12975mm，y=2025-5975mm
PLANE_SIZE_X = 12.975   # m，x方向平面尺寸（5025-12975mm -> 5.025-12.975m）
PLANE_SIZE_Y = 5.975    # m，y方向平面尺寸（2025-5975mm -> 2.025-5.975m）
PLANE_SIZE_Z = 0.01   # m，固定z=0.01m平面（近似2D）

# 训练参数
LEARNING_RATE = 0.0001  # 学习率（从1e-5改为1e-4）
EPOCHS = 10000  # 训练轮数
BATCH_SIZE = 1024  # 批次大小（物理损失和数据损失统一使用1024）
REYNOLDS_NUMBER = 100  # 雷诺数（改为100）

# 几何参数
THICKNESS = 0.01  # m，厚度（2D问题中用于边界条件）

# 物理约束点参数
N_BOUNDARY_POINTS = 0  # 边界点数量（已废弃，使用统一点生成）
# 基于singletemporal_data数据：每个文件14575个点，81个时间点（0-8s，步长0.1s）
# 总时空域点数：14575 × 81 = 1,180,575个点
N_INTERNAL_POINTS = 23612  # 物理采样点数量

# 物理方程权重
W_CONTINUITY = 1.0     # 连续性方程权重
W_MOMENTUM_X = 1.0    # x方向动量方程权重
W_MOMENTUM_Y = 1.0    # y方向动量方程权重
W_MOMENTUM_Z = 1.0    # z方向动量方程权重
W_VORTICITY = 0.0     # 涡量方程权重（新增，不计算二阶导数版本）

# 边界条件权重参数
W_DATA = 10.0    # 数据损失权重（从1.0修改为100.0）
W_PHYS = 1.0    # 物理损失权重
W_BND = 0.0     # 边界损失权重（禁用边界约束）
W_INLET_BND = 0.0    # 进气边损失权重（x=0入口边界）- 设置为0禁用边界约束
W_OTHER_BND = 0.0     # 其他边界损失权重（出口边界、壁面边界）- 设置为0禁用边界约束
W_REAL_DATA = 100.0    # 真实数据损失权重 - 修改为100.0

# 初始数据约束参数（针对只有初始时刻数据的情况）
W_INITIAL_SPATIAL = 1.0  # 初始空间数据约束权重 - 降低到1.0以解决Spatial损失降不下来的问题
INITIAL_CONSTRAINT_DURATION = 0.1  # 初始约束作用时间长度
NUM_INITIAL_STEPS = 5  # 初始约束时间步数
INITIAL_DECAY_TIME = 0.05  # 初始约束衰减时间常数
SPATIAL_INFLUENCE_RADIUS = 0.05  # 空间影响半径(mm) - 进一步缩小到0.05mm，约覆盖0.2-0.5个邻近点，提高空间精度
USE_SPATIAL_INTERPOLATION = True  # 是否使用空间插值
INITIAL_DATA_WEIGHT_SCHEDULE = 'exponential'  # 权重调度：'exponential', 'linear', 'constant'

# 真实数据文件路径
REAL_DATA_FILE = 'data2/sigle_t=0.00s.txt'  # 真实实验数据文件（使用data2文件夹，从0秒开始）

# 绘图和可视化参数
PLOT_GRID_SIZE = 200  # 绘图网格大小
PLOT_DPI = 300  # 图像分辨率
PLOT_FORMAT = 'png'  # 图像格式
SAVE_PATH = 'out_fields'  # 图像保存路径
FIGURE_SIZE = (10, 6)  # 图像尺寸 (宽, 高)
DPI = 300  # 图像分辨率
FONT_SIZE = 12  # 字体大小
COLOR_MAP = 'viridis'  # 颜色映射
VECTOR_SCALE = 1.0  # 矢量缩放因子
SAVE_PLOTS = True  # 是否保存图像
PLOT_INTERVAL = 1000  # 绘图间隔（轮数）- 改为1000轮绘制一次

# 可视化动画参数
ANIMATION_TIME_RANGE = (0.0, 5.0)  # 动画时间范围（0-5秒）
ANIMATION_NUM_FRAMES = 51  # 动画帧数（51帧对应0-5秒，每0.1秒一帧）
ANIMATION_GRID_SIZE = 100  # 动画网格大小
VELOCITY_DIFF_TIMES = [0.0, 1.0, 2.0, 4.0]  # 速度差热图时间点列表
PLOT_INTERVAL_ANIMATION = 1000  # 动画生成间隔（轮数）- 改为1000轮绘制一次
PLOT_INTERVAL_VELOCITY_DIFF = 1000  # 速度差热图绘制间隔（轮数）- 改为1000轮绘制一次

# 时间参数（瞬态模拟）
TIME_START = 0.0  # 开始时间（0.0s，基于data2数据范围0-5s）
TIME_END = 5.0   # 结束时间（7.0s，基于gemini.md指令要求）
TIME_STEP = 0.01  # 时间步长（0.01秒，共701步）
N_TIME_STEPS = 501  # 时间步骤数量（701步，0-7秒，每0.01秒一步）

# 时间分段参数（t1:真实数据范围，t2:Mamba约束范围）
T1_START = 0.0   # 真实数据开始时间（t1范围：0-2.5s）
T1_END = 2.5    # 真实数据结束时间（t1范围：0-2.5s）
T2_START = 5.0  # Mamba约束开始时间（t2范围：2.5-5s）
T2_END = 5.0    # Mamba约束结束时间（t2范围：2.5-5s）

# 训练阶段配置
N_EPOCHS = 30000  # Adam优化器训练轮数（修改为30000轮）

# Mamba时序扩展参数（与总轮数参数绑定）
TEMPORAL_EXTENSION = {
    'enable': True,  # 启用时序扩展
    'stages': 1,  # 单阶段训练
    'stage_times': [15.0],  # 单阶段结束时间（5.0-15.0s训练范围）
    'stage_epochs': [N_EPOCHS],  # 与Adam训练轮数保持一致
    'temporal_window_size': 50,  # 时序窗口大小
    'consistency_weight': 0.1,  # 时序一致性损失权重
    'overlap_ratio': 0.5,  # 窗口重叠比例
}

# 文件路径参数
GMSH_POINTS_FILE = 'gmsh_mesh_points.npy'  # GMSH网格点文件
PIV_DATA_FOLDER = 'pivresult'  # PIV数据文件夹
DEFAULT_DATA_FILE = 'data/real_jet_data.json'  # 默认数据文件

# 数值计算参数
TOLERANCE = 1e-6  # 数值容差
MAX_ITERATIONS = 1000  # 最大迭代次数

# 输出控制参数
VERBOSE = True  # 是否打印详细信息
SAVE_INTERVAL = 100  # 保存间隔（轮数）
PLOT_INTERVAL = 1000  # 绘图间隔（轮数）- 每1000轮生成一次绘图

# 训练收敛监控参数
CONVERGENCE_MONITOR = {
    'enable': True,  # 启用收敛监控
    'patience': 500,  # 耐心值：损失多少轮无改善认为收敛
    'min_delta': 1e-6,  # 最小改善阈值
    'check_frequency': 100,  # 检查频率（轮数）
    'window_size': 10,  # 平滑窗口大小
    'convergence_threshold': 1e-5,  # 收敛阈值
}

# 自适应损失权重参数
UPDATE_FREQUENCY = 100  # 权重更新频率（轮数）
PATIENCE = 50  # 早停耐心值
MIN_DELTA = 1e-6  # 最小变化阈值

# 报告频率
REPORT_EVERY = 100  # 每多少轮报告一次训练进度
HISTORY_EVERY = 100  # 每多少轮记录一次历史数据



# 设备参数
DEVICE = 'cuda'  # 计算设备（cuda/cpu）
SEED = 42  # 随机种子

# 网格参数定义（m 单位，国际单位制）
# 基于实际数据范围：x=5025-12975mm，y=2025-5975mm
X_MIN = 5.025  # x方向最小值（5025mm -> 5.025m）
X_MAX = 12.975  # x方向最大值（12975mm -> 12.975m）
Y_MIN = 2.025  # y方向最小值（2025mm -> 2.025m）
Y_MAX = 5.975  # y方向最大值（5975mm -> 5.975m）
Z_MIN = PLANE_SIZE_Z  # z方向最小值（固定）
Z_MAX = PLANE_SIZE_Z  # z方向最大值（固定）
GRID_RESOLUTION = 500 # 网格分辨率
UPSAMPLING_FACTOR = 1  # 上采样因子

# 归一化参数（基于实际数据范围，m单位）
norm_params = {
    'x_min': X_MIN,      # m，x方向最小值（5.025m）
    'x_max': X_MAX,      # m，x方向最大值（12.975m）
    'y_min': Y_MIN,      # m，y方向最小值（2.025m）
    'y_max': Y_MAX,      # m，y方向最大值（5.975m）
    'z_min': PLANE_SIZE_Z,
    'z_max': PLANE_SIZE_Z,
    'u_min': -0.22,      # m/s，基于实际数据u范围约-0.22-1.4
    'u_max': 1.4,        # m/s，u方向最大值
    'v_min': -0.54,      # m/s，基于实际数据v范围约-0.54-0.54
    'v_max': 0.54,       # m/s，v方向最大值
    't_min': TIME_START,
    't_max': TIME_END
}

def update_norm_params(x_data, y_data, z_data=None, u_data=None, v_data=None, t_data=None):
    """根据实际数据更新归一化参数（统一m制）"""
    global norm_params
    # 数据已经是m制单位，无需转换
    x_min, x_max = np.min(x_data), np.max(x_data)
    y_min, y_max = np.min(y_data), np.max(y_data)
    
    norm_params['x_min'] = x_min
    norm_params['x_max'] = x_max
    norm_params['y_min'] = y_min
    norm_params['y_max'] = y_max
    
    if z_data is not None:
        z_min, z_max = np.min(z_data), np.max(z_data)
        norm_params['z_min'] = z_min
        norm_params['z_max'] = z_max
    
    if u_data is not None:
        norm_params['u_min'] = np.min(u_data)
        norm_params['u_max'] = np.max(u_data)
    if v_data is not None:
        norm_params['v_min'] = np.min(v_data)
        norm_params['v_max'] = np.max(v_data)
    
    if t_data is not None:
        norm_params['t_min'] = np.min(t_data)
        norm_params['t_max'] = np.max(t_data)

def get_data_mode():
    """获取当前数据模式"""
    return "默认数据" if shuju == 0 else "读取数据"



# Mamba架构参数
MAMBA_STATE_DIM = 40  # Mamba状态空间维度
MAMBA_EXPAND = 2  # Mamba扩展因子
MAMBA_TIME_WINDOW = 50  # Mamba时序窗口大小
BLOCK_WEIGHTS = [2.0, 1.0, 0.5]  # 区域权重配置 [核心区域, 过渡区域, 外部区域]

# 报告频率
REPORT_EVERY = 100  # 报告频率（每多少轮报告一次）
HISTORY_EVERY = 100  # 每多少轮记录一次历史数据

# 数据分割参数
VALIDATION_SPLIT = 0.1  # 验证集比例（10%用于验证）

# 输出和文件参数
OUTPUT_DIR = 'out_fields_2d'  # 默认输出目录
ECHO_LEVEL = 1  # 输出详细程度（0:静默，1:标准，2:详细）
FRAMES_COUNT = 100  # 默认帧数
MULTIGRID_LEVELS = 0  # 多重网格级别（0:禁用）
MULTIGRID_NLVL = 5  # 多重网格层数
def print_config():
    """打印当前配置信息"""
    print("="*50)
    print("2D项目全局配置参数")
    print("="*50)
    print(f"数据模式: {get_data_mode()} (shuju={shuju})")
    print(f"网络深度: {NETWORK_DEPTH}")
    print(f"隐藏层大小: {HIDDEN_SIZE}")
    print(f"激活函数: {ACTIVATION}")
    print(f"x方向平面尺寸: {PLANE_SIZE_X} mm")
    print(f"y方向平面尺寸: {PLANE_SIZE_Y} mm")
    print(f"z方向平面尺寸: {PLANE_SIZE_Z} mm (固定)")
    print(f"时间范围: {TIME_START}s - {TIME_END}s")
    print(f"时间步长: {TIME_STEP}s")
    print(f"时间步骤: {N_TIME_STEPS}")
    print(f"Mamba时序扩展: {'启用' if TEMPORAL_EXTENSION['enable'] else '禁用'}")
    if TEMPORAL_EXTENSION['enable']:
        print(f"  - 扩展阶段: {TEMPORAL_EXTENSION['stages']}")
        print(f"  - 阶段时间: {TEMPORAL_EXTENSION['stage_times']}")
        print(f"  - 时序窗口: {TEMPORAL_EXTENSION['temporal_window_size']}")
        print(f"  - 一致性权重: {TEMPORAL_EXTENSION['consistency_weight']}")
    print(f"学习率: {LEARNING_RATE}")
    print(f"Adam训练轮数: {N_EPOCHS}")
    print(f"报告频率: {REPORT_EVERY}")
    print(f"批次大小: {BATCH_SIZE}")
    print(f"雷诺数: {REYNOLDS_NUMBER}")
    print(f"数据损失权重: {W_DATA}")
    print(f"物理损失权重: {W_PHYS}")
    print(f"边界损失权重: {W_BND}")
    print(f"进气边损失权重: {W_INLET_BND}")
    print(f"其他边界损失权重: {W_OTHER_BND}")
    print(f"绘图网格大小: {PLOT_GRID_SIZE}")
    print(f"图像分辨率: {PLOT_DPI} DPI")
    print(f"保存路径: {SAVE_PATH}")
    print(f"绘图间隔: {PLOT_INTERVAL} 轮")
    print(f"动画时间范围: {ANIMATION_TIME_RANGE}")
    print(f"动画帧数: {ANIMATION_NUM_FRAMES}")
    print(f"动画网格大小: {ANIMATION_GRID_SIZE}")
    print(f"速度差热图时间: {VELOCITY_DIFF_TIMES} s")
    print(f"动画生成间隔: {PLOT_INTERVAL_ANIMATION} 轮")
    print(f"速度差热图绘制间隔: {PLOT_INTERVAL_VELOCITY_DIFF} 轮")
    print(f"GMSH文件: {GMSH_POINTS_FILE}")
    print(f"PIV数据文件夹: {PIV_DATA_FOLDER}")
    print(f"默认数据文件: {DEFAULT_DATA_FILE}")
    print(f"真实数据文件: {REAL_DATA_FILE}")
    print(f"验证集比例: {VALIDATION_SPLIT}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"输出详细程度: {ECHO_LEVEL}")
    print(f"默认帧数: {FRAMES_COUNT}")
    print(f"多重网格级别: {MULTIGRID_LEVELS}")
    print(f"多重网格层数: {MULTIGRID_NLVL}")
    print(f"计算设备: {DEVICE}")
    print(f"随机种子: {SEED}")
    print("="*50)