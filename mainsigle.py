import numpy as np
import scipy as sc
from scipy.interpolate import griddata  # 用于数据插值
import matplotlib
from matplotlib import pyplot as plt
import os
from datetime import datetime
from sklearn.metrics import r2_score  # 用于计算R²指标

# 设置环境变量以消除警告
os.environ['LOKY_MAX_CPU_COUNT'] = '8'  # 设置Joblib使用的核心数

# 使用PyTorch GPU后端
import torch
import argparse

import loadPIVData as ldw
import fdTools as fdt
import makeProblem as mkp
import visualize
import utils
import pandas as pd  # 用于读取真实数据文件
# 导入2D配置
import config_2d as config
from config_2d import shuju, PLANE_SIZE_X, PLANE_SIZE_Y, PLANE_SIZE_Z, REAL_DATA_FILE

# 设置中文显示
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 导入时间模块
import time

# 设置设备为GPU（如果可用）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--Nci', type=int, default=config.N_INTERNAL_POINTS, help="Number of interior collocation points")
    parser.add_argument('--nNHL',type=int,default=config.HIDDEN_SIZE,help="Number of neurons in hidden layer")
    parser.add_argument('--nHL',type=int,default=config.NETWORK_DEPTH,help="Number of hidden layers")
    parser.add_argument('--W_dat', type=float, default=config.W_DATA, help="Weight for data loss")
    parser.add_argument('--W_phys', type=float, default=config.W_PHYS, help="Weight for physics loss")
    parser.add_argument('--xmin', type=float, default=config.X_MIN, help="minimum x")
    parser.add_argument('--xmax', type=float, default=config.PLANE_SIZE_X, help="maximum x")
    parser.add_argument('--ymin', type=float, default=config.Y_MIN, help="minimum y")
    parser.add_argument('--ymax', type=float, default=config.PLANE_SIZE_Y, help="maximum y")
    parser.add_argument('--t_min', type=float, default=config.TIME_START, help="最小时间(s)")
    parser.add_argument('--t_max', type=float, default=config.TIME_END, help="最大时间(s)")
    parser.add_argument('--dt', type=float, default=config.TIME_STEP, help="时间步长(s)")
    parser.add_argument('--pval',type=float, default=config.VALIDATION_SPLIT, help="percentage of training data reserved for validation")
    parser.add_argument('--nadam',type=int, default=20000, help="number of epochs using ADAM")
    parser.add_argument('--adamLR',type=float, default=config.LEARNING_RATE, help="learning rate for ADAM")
    parser.set_defaults(outdir=config.OUTPUT_DIR)
    parser.set_defaults(echo=config.ECHO_LEVEL)
    parser.set_defaults(frames=config.FRAMES_COUNT,
                        plot_every=config.PLOT_INTERVAL,
                        report_every=config.REPORT_EVERY,
                        history_every=config.HISTORY_EVERY)
    parser.set_defaults(multigrid=config.MULTIGRID_LEVELS)
    parser.set_defaults(nlvl=config.MULTIGRID_NLVL)
    return parser.parse_args()

# parse arguments 
args = parse_args()

# 创建带时间戳的输出目录
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
output_base_dir = "E:/yzy/pivresult1"
timestamped_dir = f"2d_pinn_results_{current_time}"
args.outdir = os.path.join(output_base_dir, timestamped_dir)

# 确保输出目录存在
os.makedirs(args.outdir, exist_ok=True)
print(f"输出目录: {args.outdir}")

# 创建实验信息文件
with open(os.path.join(args.outdir, 'experiment_info.txt'), 'w', encoding='utf-8') as f:
    f.write(f"2D PINN 实验信息\n")
    f.write(f"="*50 + "\n")
    f.write(f"实验时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"输出目录: {args.outdir}\n")
    f.write(f"网络结构: {args.nHL}层隐藏层, 每层{args.nNHL}神经元\n")
    f.write(f"Adam训练轮数: {args.nadam}\n")
    f.write(f"Adam学习率: {args.adamLR}\n")
    f.write(f"时间范围: {args.t_min}s - {args.t_max}s\n")
    f.write(f"初始数据文件: {config.REAL_DATA_FILE}\n")
    f.write(f"Reynolds数: {config.REYNOLDS_NUMBER}\n")
    f.write(f"计算域: x=[{args.xmin}, {args.xmax}]m, y=[{args.ymin}, {args.ymax}]m\n")
    f.write(f"时间步数: {config.N_TIME_STEPS}\n")

# 导入全局配置
import config_2d

# 打印配置信息
config.print_config()

# 读取singletemporal_data数据作为初始条件
def load_initial_data():
    """加载singletemporal_data/sigle_t=0.0s.txt作为初始条件数据"""
    try:
        # 使用data2文件夹中的数据
        data_file = 'data2/sigle_t=0.00s.txt'
        data = pd.read_csv(data_file, sep=r'\s+')
        print(f"成功加载 {len(data)} 个数据点从 {data_file}")
        
        # 提取坐标和速度分量（转换为 m 单位）
        x_data = data['x(mm)'].values / 1000.0  # mm -> m
        y_data = data['y(mm)'].values / 1000.0  # mm -> m
        # 2D数据没有z坐标，使用固定值
        z_data = np.zeros_like(x_data)  # 2D问题，z坐标设为0
        u_data = data['u(m/s)'].values
        v_data = data['v(m/s)'].values
        
        print(f"数据范围: x=[{x_data.min():.3f}, {x_data.max():.3f}] m, y=[{y_data.min():.3f}, {y_data.max():.3f}] m")
        print(f"速度范围: u=[{u_data.min():.2f}, {u_data.max():.2f}]m/s, v=[{v_data.min():.2f}, {v_data.max():.2f}]m/s")
        
        return x_data, y_data, z_data, u_data, v_data
    except Exception as e:
        print(f"读取数据文件失败: {e}")
        return None, None, None, None, None

# 加载初始数据
x_data, y_data, z_data, u_data, v_data = load_initial_data()

print(f"当前使用的归一化范围 (应为全局配置):")
print(f"  u: [{config.norm_params['u_min']}, {config.norm_params['u_max']}]")
print(f"  v: [{config.norm_params['v_min']}, {config.norm_params['v_max']}]")

# 定义归一化函数
def normalize_data(x, y, u=None, v=None, w=None, t=None):
    """将数据归一化到[-1, 1]区间（2D问题，不包含z坐标）"""
    # 坐标归一化
    x_norm = 2.0 * (x - config.norm_params['x_min']) / (config.norm_params['x_max'] - config.norm_params['x_min']) - 1.0
    y_norm = 2.0 * (y - config.norm_params['y_min']) / (config.norm_params['y_max'] - config.norm_params['y_min']) - 1.0
    
    result = [x_norm, y_norm]
    
    if u is not None:
        u_range = config.norm_params['u_max'] - config.norm_params['u_min']
        if u_range > 1e-10:
            u_norm = 2.0 * (u - config.norm_params['u_min']) / u_range - 1.0
        else:
            u_norm = np.zeros_like(u)
        result.append(u_norm)
    
    if v is not None:
        v_range = config.norm_params['v_max'] - config.norm_params['v_min']
        if v_range > 1e-10:
            v_norm = 2.0 * (v - config.norm_params['v_min']) / v_range - 1.0
        else:
            v_norm = np.zeros_like(v)
        result.append(v_norm)
    
    if w is not None:
        # 2D项目不需要w速度数据，直接返回零数组
        w_norm = np.zeros_like(w)
        result.append(w_norm)
    
    if t is not None:
        t_range = config.norm_params['t_max'] - config.norm_params['t_min']
        if t_range > 1e-10:
            t_norm = 2.0 * (t - config.norm_params['t_min']) / t_range - 1.0
        else:
            t_norm = np.zeros_like(t)
        result.append(t_norm)
    
    return result

def denormalize_velocity(u_norm, v_norm):
    """将归一化的速度反归一化"""
    u = 0.5 * (u_norm + 1.0) * (config.norm_params['u_max'] - config.norm_params['u_min']) + config.norm_params['u_min']
    v = 0.5 * (v_norm + 1.0) * (config.norm_params['v_max'] - config.norm_params['v_min']) + config.norm_params['v_min']
    return u, v


def get_ground_truth_data(x_coords, y_coords, t_coords):
    """从坐标点获取真实数据值"""
    
    # 将归一化坐标反归一化到物理空间
    x_denorm = (x_coords.cpu().numpy() + 1.0) / 2.0 * (config.norm_params['x_max'] - config.norm_params['x_min']) + config.norm_params['x_min']
    y_denorm = (y_coords.cpu().numpy() + 1.0) / 2.0 * (config.norm_params['y_max'] - config.norm_params['y_min']) + config.norm_params['y_min']
    t_denorm = (t_coords.cpu().numpy() + 1.0) / 2.0 * (config.norm_params['t_max'] - config.norm_params['t_min']) + config.norm_params['t_min']
    
    # 转换为mm单位（与数据文件一致）
    x_mm = x_denorm * 1000.0  # m -> mm
    y_mm = y_denorm * 1000.0  # m -> mm
    
    # 初始化结果数组
    u_data = np.zeros(len(x_mm))
    v_data = np.zeros(len(x_mm))
    
    # 对每个时间点分别处理
    unique_times = np.unique(t_denorm)
    
    for t_phys in unique_times:
        # 处理浮点数精度问题：四舍五入到最接近的0.01秒
        t_rounded = round(t_phys * 100) / 100.0
        
        # 构建文件名
        filename = f'sigle_t={t_rounded:.2f}s.txt'
        filepath = os.path.join('data2', filename)
        
        if not os.path.exists(filepath):
            print(f"⚠️  无法加载时刻{t_phys:.2f}s数据: 文件不存在 {filepath}")
            continue
        
        try:
            # 读取数据
            data = pd.read_csv(filepath, sep=r'\s+')
            
            # 提取真实数据
            x_real_mm = data['x(mm)'].values
            y_real_mm = data['y(mm)'].values
            u_real = data['u(m/s)'].values
            v_real = data['v(m/s)'].values
            
            # 找到当前时间点的索引
            time_mask = np.abs(t_denorm - t_phys) < 0.01
            
            if np.sum(time_mask) > 0:
                # 使用插值获取对应点的速度值
                points_real = np.column_stack([x_real_mm, y_real_mm])
                # 修复：使用一维索引操作，避免二维索引错误
                x_mm_masked = x_mm[time_mask]
                y_mm_masked = y_mm[time_mask]
                points_query = np.column_stack([x_mm_masked, y_mm_masked])
                
                # 线性插值
                u_interp = griddata(points_real, u_real, points_query, method='linear', fill_value=0.0)
                v_interp = griddata(points_real, v_real, points_query, method='linear', fill_value=0.0)
                
                # 填充结果
                u_data[time_mask] = u_interp
                v_data[time_mask] = v_interp
                
        except Exception as e:
            print(f"[错误] 加载时刻{t_phys:.2f}s数据失败: {e}")
    
    # 转换为张量并归一化
    u_norm = 2.0 * (u_data - config.norm_params['u_min']) / (config.norm_params['u_max'] - config.norm_params['u_min']) - 1.0
    v_norm = 2.0 * (v_data - config.norm_params['v_min']) / (config.norm_params['v_max'] - config.norm_params['v_min']) - 1.0
    
    u_tensor = torch.tensor(u_norm, dtype=torch.float32).view(-1, 1)
    v_tensor = torch.tensor(v_norm, dtype=torch.float32).view(-1, 1)
    
    return u_tensor, v_tensor

def interpolate_from_cached_data(x_coords, y_coords, t_coords, cached_data):
    """从缓存数据中插值获取速度值，避免重复调用get_ground_truth_data"""
    
    # 提取缓存数据
    x_cached = cached_data['x_data'].cpu().numpy()
    y_cached = cached_data['y_data'].cpu().numpy()
    t_cached = cached_data['t_data'].cpu().numpy()
    u_cached = cached_data['u_data'].cpu().numpy()
    v_cached = cached_data['v_data'].cpu().numpy()
    
    # 转换输入坐标为numpy数组
    x_query = x_coords.cpu().numpy()
    y_query = y_coords.cpu().numpy()
    t_query = t_coords.cpu().numpy()
    
    # 初始化结果数组
    u_interp = np.zeros(len(x_query))
    v_interp = np.zeros(len(x_query))
    
    # 对每个时间点分别进行插值
    unique_times = np.unique(t_query)
    
    for t_val in unique_times:
        # 找到当前时间点的查询点
        time_mask = np.abs(t_query - t_val) < 0.01
        
        if np.sum(time_mask) > 0:
            # 找到缓存数据中对应时间点的数据
            cached_time_mask = np.abs(t_cached - t_val) < 0.01
            
            if np.sum(cached_time_mask) > 0:
                # 使用缓存数据进行插值
                points_cached = np.column_stack([x_cached[cached_time_mask], y_cached[cached_time_mask]])
                points_query = np.column_stack([x_query[time_mask], y_query[time_mask]])
                
                # 线性插值
                u_interp_local = griddata(points_cached, u_cached[cached_time_mask], 
                                         points_query, method='linear', fill_value=0.0)
                v_interp_local = griddata(points_cached, v_cached[cached_time_mask], 
                                         points_query, method='linear', fill_value=0.0)
                
                # 填充结果 - 修复索引问题
                # 确保u_interp_local和v_interp_local是一维数组
                if u_interp_local.ndim > 1:
                    u_interp_local = u_interp_local.flatten()
                if v_interp_local.ndim > 1:
                    v_interp_local = v_interp_local.flatten()
                
                # 使用整数索引而不是布尔索引
                time_indices = np.where(time_mask)[0]
                u_interp[time_indices] = u_interp_local
                v_interp[time_indices] = v_interp_local
            else:
                # 如果没有对应时间点的缓存数据，使用默认值
                u_interp[time_mask] = 0.0
                v_interp[time_mask] = 0.0
    
    # 转换为张量
    u_tensor = torch.tensor(u_interp, dtype=torch.float32, device=x_coords.device).unsqueeze(1)
    v_tensor = torch.tensor(v_interp, dtype=torch.float32, device=x_coords.device).unsqueeze(1)
    
    return u_tensor, v_tensor

# 创建2D网格点（使用随机生成方法）
def create_2d_mesh_points():
    """创建2D网格点用于训练，使用随机生成方法"""
    
    print("使用随机网格点生成方法")
    
    # 使用配置的几何尺寸（m单位）
    x_min, x_max = config.X_MIN, config.PLANE_SIZE_X
    y_min, y_max = config.Y_MIN, config.PLANE_SIZE_Y
    
    # 生成统一的点集，不区分内部点和边界点
    n_total = args.Nci  # 总点数由配置决定
    
    # 生成均匀分布的点
    def generate_uniform_points(n_points, x_min, x_max, y_min, y_max):
        """
        生成均匀分布的点
        """
        x_points = np.random.uniform(x_min, x_max, n_points)
        y_points = np.random.uniform(y_min, y_max, n_points)
        return x_points, y_points
    
    # 生成均匀分布的点集
    x_all, y_all = generate_uniform_points(n_total, x_min, x_max, y_min, y_max)
    
    # 创建边界标志和类型（根据坐标位置动态判断）
    boundary_flags = np.zeros(len(x_all), dtype=int)
    boundary_types = np.zeros(len(x_all), dtype=int)
    
    # 边界判断容差
    tolerance = 1e-6
    
    # 判断边界类型
    bottom_mask = np.abs(y_all - y_min) < tolerance  # 底部
    top_mask = np.abs(y_all - y_max) < tolerance     # 顶部
    left_mask = np.abs(x_all - x_min) < tolerance    # 左侧
    right_mask = np.abs(x_all - x_max) < tolerance   # 右侧
    
    # 设置边界标志
    boundary_flags[bottom_mask | top_mask | left_mask | right_mask] = 1
    
    # 设置边界类型
    boundary_types[bottom_mask] = 1  # 底部固体壁面
    boundary_types[top_mask] = 2    # 顶部固体壁面
    boundary_types[left_mask] = 3   # 左侧固体壁面
    boundary_types[right_mask] = 4  # 右侧出口
    
    return x_all, y_all, boundary_flags, boundary_types

# 创建训练点
x_all, y_all, boundary_flags, boundary_types = create_2d_mesh_points()
# 删除训练点统计信息的打印

# ==========================================
# [基础类修复] 必须支持 force_stateless 参数
# ==========================================
# ==============================================================================
# [高性能核心] JIT 编译的扫描函数 (保持不变，用于加速)
# ==============================================================================
@torch.jit.script
def mamba_rnn_scan(x_proj: torch.Tensor, A_discrete: torch.Tensor, h_init: torch.Tensor) -> torch.Tensor:
    """
    执行 RNN 扫描: h_t = tanh( A_t * h_{t-1} + x_proj_t )
    """
    batch_size = x_proj.size(0)
    seq_len = x_proj.size(1)
    
    current_state = h_init
    states = []
    
    for t in range(seq_len):
        A_t = A_discrete[:, t, :, :]
        x_t = x_proj[:, t, :]
        state_evolved = torch.bmm(current_state.unsqueeze(1), A_t.transpose(1, 2)).squeeze(1)
        current_state = torch.tanh(state_evolved + x_t)
        states.append(current_state)
        
    return torch.stack(states, dim=1)

# ==============================================================================
# [高性能核心] Mamba 层 (修改：支持 h_prev 输入和 h_last 输出)
# ==============================================================================
class SelectiveStateSpaceMamba(torch.nn.Module):
    def __init__(self, hidden_size, state_dim=16, dt_rank=1, force_stateless=False):
        super(SelectiveStateSpaceMamba, self).__init__()
        
        self.hidden_size = hidden_size
        self.state_dim = state_dim
        self.dt_rank = dt_rank
        self.force_stateless = force_stateless
        
        # 参数定义
        self.A = torch.nn.Parameter(torch.randn(state_dim, state_dim) * 0.02)
        self.B = torch.nn.Parameter(torch.randn(hidden_size, state_dim) * 0.02)
        self.C = torch.nn.Parameter(torch.randn(state_dim, hidden_size) * 0.02)
        
        self.dt_proj = torch.nn.Linear(hidden_size, dt_rank)
        self.dt_bias = torch.nn.Parameter(torch.randn(dt_rank) * 0.02)
        
        self.selective_gate = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size // 2),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_size // 2, hidden_size),
            torch.nn.Sigmoid()
        )
        self.residual_scale = torch.nn.Parameter(torch.ones(1))
        self.layer_norm = torch.nn.LayerNorm(hidden_size)
        
        self.selective_param_generator = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size // 2),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_size // 2, 3 * state_dim),
            torch.nn.Tanh()
        )
        
    def forward(self, x, t, h_prev=None):
        """
        Args:
            h_prev: [Batch, State_Dim] 上一步的状态 (用于推理接力)
        Returns:
            output: [Batch, Seq_Len, Hidden]
            h_last: [Batch, State_Dim] 最后一步的状态
        """
        # 1. 维度标准化
        is_sequence = (x.dim() == 3)
        if not is_sequence:
            x = x.unsqueeze(1)
            t = t.unsqueeze(1)
            
        batch_size, seq_len, hidden = x.shape
        
        # 2. 参数计算 (并行)
        t_expanded = t.expand(batch_size, seq_len, self.hidden_size)
        dt = torch.nn.functional.softplus(self.dt_proj(x) + self.dt_bias)
        selective_params = self.selective_param_generator(x).view(batch_size, seq_len, 3, self.state_dim)
        
        A_dynamic = self.A.view(1, 1, self.state_dim, self.state_dim) + \
                    torch.diag_embed(selective_params[:, :, 0]) * 0.1
        B_dynamic = self.B.view(1, 1, self.hidden_size, self.state_dim) + \
                    selective_params[:, :, 1].unsqueeze(2) * 0.1
        C_dynamic = self.C.view(1, 1, self.state_dim, self.hidden_size) + \
                    selective_params[:, :, 2].unsqueeze(3) * 0.1
        
        # 3. 离散化 & 输入投影
        dt_expanded = dt.unsqueeze(-1)
        A_discrete = torch.exp(A_dynamic * dt_expanded)
        B_discrete = B_dynamic * dt_expanded
        input_proj = torch.matmul(x.unsqueeze(2), B_discrete).squeeze(2)
        
        # 4. 状态扫描 (关键逻辑修正)
        if h_prev is None:
            h0 = torch.zeros(batch_size, self.state_dim, device=x.device) # 训练时或无记忆时从0开始
        else:
            h0 = h_prev # 推理时接力上一步
            
        if self.force_stateless:
            states = torch.tanh(input_proj)
        else:
            states = mamba_rnn_scan(input_proj, A_discrete, h0)
            
        # 5. 输出投影
        output = torch.matmul(states.unsqueeze(2), C_dynamic).squeeze(2)
        gate = self.selective_gate(x + t_expanded)
        output = output * gate
        final_output = self.layer_norm(x + self.residual_scale * output)
        
        if not is_sequence:
            final_output = final_output.squeeze(1)
            
        # 返回输出 + 最后状态
        last_state = states[:, -1, :] 
        return final_output, last_state

# 定义2D PINN网络（纯MLP结构）
# [修改] 神经元改为20，输出改为3 (u,v,p)
class PINN_2D(torch.nn.Module):
    def __init__(self, n_input=3, n_output=3, n_layers=8, n_neurons=20): # 修改：n_output=3, n_neurons=20
        super(PINN_2D, self).__init__()
        
        self.n_layers = n_layers
        self.n_neurons = n_neurons
        
        self.layers = torch.nn.ModuleList()
        
        # 输入层
        self.layers.append(torch.nn.Linear(n_input, n_neurons))
        
        # 隐藏层
        for _ in range(n_layers - 1):
            self.layers.append(torch.nn.Linear(n_neurons, n_neurons))
        
        # 输出层
        self.layers.append(torch.nn.Linear(n_neurons, n_output))
        
        # 激活函数
        self.activation = torch.nn.Tanh()
        
    def _initialize_weights(self):
        """初始化网络权重，使用Xavier初始化避免梯度爆炸"""
        for layer in self.layers:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight, gain=torch.nn.init.calculate_gain('tanh'))
                if layer.bias is not None:
                    torch.nn.init.zeros_(layer.bias)
        
        # 对输出层使用更小的初始化范围
        if self.layers:
            torch.nn.init.xavier_uniform_(self.layers[-1].weight, gain=0.1)
            if self.layers[-1].bias is not None:
                torch.nn.init.zeros_(self.layers[-1].bias)
    
    def forward(self, x):
        # 前向传播
        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.activation(x)
        
        # 输出层
        x = self.layers[-1](x)
        return x

# [修改] 创建模型：输出3维(u,v,p)，每层20神经元
model = PINN_2D(n_input=3, n_output=3, n_layers=args.nHL, n_neurons=20).to(device) # 强制20神经元

# 初始化模型权重
model._initialize_weights()
print(f"创建了2D PINN模型: 输入3维(x,y,t), 输出3维(u,v,p), {args.nHL}层隐藏层, 每层20神经元")

# Mamba序列预测器（Teacher模型）
# ==============================================================================
# [高性能核心] Mamba 序列预测器 (修改：管理多层状态传递)
# ==============================================================================
# ==============================================================================
# [高性能核心] Mamba 序列预测器 (支持 Seq2Seq 直接输出未来)
# ==============================================================================
class MambaSequencePredictor(torch.nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64, state_dim=16, num_layers=3, pred_steps=200): # [修改] 默认 200 步
        super(MambaSequencePredictor, self).__init__()
        
        self.input_dim = input_dim 
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        self.num_layers = num_layers
        self.pred_steps = pred_steps 
        
        self.input_proj = torch.nn.Linear(input_dim, hidden_dim)
        
        self.mamba_layers = torch.nn.ModuleList([
            SelectiveStateSpaceMamba(hidden_dim, state_dim=state_dim, force_stateless=False) 
            for _ in range(num_layers)
        ])
        
        self.activation = torch.nn.Tanh()
        self.output_proj = torch.nn.Linear(hidden_dim, 2)
        
        # [修改] 升级为 MLP 头，输出 200 * 2 = 400 维
        self.direct_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim * 2),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim * 2, pred_steps * 2) 
        ) 
        
    def forward(self, x, use_direct_head=False):
        x_proj = self.input_proj(x)
        batch_size, seq_len, _ = x_proj.shape
        t_val = torch.zeros((batch_size, seq_len, 1), device=x.device)
        
        current_x = x_proj
        for layer in self.mamba_layers:
            current_x, _ = layer(current_x, t_val)
            current_x = self.activation(current_x)
            
        if use_direct_head:
            last_feat = current_x[:, -1, :] 
            future_pred = self.direct_head(last_feat)
            return future_pred.view(batch_size, self.pred_steps, 2)
        else:
            return self.output_proj(current_x)

    def predict_sequence(self, initial_uv, fixed_xy, seq_length=10, dt=0.1, warmup_uv=None):
        if warmup_uv is None:
            return torch.zeros((initial_uv.shape[0], seq_length, 2), device=initial_uv.device)
        batch_size, w_len, _ = warmup_uv.shape
        xy_expanded = fixed_xy.unsqueeze(1).expand(-1, w_len, -1)
        warmup_input = torch.cat([xy_expanded, warmup_uv], dim=2)
        
        self.eval()
        with torch.no_grad():
            predictions = self.forward(warmup_input, use_direct_head=True)
        return predictions

# ==============================================================================
# [配置调整] Mamba 参数: 极速瘦身版 (Ultra-Lite)
# ==============================================================================
# 针对强周期流场，只需极少的参数即可拟合。
# Hidden=32, State=16, Layers=2 是捕捉简单周期信号的黄金比例。
# [修改] Mamba参数减半：Hidden=32, State=8
mamba_teacher = MambaSequencePredictor(
    input_dim=4,  
    hidden_dim=32,   # 降为原来的一半 (64->32)
    state_dim=8,     # 降为原来的一半 (16->8)
    num_layers=3     
).to(device)

# 禁用 torch.compile 以避免 Windows/Dynamo 兼容性报错
print("⚠️ 已禁用 torch.compile 以避免运行时错误")
# try:
#     mamba_teacher = torch.compile(mamba_teacher, mode='reduce-overhead', fullgraph=False)
#     print("✓ Mamba模型编译优化成功")
# except Exception as e:
#     print(f"⚠️  Mamba模型编译优化失败: {e}")

print(f"创建了Mamba模型: Hidden=64 | State=16 | Layers=3")

# 定义速度计算函数（模型直接输出速度，不再通过流函数计算）
def compute_velocity_from_stream_function(psi, x_tensor, y_tensor):
    """⚠️  已弃用：模型现在直接输出速度(u,v)，不再通过流函数计算"""
    
    # 返回零速度作为占位符，避免破坏现有代码结构
    u = torch.zeros_like(psi)
    v = torch.zeros_like(psi)
    
    return u, v

# 新增：直接从模型输出获取速度
def get_velocity_from_model_output(output):
    """从模型输出直接提取速度分量"""
    u_norm = output[:, 0:1]  # 速度u分量
    v_norm = output[:, 1:2]  # 速度v分量
    return u_norm, v_norm
def physics_loss_2d(model, x_phys, y_phys, t_phys, Re, norm_params):
    """计算2D Navier-Stokes方程残差损失（包含压力项P）- 修正版"""
    
    try:
        # 需要计算导数的张量
        x_phys_tensor = x_phys.clone().detach().requires_grad_(True)
        y_phys_tensor = y_phys.clone().detach().requires_grad_(True)
        t_phys_tensor = t_phys.clone().detach().requires_grad_(True)
        
        # 模型预测
        inputs = torch.cat([x_phys_tensor, y_phys_tensor, t_phys_tensor], dim=1)
        output = model(inputs)
        
        u_norm = output[:, 0:1]
        v_norm = output[:, 1:2]
        p_norm = output[:, 2:3]
        
        # 反归一化速度值 (正确)
        u_phys, v_phys = denormalize_velocity(u_norm, v_norm)
        
        # ================== 修正开始 ==================
        # 1. 获取坐标缩放系数 (d_norm / d_phys)
        x_scale = 2.0 / (norm_params['x_max'] - norm_params['x_min'])
        y_scale = 2.0 / (norm_params['y_max'] - norm_params['y_min'])
        t_scale = 2.0 / (norm_params['t_max'] - norm_params['t_min'])
        
        # 2. 获取值域缩放系数 (u_range / 2.0)
        # 这是 main.py 之前缺失的部分！
        u_range = norm_params['u_max'] - norm_params['u_min']
        v_range = norm_params['v_max'] - norm_params['v_min']
        coef_u = u_range / 2.0
        coef_v = v_range / 2.0
        
        # 3. 计算导数 (必须同时乘坐标系数和值系数)
        # 链式法则: du_phys/dt = (du_norm/dt_norm) * (dt_norm/dt) * (du_phys/du_norm)
        #                      = grad * t_scale * coef_u
        
        # Helper function specifically for grad to clean up code
        def get_grad(y, x):
            return torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True)[0]

        # 一阶导数 (修正后)
        u_t = get_grad(u_norm, t_phys_tensor) * t_scale * coef_u
        v_t = get_grad(v_norm, t_phys_tensor) * t_scale * coef_v
        
        u_x = get_grad(u_norm, x_phys_tensor) * x_scale * coef_u
        u_y = get_grad(u_norm, y_phys_tensor) * y_scale * coef_u
        
        v_x = get_grad(v_norm, x_phys_tensor) * x_scale * coef_v
        v_y = get_grad(v_norm, y_phys_tensor) * y_scale * coef_v
        
        # 压力导数 (通常假设p直接输出或有单独缩放，这里保持原样或仅乘坐标缩放)
        p_x = get_grad(p_norm, x_phys_tensor) * x_scale 
        p_y = get_grad(p_norm, y_phys_tensor) * y_scale 
        
        # 二阶导数 (修正后)
        # 注意：对 u_x (物理导数) 再次求导，只需要乘坐标系数 x_scale
        # 或者对 u_xx_norm 求导，需要乘 x_scale^2 * coef_u
        # 这里用更稳妥的方式：直接对 u_x (物理) 求导
        u_xx = get_grad(u_x, x_phys_tensor) * x_scale 
        u_yy = get_grad(u_y, y_phys_tensor) * y_scale
        v_xx = get_grad(v_x, x_phys_tensor) * x_scale
        v_yy = get_grad(v_y, y_phys_tensor) * y_scale
        # ================== 修正结束 ==================

        # 2D连续性方程
        continuity = u_x + v_y
        
        # 2D动量方程
        x_momentum = u_t + u_phys * u_x + v_phys * u_y + p_x - (1/Re) * (u_xx + u_yy)
        y_momentum = v_t + u_phys * v_x + v_phys * v_y + p_y - (1/Re) * (v_xx + v_yy)
        
        # 计算损失
        continuity_loss = torch.mean(continuity**2)
        x_momentum_loss = torch.mean(x_momentum**2)
        y_momentum_loss = torch.mean(y_momentum**2)
        vorticity_loss = torch.tensor(0.0, device=x_phys.device)
        
        total_physics_loss = continuity_loss + x_momentum_loss + y_momentum_loss
        
        return total_physics_loss, continuity_loss, x_momentum_loss, y_momentum_loss, vorticity_loss
            
    except Exception as e:
        print(f"⚠️  物理损失计算异常: {e}")
        # 返回 safe tensors... (保持原样)
        return (torch.tensor(1e-6, device=x_phys.device), 
               torch.tensor(1e-6, device=x_phys.device),
               torch.tensor(1e-6, device=x_phys.device), 
               torch.tensor(1e-6, device=x_phys.device),
               torch.tensor(1e-6, device=x_phys.device))

# 创建可视化函数
def plot_2d_flow_field(model, t_value, save_dir, epoch):
    """绘制2D流场"""
    
    # 创建网格 - 使用streamplot要求的格式
    nx, ny = 100, 100
    x = np.linspace(config.norm_params['x_min'], config.norm_params['x_max'], nx)
    y = np.linspace(config.norm_params['y_min'], config.norm_params['y_max'], ny)
    X, Y = np.meshgrid(x, y)  # X: (nx, ny), Y: (nx, ny) - 默认格式适合streamplot
    
    # 展平网格 - 按行优先顺序
    X_flat = X.ravel()
    Y_flat = Y.ravel()
    T_flat = np.full_like(X_flat, t_value)
    
    # 归一化输入
    X_norm, Y_norm, T_norm = normalize_data(X_flat, Y_flat, t=T_flat)
    
    # 转换为tensor
    X_tensor = torch.tensor(X_norm, dtype=torch.float32, device=device).unsqueeze(1)
    Y_tensor = torch.tensor(Y_norm, dtype=torch.float32, device=device).unsqueeze(1)
    T_tensor = torch.tensor(T_norm, dtype=torch.float32, device=device).unsqueeze(1)
    
    # 模型预测
    model.eval()
    with torch.no_grad():
        # 将x, y, t拼接成一个输入张量，模型只接受一个参数
        inputs = torch.cat([X_tensor, Y_tensor, T_tensor], dim=1)
        output = model(inputs)
        u_norm = output[:, 0].cpu().numpy()
        v_norm = output[:, 1].cpu().numpy()
        
        # 反归一化速度
        u_pred, v_pred = denormalize_velocity(u_norm, v_norm)
        # 重塑为网格形状 - 使用reshape保持原有顺序
        u_pred = u_pred.reshape(X.shape)  # 与X, Y相同的形状
        v_pred = v_pred.reshape(Y.shape)  # 与X, Y相同的形状
        speed = np.sqrt(u_pred**2 + v_pred**2)
    
    # 绘制流线图
    plt.figure(figsize=(12, 10))
    
    # 速度大小背景
    im = plt.contourf(X, Y, speed, 50, cmap='jet', alpha=0.8)
    plt.colorbar(im, label='速度大小 (m/s)')
    
    # 流线图
    plt.streamplot(X, Y, u_pred, v_pred, color='white', linewidth=1, density=2, arrowsize=1)
    
    plt.xlim(config.norm_params['x_min'], config.norm_params['x_max'])
    plt.ylim(config.norm_params['y_min'], config.norm_params['y_max'])
    plt.xlabel('x坐标 (mm)')
    plt.ylabel('y坐标 (mm)')
    plt.title(f'2D流场 t={t_value:.1f}s (epoch={epoch})')
    plt.gca().set_aspect('equal', adjustable='box')
    
    # 保存图片
    filename = f'flow_field_t{t_value:.1f}_epoch{epoch}.png'
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"保存2D流场图: {filepath}")

# 训练函数 - 使用双模态协同训练系统
def train_2d_pinn():
    """训练2D PINN模型（调用双模态协同训练系统）"""
    
    print("[启用] 双模态协同训练系统")
    print("[配置] MambaInput=4D(xyuv) | Weight=100 | Sample=All")
    return train_dual_model_system()


def get_mamba_time_strategy_weights(t_batch):
    """
    优化后的Mamba约束权重 - 解决损失波动问题（适配0-5s时间范围）
    
    Args:
        t_batch: 时间张量（归一化到[-1,1]范围）
        
    Returns:
        weights: 与t_batch相同形状的权重张量
    """
    # 反归一化时间到物理空间 [0.0, 5.0]
    t_physical = (t_batch + 1.0) / 2.0 * 5.0 + 0.0
    
    # 初始化权重张量
    weights = torch.zeros_like(t_physical)
    
    # 优化后的权重设置 - 更平滑的过渡
    # 0.0-2.0s: 基础Mamba约束（权重从0.05渐变到0.15）
    mask_early = (t_physical >= 0.0) & (t_physical < 2.0)
    early_weights = 0.05 + 0.1 * (t_physical - 0.0) / 2.0
    weights = torch.where(mask_early, early_weights, weights)
    
    # 2.0-4.0s: 中等Mamba约束（权重从0.15渐变到0.3）
    mask_mid = (t_physical >= 2.0) & (t_physical < 4.0)
    mid_weights = 0.15 + 0.15 * (t_physical - 2.0) / 2.0
    weights = torch.where(mask_mid, mid_weights, weights)
    
    # 4.0-5.0s: 增强Mamba约束（权重从0.3渐变到0.5）
    mask_late = (t_physical >= 4.0) & (t_physical <= 5.0)
    late_weights = 0.3 + 0.2 * (t_physical - 4.0) / 1.0
    weights = torch.where(mask_late, late_weights, weights)
    
    # 确保最小权重不为零
    weights = torch.clamp(weights, min=0.01, max=0.5)
    
    return weights

def mamba_temporal_consistency_loss(model, x, y, t_current, t_previous):
    """优化后的Mamba时序一致性损失 - 增强数值稳定性"""
    
    try:
        # 当前时间步预测 - 将x, y, t拼接成一个输入张量，模型只接受一个参数
        inputs_current = torch.cat([x, y, t_current], dim=1)
        output_current = model(inputs_current)
        
        # 前一时间步预测 - 将x, y, t拼接成一个输入张量，模型只接受一个参数
        inputs_previous = torch.cat([x, y, t_previous], dim=1)
        output_previous = model(inputs_previous)
        
        # 数值稳定性检查
        if torch.isnan(output_current).any() or torch.isinf(output_current).any() or \
           torch.isnan(output_previous).any() or torch.isinf(output_previous).any():
            return torch.tensor(1e-6, device=x.device, requires_grad=True)
        
        # 直接提取速度分量（模型现在输出u,v而不是ψ,p）
        u_current_norm = output_current[:, 0:1]
        v_current_norm = output_current[:, 1:2]
        u_previous_norm = output_previous[:, 0:1]
        v_previous_norm = output_previous[:, 1:2]
        
        # 数值裁剪，避免极端值
        max_norm = 10.0
        u_current_norm = torch.clamp(u_current_norm, min=-max_norm, max=max_norm)
        v_current_norm = torch.clamp(v_current_norm, min=-max_norm, max=max_norm)
        u_previous_norm = torch.clamp(u_previous_norm, min=-max_norm, max=max_norm)
        v_previous_norm = torch.clamp(v_previous_norm, min=-max_norm, max=max_norm)
        
        # 获取归一化参数
        u_min, u_max = config.norm_params['u_min'], config.norm_params['u_max']
        v_min, v_max = config.norm_params['v_min'], config.norm_params['v_max']
        
        # 反归一化到物理空间
        u_current = u_current_norm * (u_max - u_min)/2 + (u_max + u_min)/2
        v_current = v_current_norm * (v_max - v_min)/2 + (v_max + v_min)/2
        u_previous = u_previous_norm * (u_max - u_min)/2 + (u_max + u_min)/2
        v_previous = v_previous_norm * (v_max - v_min)/2 + (v_max + v_min)/2
        
        # 将归一化时间转换为物理时间
        t_current_phys = (t_current + 1.0) / 2.0 * 5.0 + 0.0
        t_previous_phys = (t_previous + 1.0) / 2.0 * 5.0 + 0.0
        time_diff = t_current_phys - t_previous_phys
        
        # 避免除零和过小时间差
        time_diff = torch.clamp(time_diff, min=1e-4, max=1.0)
        
        # 物理空间的时间导数
        temporal_derivative_u = (u_current - u_previous) / time_diff
        temporal_derivative_v = (v_current - v_previous) / time_diff
        
        # 速度变化限制
        max_velocity_change = 5.0
        du = torch.clamp(u_current - u_previous, min=-max_velocity_change, max=max_velocity_change)
        dv = torch.clamp(v_current - v_previous, min=-max_velocity_change, max=max_velocity_change)
        
        # 基础时序一致性损失（物理空间）
        consistency_loss = torch.mean(temporal_derivative_u ** 2 + temporal_derivative_v ** 2)
        
        # 速度变化率约束（降低权重）
        velocity_change_constraint = torch.mean(du ** 2 + dv ** 2)
        
        # 能量守恒约束（使用相对变化）
        energy_current = torch.mean(u_current ** 2 + v_current ** 2)
        energy_previous = torch.mean(u_previous ** 2 + v_previous ** 2)
        energy_base = torch.maximum(energy_previous, torch.tensor(1e-6, device=energy_previous.device))
        energy_change = torch.abs(energy_current - energy_previous) / energy_base
        
        # 组合所有约束（降低权重避免主导）
        enhanced_consistency_loss = consistency_loss + \
                                   0.05 * velocity_change_constraint + \
                                   0.001 * energy_change
        
        # 最终数值检查
        if torch.isnan(enhanced_consistency_loss) or torch.isinf(enhanced_consistency_loss):
            return torch.tensor(1e-6, device=x.device, requires_grad=True)
        
        return enhanced_consistency_loss
        
    except Exception as e:
        print(f"Mamba时序一致性损失计算异常: {e}")
        return torch.tensor(1e-6, device=x.device, requires_grad=True)


def load_temporal_data_range(t_start, t_end, max_points_per_time=12800):
    """
    加载指定时间范围内的所有时序数据（适配data2文件夹格式）
    
    Args:
        t_start: 开始时间
        t_end: 结束时间
        max_points_per_time: 每个时间点最多加载的数据点数
        
    Returns:
        x_all, y_all, t_all, u_all, v_all: 合并后的所有数据
    """
    import pandas as pd
    import numpy as np
    
    # 存储所有数据
    x_list, y_list, t_list, u_list, v_list = [], [], [], [], []
    
    # 遍历时间范围（步长0.01s）
    t_current = t_start
    while t_current <= t_end:
        try:
            # 格式化时间字符串（两位小数，匹配data2文件格式）
            time_str = f"{t_current:.2f}"
            data_file = f'data2/sigle_t={time_str}s.txt'  # 修改为data2文件夹
            
            # 读取数据
            data = pd.read_csv(data_file, sep=r'\s+')
            
            # 限制每个时间点的数据点数
            n_points = min(len(data), max_points_per_time)
            if n_points > 0:
                # 随机采样或取前n_points个点
                if len(data) > n_points:
                    data = data.sample(n=n_points, random_state=42)
                
                # 转换为米单位（mm→m）
                x_list.extend(data['x(mm)'].values / 1000.0)  # mm→m
                y_list.extend(data['y(mm)'].values / 1000.0)  # mm→m
                t_list.extend([t_current] * n_points)
                u_list.extend(data['u(m/s)'].values)
                v_list.extend(data['v(m/s)'].values)
                
        except Exception as e:
            print(f"[警告] 加载时刻{t_current}s数据失败: {e}")
        
        # 时间步进
        t_current += 0.01
        t_current = round(t_current, 3)  # 避免浮点精度问题
    
    # 转换为numpy数组
    x_all = np.array(x_list)
    y_all = np.array(y_list)
    t_all = np.array(t_list)
    u_all = np.array(u_list)
    v_all = np.array(v_list)
    
    return x_all, y_all, t_all, u_all, v_all


def sample_spatiotemporal_points_uniform(batch_size, device, time_range=(0.0, 5.0), space_bounds=None):
    """
    均匀采样时空点（网格采样）- 优化版：预缓存全域网格点，从缓存中采样
    
    Args:
        batch_size: 批次大小
        device: PyTorch设备
        time_range: 时间范围 (t_start, t_end)
        space_bounds: 空间边界 (x_min, x_max, y_min, y_max)，如果为None则使用config中的边界
        
    Returns:
        x_batch, y_batch, t_batch, boundary_types: 归一化后的时空坐标张量和边界类型
    """
    import numpy as np
    
    # 使用config中的空间边界
    if space_bounds is None:
        x_min, x_max = config.X_MIN, config.X_MAX
        y_min, y_max = config.Y_MIN, config.Y_MAX
    else:
        x_min, x_max, y_min, y_max = space_bounds
    
    t_start, t_end = time_range
    
    # 预缓存全域网格点（与数据分辨率一致）
    if not hasattr(sample_spatiotemporal_points_uniform, '_cached_all_points'):
        print("[优化] 首次调用：预缓存全域网格点...")
        
        # 根据数据文件分辨率设置网格分辨率
        # 数据文件：x范围5025-12975（间隔50mm），y范围2025-8025（间隔50mm），时间范围0-5s（间隔0.01s）
        n_x = 160  # (12975-5025)/50 + 1 = 7950/50 + 1 = 159 + 1 = 160
        n_y = 121  # (8025-2025)/50 + 1 = 6000/50 + 1 = 120 + 1 = 121  
        n_t = 501  # (5.0-0.0)/0.01 + 1 = 500 + 1 = 501
        
        # 生成均匀网格点（与数据文件完全一致的分辨率）
        x_grid = np.linspace(x_min, x_max, n_x)
        y_grid = np.linspace(y_min, y_max, n_y)
        t_grid = np.linspace(t_start, t_end, n_t)
        
        # 创建网格
        x_mesh, y_mesh, t_mesh = np.meshgrid(x_grid, y_grid, t_grid, indexing='ij')
        
        # 展平
        x_all = x_mesh.flatten()
        y_all = y_mesh.flatten()
        t_all = t_mesh.flatten()
        
        # 动态确定边界类型
        boundary_types = np.zeros(len(x_all), dtype=int)
        tol = 1e-6
        
        # 检查每个点是否在边界上
        on_bottom = np.abs(y_all - y_min) < tol
        on_top = np.abs(y_all - y_max) < tol
        on_left = np.abs(x_all - x_min) < tol
        on_right = np.abs(x_all - x_max) < tol
        
        # 分配边界类型
        boundary_types[on_bottom] = 1
        boundary_types[on_top & ~on_bottom] = 2
        boundary_types[on_left & ~on_bottom & ~on_top] = 3
        boundary_types[on_right & ~on_bottom & ~on_top & ~on_left] = 4
        
        # 归一化到[-1, 1]
        x_norm = 2.0 * (x_all - x_min) / (x_max - x_min) - 1.0
        y_norm = 2.0 * (y_all - y_min) / (y_max - y_min) - 1.0
        t_norm = 2.0 * (t_all - t_start) / (t_end - t_start) - 1.0
        
        # 缓存所有数据点
        sample_spatiotemporal_points_uniform._cached_all_points = {
            'x_norm': x_norm, 'y_norm': y_norm, 't_norm': t_norm, 'boundary_types': boundary_types
        }
        print(f"[完成] 预缓存完成：共{len(x_all)}个全域网格点（{n_x}x{n_y}x{n_t}），与数据分辨率一致")
    
    # 从缓存中随机采样batch_size个点
    cached_data = sample_spatiotemporal_points_uniform._cached_all_points
    total_points = len(cached_data['x_norm'])
    
    if total_points > batch_size:
        indices = np.random.choice(total_points, batch_size, replace=False)
    else:
        indices = np.random.choice(total_points, batch_size, replace=True)
    
    # 提取采样点
    x_norm_batch = cached_data['x_norm'][indices]
    y_norm_batch = cached_data['y_norm'][indices]
    t_norm_batch = cached_data['t_norm'][indices]
    boundary_batch = cached_data['boundary_types'][indices]
    
    # 转换为PyTorch张量并指定设备
    x_tensor = torch.tensor(x_norm_batch, dtype=torch.float32).view(-1, 1).to(device)
    y_tensor = torch.tensor(y_norm_batch, dtype=torch.float32).view(-1, 1).to(device)
    t_tensor = torch.tensor(t_norm_batch, dtype=torch.float32).view(-1, 1).to(device)
    boundary_tensor = torch.tensor(boundary_batch, dtype=torch.float32).view(-1, 1).to(device)
    
    return x_tensor, y_tensor, t_tensor, boundary_tensor
def sample_spatiotemporal_data_points_uniform(batch_size, time_range=(0.0, 5.0)):
    """
    从真实数据中均匀采样时空点（用于数据损失计算）- 优化版：预缓存所有数据点，从缓存中采样
    
    Args:
        batch_size: 批次大小
        time_range: 时间范围 (t_start, t_end)，默认0.0-5.0s（有真实数据的范围）
        
    Returns:
        x_batch, y_batch, t_batch, u_batch, v_batch: 时空坐标和速度张量，如果没有数据则返回None
    """
    import numpy as np
    
    # 预加载所有可用数据并归一化缓存
    if not hasattr(sample_spatiotemporal_data_points_uniform, '_cached_all_data'):
        print("[优化] 首次调用：预缓存所有数据点并归一化...")
        # 加载0-5s范围的真实数据（data2文件夹）
        t_start, t_end = 0.0, 5.0
        x_all, y_all, t_all, u_all, v_all = load_temporal_data_range(t_start, t_end, max_points_per_time=12800)
        
        if x_all is not None and len(x_all) > 0:
            # 归一化到[-1, 1]
            x_norm = 2.0 * (x_all - config.X_MIN) / (config.X_MAX - config.X_MIN) - 1.0
            y_norm = 2.0 * (y_all - config.Y_MIN) / (config.Y_MAX - config.Y_MIN) - 1.0
            t_norm = 2.0 * (t_all - time_range[0]) / (time_range[1] - time_range[0]) - 1.0
            
            # 缓存归一化后的所有数据点
            sample_spatiotemporal_data_points_uniform._cached_all_data = {
                'x_norm': x_norm, 'y_norm': y_norm, 't_norm': t_norm, 'u': u_all, 'v': v_all
            }
            print(f"[完成] 预缓存完成：共{len(x_all)}个数据点")
        else:
            print("[警告] 无可用时序数据")
            return None, None, None, None, None
    
    cached_data = sample_spatiotemporal_data_points_uniform._cached_all_data
    x_norm_all = cached_data['x_norm']
    y_norm_all = cached_data['y_norm']
    t_norm_all = cached_data['t_norm']
    u_all = cached_data['u']
    v_all = cached_data['v']
    
    if len(x_norm_all) == 0:
        return None, None, None, None, None
    
    # 从预缓存中随机采样batch_size个点
    total_points = len(x_norm_all)
    
    if total_points > batch_size:
        indices = np.random.choice(total_points, batch_size, replace=False)
    else:
        indices = np.random.choice(total_points, batch_size, replace=True)
    
    # 提取采样点
    x_norm_batch = x_norm_all[indices]
    y_norm_batch = y_norm_all[indices]
    t_norm_batch = t_norm_all[indices]
    u_batch = u_all[indices]
    v_batch = v_all[indices]
    
    # 转换为PyTorch张量并指定设备
    x_tensor = torch.tensor(x_norm_batch, dtype=torch.float32).view(-1, 1).to(device)
    y_tensor = torch.tensor(y_norm_batch, dtype=torch.float32).view(-1, 1).to(device)
    t_tensor = torch.tensor(t_norm_batch, dtype=torch.float32).view(-1, 1).to(device)
    u_tensor = torch.tensor(u_batch, dtype=torch.float32).view(-1, 1).to(device)
    v_tensor = torch.tensor(v_batch, dtype=torch.float32).view(-1, 1).to(device)
    
    return x_tensor, y_tensor, t_tensor, u_tensor, v_tensor



    """计算R²评分（支持w速度）"""
    if x_data is None or len(x_data) == 0:
        return 0.0, 0.0, 0.0
    
    # 归一化（修正参数数量匹配问题）
    if w_data is not None:
        x_norm, y_norm, u_norm, v_norm, w_norm, t_norm = normalize_data(x_data, y_data, u_data, v_data, w_data, t_data)
    else:
        x_norm, y_norm, u_norm, v_norm, t_norm = normalize_data(x_data, y_data, u_data, v_data, t_data)
        w_norm = None
    
    # 转换为张量
    x_tensor = torch.tensor(x_norm, dtype=torch.float32).view(-1, 1).to(device)
    y_tensor = torch.tensor(y_norm, dtype=torch.float32).view(-1, 1).to(device)
    t_tensor = torch.tensor(t_norm, dtype=torch.float32).view(-1, 1).to(device)
    
    # 预测
    with torch.no_grad():
        # 将x, y, t合并成一个输入张量
        inputs = torch.cat([x_tensor, y_tensor, t_tensor], dim=1)
        outputs = model(inputs)
        u_pred_norm = outputs[:, 0].cpu().numpy()
        v_pred_norm = outputs[:, 1].cpu().numpy()
        
        # 反归一化速度
        u_pred, v_pred = denormalize_velocity(u_pred_norm, v_pred_norm)
        
        # 计算R²
        u_r2 = r2_score(u_data, u_pred)
        v_r2 = r2_score(v_data, v_pred)
        speed_r2 = r2_score(np.sqrt(u_data**2 + v_data**2), np.sqrt(u_pred**2 + v_pred**2))
        
        # 返回速度场R²评分
        return u_r2, v_r2, speed_r2
    
    return u_r2, v_r2, speed_r2


def sample_spatiotemporal_points_from_real_data(batch_size, device, time_range=(0.00, 5.00)):
    """
    从真实数据文件中读取xy坐标，随机匹配0.00-5.00s时间坐标的物理采样
    
    Args:
        batch_size: 批次大小
        device: PyTorch设备
        time_range: 时间范围 (t_start, t_end)，默认0.00-5.00s
        
    Returns:
        x_batch, y_batch, t_batch, boundary_types: 归一化后的时空坐标张量和边界类型
    """
    import numpy as np
    
    # 读取t=0.00s的数据文件获取xy坐标
    file_path = os.path.join('data2', f'sigle_t={0.00:.2f}s.txt')
    
    try:
        data = pd.read_csv(file_path, sep=r'\s+')
        # 提取xy坐标（mm转m）
        x_coords = data['x(mm)'].values / 1000.0  # mm -> m
        y_coords = data['y(mm)'].values / 1000.0  # mm -> m
        
        # 随机选择xy坐标
        indices = np.random.choice(len(x_coords), size=min(batch_size, len(x_coords)), replace=False)
        x_batch = x_coords[indices]
        y_batch = y_coords[indices]
        
    except FileNotFoundError:
        print(f"[警告] 文件 {file_path} 不存在，使用默认空间范围")
        x_min, x_max = config.X_MIN, config.X_MAX
        y_min, y_max = config.Y_MIN, config.Y_MAX
        x_batch = np.random.uniform(x_min, x_max, batch_size)
        y_batch = np.random.uniform(y_min, y_max, batch_size)
    
    # 从真实数据时间步中随机选择时间坐标（步长0.01s）
    t_start, t_end = time_range
    available_times = np.arange(t_start, t_end + 0.01, 0.01)  # 0.01s步长
    t_batch = np.random.choice(available_times, size=batch_size, replace=True)
    
    # 动态确定边界类型
    boundary_types = np.zeros(batch_size, dtype=int)
    
    # 定义边界容差
    tol = 1e-6
    
    # 使用config中的空间边界
    x_min, x_max = config.X_MIN, config.X_MAX
    y_min, y_max = config.Y_MIN, config.Y_MAX
    
    # 检查边界
    on_bottom = np.abs(y_batch - y_min) < tol
    on_top = np.abs(y_batch - y_max) < tol
    on_left = np.abs(x_batch - x_min) < tol
    on_right = np.abs(x_batch - x_max) < tol
    
    # 分配边界类型（优先级：底部 > 顶部 > 左侧 > 右侧）
    boundary_types[on_bottom] = 1
    boundary_types[on_top & ~on_bottom] = 2
    boundary_types[on_left & ~on_bottom & ~on_top] = 3
    boundary_types[on_right & ~on_bottom & ~on_top & ~on_left] = 4
    
    # 归一化到[-1, 1]
    x_norm = 2.0 * (x_batch - x_min) / (x_max - x_min) - 1.0
    y_norm = 2.0 * (y_batch - y_min) / (y_max - y_min) - 1.0
    t_norm = 2.0 * (t_batch - t_start) / (t_end - t_start) - 1.0
    
    # 转换为PyTorch张量并指定设备
    x_tensor = torch.tensor(x_norm, dtype=torch.float32).view(-1, 1).to(device)
    y_tensor = torch.tensor(y_norm, dtype=torch.float32).view(-1, 1).to(device)
    t_tensor = torch.tensor(t_norm, dtype=torch.float32).view(-1, 1).to(device)
    boundary_tensor = torch.tensor(boundary_types, dtype=torch.float32).view(-1, 1).to(device)
    
    return x_tensor, y_tensor, t_tensor, boundary_tensor



    """计算MAE评分（支持w速度）"""
    if x_data is None or len(x_data) == 0:
        if w_data is not None:
            return 0.0, 0.0, 0.0, 0.0
        return 0.0, 0.0, 0.0
    
    # 归一化（修正参数数量匹配问题）
    if w_data is not None:
        x_norm, y_norm, u_norm, v_norm, w_norm, t_norm = normalize_data(x_data, y_data, u_data, v_data, w_data, t_data)
    else:
        x_norm, y_norm, u_norm, v_norm, t_norm = normalize_data(x_data, y_data, u_data, v_data, t_data)
        w_norm = None
    
    # 转换为张量
    x_tensor = torch.tensor(x_norm, dtype=torch.float32).view(-1, 1).to(device)
    y_tensor = torch.tensor(y_norm, dtype=torch.float32).view(-1, 1).to(device)
    t_tensor = torch.tensor(t_norm, dtype=torch.float32).view(-1, 1).to(device)
    
    # 预测
    with torch.no_grad():
        # 将x, y, t合并成一个输入张量
        inputs = torch.cat([x_tensor, y_tensor, t_tensor], dim=1)
        outputs = model(inputs)
        u_pred_norm = outputs[:, 0].cpu().numpy()
        v_pred_norm = outputs[:, 1].cpu().numpy()
        
        # 反归一化速度
        if w_data is not None:
            w_pred_norm = outputs[:, 2].cpu().numpy()
            u_pred, v_pred, w_pred = denormalize_velocity(u_pred_norm, v_pred_norm, w_pred_norm)
            # 计算w MAE
            w_mae = np.mean(np.abs(w_data - w_pred))
        else:
            u_pred, v_pred = denormalize_velocity(u_pred_norm, v_pred_norm)
            w_mae = 0.0
        
        # 计算MAE
        u_mae = np.mean(np.abs(u_data - u_pred))
        v_mae = np.mean(np.abs(v_data - v_pred))
        speed_mae = np.mean(np.abs(np.sqrt(u_data**2 + v_data**2) - np.sqrt(u_pred**2 + v_pred**2)))
    
    if w_data is not None:
        return u_mae, v_mae, w_mae, speed_mae
    else:
        return u_mae, v_mae, speed_mae


        # 对mamba_consistency_loss_val进行均值化处理
        mamba_consistency_loss_val = mamba_consistency_loss_val.mean()
        
        # 调试信息：检查Mamba损失计算详情
        if epoch % 100 == 0:  # 每100轮打印一次调试信息
            avg_time_debug = (t_batch.mean().item() + 1.0) / 2.0 * 10.0 + 5.0
            print(f"  [Mamba调试] epoch={epoch}, t={avg_time_debug:.2f}s, weight={mamba_weights.mean().item():.4f}, loss={mamba_consistency_loss_val.item():.6f}")
        
        # 调试信息：检查连续性损失（增强精度显示）
        if epoch % 100 == 0:
            print(f"  [Continuity调试] epoch={epoch}, continuity_loss={continuity_loss.item():.10f}, 连续性残差范围: [{continuity.min().item():.6f}, {continuity.max().item():.6f}], 残差标准差: {continuity.std().item():.6f}")
            # 检查连续性损失是否异常
            if continuity_loss.item() < 1e-10:
                print(f"    [警告] 连续性损失过小，可能存在问题！u_x范围: [{u_x.min().item():.6f}, {u_x.max().item():.6f}], v_y范围: [{v_y.min().item():.6f}, {v_y.max().item():.6f}]")
        
        # 总损失
        total_loss = physics_loss + boundary_loss_val + data_loss + mamba_consistency_loss_val
        
        # 确保total_loss是标量
        if total_loss.dim() > 0:
            total_loss = total_loss.sum()
        
        # 反向传播
        total_loss.backward()
        
        # 如果有T数据，记录T指标
        if has_T_data:
            history['epoch_r2_T'].append(T_r2)
            history['epoch_mae_T'].append(T_mae)
        
        # 定期报告（更新打印格式，显示所有损失项）
        if (epoch - n_epochs) % args.report_every == 0:
            print(f"Iter {epoch - n_epochs:4d}: Total={total_loss.item():.6f}, "
                  f"Physics={physics_loss.item():.6f}, "
                  f"Boundary={boundary_loss_val.item():.6f}, "
                  f"Data={data_loss:.6f}, "
                  f"Mamba={mamba_consistency_loss_val:.6f}")
            
            # 如果同时是R²/MAE计算周期，打印指标
            if (epoch - n_epochs) % 100 == 0:
                if has_w_data and has_T_data:
                    print(f"  R²指标 - u: {u_r2:.4f}, v: {v_r2:.4f}, w: {w_r2:.4f}, speed: {speed_r2:.4f}, T: {T_r2:.4f}")
                    print(f"  MAE指标 - u: {u_mae:.6f}, v: {v_mae:.6f}, w: {w_mae:.6f}, speed: {speed_mae:.6f}, T: {T_mae:.6f}")
                elif has_w_data:
                    print(f"  R²指标 - u: {u_r2:.4f}, v: {v_r2:.4f}, speed: {speed_r2:.4f}")
                    print(f"  MAE指标 - u: {u_mae:.6f}, v: {v_mae:.6f}, w: {w_mae:.6f}, speed: {speed_mae:.6f}")
                elif has_T_data:
                    print(f"  R²指标 - u: {u_r2:.4f}, v: {v_r2:.4f}, speed: {speed_r2:.4f}, T: {T_r2:.4f}")
                    print(f"  MAE指标 - u: {u_mae:.6f}, v: {v_mae:.6f}, speed: {speed_mae:.6f}, T: {T_mae:.6f}")
                else:
                    print(f"  R²指标 - u: {u_r2:.4f}, v: {v_r2:.4f}, speed: {speed_r2:.4f}")
                    print(f"  MAE指标 - u: {u_mae:.6f}, v: {v_mae:.6f}, speed: {speed_mae:.6f}")
        
        # 移除错误的return语句，让循环正常结束
        
        # 定期绘图 - 生成热图动画（每1000轮绘制一次，第0轮不绘制）
        if epoch % config.PLOT_INTERVAL_ANIMATION == 0 and epoch > 0:
            # 生成流场动画（类似flow_field_t*.png样式）
            from visualize import generate_heatmap_animation
            generate_heatmap_animation(
                model, config.norm_params, save_dir, 
                time_range=config.ANIMATION_TIME_RANGE, 
                num_frames=config.ANIMATION_NUM_FRAMES, 
                grid_size=config.ANIMATION_GRID_SIZE, 
                epoch=epoch
            )
            
            # 生成速度差热图（与原始数据比较）
            if x_data is not None:
                from visualize import plot_velocity_difference_heatmap
                plot_velocity_difference_heatmap(model, config.VELOCITY_DIFF_TIME, save_dir, epoch)

    # 训练完成后立即保存模型和训练历史（添加异常处理）
    print("2D PINN训练完成!")

    # 创建目录结构说明
    create_directory_structure(save_dir)

    # 保存详细的评估结果到JSON文件
    print("📊 保存详细评估结果...")
    try:
        # 计算一些基本评估指标
        eval_results = {
            'overall_metrics': {
                'final_total_loss': float(history['total_loss'][-1]) if history['total_loss'] else None,
                'final_physics_loss': float(history['physics_loss'][-1]) if history['physics_loss'] else None,
                'final_boundary_loss': float(history['boundary_loss'][-1]) if history['boundary_loss'] else None,
                'final_data_loss': float(history['data_loss'][-1]) if history['data_loss'] else None,
                'training_epochs': len(history['total_loss']),
                'learning_rate': float(history['learning_rate'][-1]) if history['learning_rate'] else None
            },
            'loss_statistics': {
                'total_loss': {
                    'mean': float(np.mean(history['total_loss'])) if history['total_loss'] else None,
                    'std': float(np.std(history['total_loss'])) if history['total_loss'] else None,
                    'min': float(np.min(history['total_loss'])) if history['total_loss'] else None,
                    'max': float(np.max(history['total_loss'])) if history['total_loss'] else None,
                    'length': len(history['total_loss'])
                },
                'physics_loss': {
                    'mean': float(np.mean(history['physics_loss'])) if history['physics_loss'] else None,
                    'std': float(np.std(history['physics_loss'])) if history['physics_loss'] else None,
                    'min': float(np.min(history['physics_loss'])) if history['physics_loss'] else None,
                    'max': float(np.max(history['physics_loss'])) if history['physics_loss'] else None,
                    'length': len(history['physics_loss'])
                },
                'boundary_loss': {
                    'mean': float(np.mean(history['boundary_loss'])) if history['boundary_loss'] else None,
                    'std': float(np.std(history['boundary_loss'])) if history['boundary_loss'] else None,
                    'min': float(np.min(history['boundary_loss'])) if history['boundary_loss'] else None,
                    'max': float(np.max(history['boundary_loss'])) if history['boundary_loss'] else None,
                    'length': len(history['boundary_loss'])
                },
                'data_loss': {
                    'mean': float(np.mean(history['data_loss'])) if history['data_loss'] else None,
                    'std': float(np.std(history['data_loss'])) if history['data_loss'] else None,
                    'min': float(np.min(history['data_loss'])) if history['data_loss'] else None,
                    'max': float(np.max(history['data_loss'])) if history['data_loss'] else None,
                    'length': len(history['data_loss'])
                }
            }
        }
        
        # 保存到JSON文件
        import json
        eval_path = os.path.join(save_dir, 'evaluation_results.json')
        with open(eval_path, 'w') as f:
            json.dump(eval_results, f, indent=2)
        print(f"✅ 评估结果已保存: {eval_path}")
        
    except Exception as e:
        print(f"❌ 保存评估结果时出错: {e}")

    # 生成完整的模型综合评价图
    print("🎯 开始生成模型综合评价图...")
    try:
        from visualize import generate_comprehensive_evaluation_plots
        evaluation_plots = generate_comprehensive_evaluation_plots(
            model=model,
            history=history, 
            norm_params=config.norm_params,
            save_path=os.path.join(save_dir, 'evaluation_plots')
        )
        print(f"[成功] 生成 {len(evaluation_plots)} 张综合评价图")
    except Exception as e:
        print(f"[错误] 生成综合评价图时出错: {e}")
        print(f"[提示] 可以手动运行 visualize.generate_comprehensive_evaluation_plots() 来生成评价图")

    # 生成训练好后的校准图（Parity Plot）
    print("📊 开始生成校准图（Parity Plot）...")
    try:
        generate_parity_plot(model, save_dir, sample_size=2000)
        print("✅ 校准图生成完成")
    except Exception as e:
        print(f"❌ 生成校准图时出错: {e}")


def calculate_r2_and_mae(model, data_0_5=None, data_5_10=None, use_cached_tensors=False):
    """
    带深度调试功能的评估函数
    """
    import numpy as np
    import torch
    from sklearn.metrics import r2_score, mean_absolute_error
    
    model.eval()
    device = next(model.parameters()).device
    results = {}
    
    def compute_metrics(data_dict, prefix=""):
        if data_dict is None: return {}
            
        # 1. 获取输入
        if use_cached_tensors:
            x_tensor = data_dict['x_norm']
            y_tensor = data_dict['y_norm']
            t_tensor = data_dict['t_norm']
            u_true = data_dict['u_phys'] # 真实物理值
            v_true = data_dict['v_phys'] # 真实物理值
        else:
            # (省略非缓存逻辑，保持原有代码即可，重点是下面)
            pass 
            
        # 2. 模型预测
        with torch.no_grad():
            inputs = torch.cat([x_tensor, y_tensor, t_tensor], dim=1)
            outputs = model(inputs)
            u_pred_norm = outputs[:, 0:1].cpu().numpy().flatten()
            v_pred_norm = outputs[:, 1:2].cpu().numpy().flatten()
            
        # 3. 反归一化
        u_pred, v_pred = denormalize_velocity(u_pred_norm, v_pred_norm)
        # 5. 安全计算 R2 (防止 NaN)
        try:
            metric_u_r2 = r2_score(u_true, u_pred)
            metric_v_r2 = r2_score(v_true, v_pred)
            speed_true = np.sqrt(u_true**2 + v_true**2)
            speed_pred = np.sqrt(u_pred**2 + v_pred**2)
            metric_speed_r2 = r2_score(speed_true, speed_pred)
        except Exception as e:
            print(f"⚠️ R2计算出错: {e}")
            metric_u_r2, metric_v_r2, metric_speed_r2 = -999, -999, -999

        return {
            f'{prefix}u_r2': metric_u_r2, f'{prefix}v_r2': metric_v_r2, f'{prefix}speed_r2': metric_speed_r2,
            f'{prefix}u_mae': mean_absolute_error(u_true, u_pred),
            f'{prefix}v_mae': mean_absolute_error(v_true, v_pred),
            f'{prefix}speed_mae': mean_absolute_error(speed_true, speed_pred)
        }

    results.update(compute_metrics(data_0_5, prefix=""))
    results.update(compute_metrics(data_5_10, prefix="extrap_"))
    return results


def update_training_history_with_r2_mae(history, epoch, r2_mae_results):
    """
    更新训练历史记录，添加R²和MAE数据
    
    Args:
        history: 训练历史字典
        epoch: 当前轮次
        r2_mae_results: R²和MAE计算结果
    """
    # 初始化R²和MAE记录列表
    if 'epoch_r2_u' not in history:
        history['epoch_r2_u'] = []
        history['epoch_r2_v'] = []
        history['epoch_r2_speed'] = []
        history['epoch_mae_u'] = []
        history['epoch_mae_v'] = []
        history['epoch_mae_speed'] = []
        history['epoch_r2_timestamps'] = []
    
    # 添加当前轮次的R²和MAE数据
    history['epoch_r2_u'].append(r2_mae_results['u_r2'])
    history['epoch_r2_v'].append(r2_mae_results['v_r2'])
    history['epoch_r2_speed'].append(r2_mae_results['speed_r2'])
    history['epoch_mae_u'].append(r2_mae_results['u_mae'])
    history['epoch_mae_v'].append(r2_mae_results['v_mae'])
    history['epoch_mae_speed'].append(r2_mae_results['speed_mae'])
    history['epoch_r2_timestamps'].append(epoch)
    
    return history





def boundary_prediction_constraint(model, x_tensor, y_tensor, t_tensor):
    """边界预测约束函数，防止预测发散
    
    Args:
        model: PINN模型
        x_tensor: x坐标张量
        y_tensor: y坐标张量
        t_tensor: 时间张量
        
    Returns:
        constraint_loss: 边界约束损失
    """
    # 维度检查：确保所有输入张量都是2维的
    if x_tensor.dim() == 1:
        x_tensor = x_tensor.unsqueeze(1)
    if y_tensor.dim() == 1:
        y_tensor = y_tensor.unsqueeze(1)
    if t_tensor.dim() == 1:
        t_tensor = t_tensor.unsqueeze(1)
    
    # 获取模型预测
    inputs = torch.cat([x_tensor, y_tensor, t_tensor], dim=1)
    predictions = model(inputs)
    
    # 关键修复：将归一化预测值反归一化到物理空间
    # 获取归一化参数
    u_min, u_max = config.norm_params['u_min'], config.norm_params['u_max']
    v_min, v_max = config.norm_params['v_min'], config.norm_params['v_max']
    
    # 提取归一化速度分量
    u_norm = predictions[:, 0]
    v_norm = predictions[:, 1]
    
    # 反归一化到物理空间（m/s）
    u_actual = u_norm * (u_max - u_min)/2 + (u_max + u_min)/2
    v_actual = v_norm * (v_max - v_min)/2 + (v_max + v_min)/2
    
    # 边界约束：限制速度值在合理范围内（物理空间）
    # 设置最大速度阈值（例如10 m/s）
    max_velocity = 10.0
    
    # 计算超出阈值的惩罚（物理空间）
    u_exceed = torch.relu(torch.abs(u_actual) - max_velocity)
    v_exceed = torch.relu(torch.abs(v_actual) - max_velocity)
    
    # 约束损失 = 超出阈值的平方和
    constraint_loss = torch.mean(u_exceed**2 + v_exceed**2)
    
    # 添加归一化因子，确保数值范围一致
    constraint_loss = constraint_loss * 1e-6
    
    return constraint_loss

def unified_data_loss(model, t_physical, x_init=None, y_init=None, t_init=None, u_init=None, v_init=None, 
                       data_weight=100.0):
    """简化数据损失函数：直接计算MSE损失"""
    
    # 检查是否有可用的批次数据
    if x_init is not None and len(x_init) > 0:
        # 维度检查：确保所有输入张量都是2维的
        if x_init.dim() == 1:
            x_init = x_init.unsqueeze(1)
        if y_init.dim() == 1:
            y_init = y_init.unsqueeze(1)
        if t_init.dim() == 1:
            t_init = t_init.unsqueeze(1)
        
        # 模型预测
        inputs = torch.cat([x_init, y_init, t_init], dim=1)
        output = model(inputs)
        
        u_pred = output[:, 0:1]
        v_pred = output[:, 1:2]
        
        # 计算MSE损失
        u_loss = torch.mean((u_pred - u_init)**2)
        v_loss = torch.mean((v_pred - v_init)**2)
        total_loss = u_loss + v_loss
        
        return data_weight * total_loss
    
    # 如果没有批次数据，尝试加载单个时间点的数据
    elif 5.0 <= t_physical <= 10.0:
        # 加载当前时间点的真实数据
        x_norm, y_norm, u_norm, v_norm, t_norm = load_temporal_data_at_time(t_physical)
        
        if x_norm is not None and len(x_norm) > 0:
            # 转换为张量
            x_tensor = torch.tensor(x_norm, dtype=torch.float32, device=device).reshape(-1, 1)
            y_tensor = torch.tensor(y_norm, dtype=torch.float32, device=device).reshape(-1, 1)
            t_tensor = torch.full_like(x_tensor, t_norm)
            u_tensor = torch.tensor(u_norm, dtype=torch.float32, device=device).reshape(-1, 1)
            v_tensor = torch.tensor(v_norm, dtype=torch.float32, device=device).reshape(-1, 1)
            
            # 模型预测
            inputs = torch.cat([x_tensor, y_tensor, t_tensor], dim=1)
            output = model(inputs)
            
            u_pred = output[:, 0:1]
            v_pred = output[:, 1:2]
            
            # 计算MSE损失
            u_loss = torch.mean((u_pred - u_tensor)**2)
            v_loss = torch.mean((v_pred - v_tensor)**2)
            total_loss = u_loss + v_loss
            
            return data_weight * total_loss
    
    # 无数据时返回0
    return torch.tensor(0.0, device=device, requires_grad=True)



def load_temporal_data_at_time(t_physical):
    """加载指定物理时间的真实数据并进行归一化处理"""
    
    # 支持0.00-10.00s范围内的所有数据（1001个时间步，步长0.01s）
    if t_physical < 0.0 or t_physical > 10.0:
        return None, None, None, None
    
    # 处理浮点数精度问题：四舍五入到最接近的0.01秒
    t_rounded = min(round(t_physical * 100) / 100.0, 10.0)  # 四舍五入到0.01秒精度，限制最大值为10.0
    
    # 构建文件名（使用data1文件夹中的文件）
    filename = f'sigle_t={t_rounded:.2f}s.txt'
    filepath = os.path.join('data1', filename)
    
    if not os.path.exists(filepath):
        print(f"⚠️  加载时刻{t_physical:.2f}s数据失败: 文件不存在 {filepath}")
        return None, None, None, None
    
    try:
        # 读取数据（使用pandas读取有表头的文件）
        data = pd.read_csv(filepath, sep=r'\s+')
        
        # 增加采样点到500个以提高数据利用率
        if len(data) > 0:
            sample_size = min(len(data), 500)  # 从50增加到500个点
            # 随机选择索引
            indices = np.random.choice(len(data), size=sample_size, replace=False)
            data = data.iloc[indices]
        
        # 提取坐标和速度（原始数据：坐标mm，速度m/s）
        x_mm = data['x(mm)'].values  # mm单位
        y_mm = data['y(mm)'].values  # mm单位
        u_phys = data['u(m/s)'].values  # m/s
        v_phys = data['v(m/s)'].values  # m/s
        
        # 统一单位：将坐标从mm转换为m
        x_phys = x_mm / 1000.0  # mm -> m
        y_phys = y_mm / 1000.0  # mm -> m
        
        # 统一归一化处理：将物理单位转换为模型输入的[-1,1]范围
        # 时间归一化：物理时间t_physical -> [-1,1]
        t_norm = 2.0 * (t_physical - config.norm_params['t_min']) / (config.norm_params['t_max'] - config.norm_params['t_min']) - 1.0
        
        # 坐标归一化：m -> [-1,1]
        x_norm = 2.0 * (x_phys - config.norm_params['x_min']) / (config.norm_params['x_max'] - config.norm_params['x_min']) - 1.0
        y_norm = 2.0 * (y_phys - config.norm_params['y_min']) / (config.norm_params['y_max'] - config.norm_params['y_min']) - 1.0
        
        # 速度归一化：m/s -> [-1,1]
        u_norm = 2.0 * (u_phys - config.norm_params['u_min']) / (config.norm_params['u_max'] - config.norm_params['u_min']) - 1.0
        v_norm = 2.0 * (v_phys - config.norm_params['v_min']) / (config.norm_params['v_max'] - config.norm_params['v_min']) - 1.0
        
        return x_norm, y_norm, u_norm, v_norm, t_norm
        
    except Exception as e:
        print(f"[错误] 加载时刻{t_physical:.2f}s数据失败: {e}")
        return None, None, None, None, None

def create_directory_structure(save_dir):
    """创建目录结构说明文件"""
    structure_file = os.path.join(save_dir, '目录结构说明.txt')
    with open(structure_file, 'w', encoding='utf-8') as f:
        f.write("2D PINN 结果目录结构说明\n")
        f.write("="*50 + "\n\n")
        f.write("📁 主目录: 包含所有实验结果\n")
        f.write("  ├── experiment_info.txt - 实验基本信息\n")
        f.write("  ├── 目录结构说明.txt - 本文件\n")
        f.write("  ├── final_model.pth - 训练完成的模型权重\n")
        f.write("  ├── training_history.pkl - 训练历史数据\n")
        f.write("  ├── flow_field_t*.png - 各时间步的流场图\n")
        f.write("  └── 📁 time_evolution/ - 时间演化详细结果\n")
        f.write("      └── flow_field_t*.png - 详细时间演化图\n\n")
        f.write("文件命名规则:\n")
        f.write("- flow_field_t{时间}_epoch{轮数}.png: 训练过程中的流场图\n")
        f.write("- flow_field_t{时间}_epoch{序号}.png: 时间演化预测图\n\n")
        f.write("时间格式: t{时间值}s, 例如 t0.0s, t0.5s, t1.0s\n")
        f.write("轮数格式: epoch{数字}, 例如 epoch0, epoch1000, epoch2000\n")

    print(f"创建了目录结构说明文件: {structure_file}")

# 预测时间演化
def analyze_temporal_evolution(model, x_data, y_data, t_data, u_data, v_data):
    """分析时间演化误差（简化版，只支持u/v速度）"""
    if x_data is None or len(x_data) == 0:
        return None
    
    # 获取唯一时间点
    unique_times = np.unique(t_data)
    time_errors = {'time': [], 'u_mae': [], 'v_mae': [], 'speed_mae': [], 'u_r2': [], 'v_r2': [], 'speed_r2': []}
    
    for time_val in unique_times:
        # 选择当前时间点的数据
        mask = t_data == time_val
        x_t = x_data[mask]
        y_t = y_data[mask]
        t_t = t_data[mask]
        u_t = u_data[mask]
        v_t = v_data[mask]
        
        if len(x_t) == 0:
            continue
        
        # R²计算已禁用
        u_r2, v_r2, speed_r2 = 0.0, 0.0, 0.0
        
        # 归一化并预测
        x_norm, y_norm, u_norm, v_norm, t_norm = normalize_data(x_t, y_t, u_t, v_t, t_t)
        x_tensor = torch.tensor(x_norm, dtype=torch.float32).view(-1, 1).to(device)
        y_tensor = torch.tensor(y_norm, dtype=torch.float32).view(-1, 1).to(device)
        t_tensor = torch.tensor(t_norm, dtype=torch.float32).view(-1, 1).to(device)
        
        with torch.no_grad():
            # 将x, y, t拼接成一个输入张量，模型只接受一个参数
            inputs = torch.cat([x_tensor, y_tensor, t_tensor], dim=1)
            outputs = model(inputs)
            u_pred_norm = outputs[:, 0].cpu().numpy()
            v_pred_norm = outputs[:, 1].cpu().numpy()
        
        # 反归一化
        u_pred, v_pred = denormalize_velocity(u_pred_norm, v_pred_norm)
        
        # 计算MAE
        u_mae = np.mean(np.abs(u_t - u_pred))
        v_mae = np.mean(np.abs(v_t - v_pred))
        speed_mae = np.mean(np.abs(np.sqrt(u_t**2 + v_t**2) - np.sqrt(u_pred**2 + v_pred**2)))
        
        # 记录结果
        time_errors['time'].append(time_val)
        time_errors['u_mae'].append(u_mae)
        time_errors['v_mae'].append(v_mae)
        time_errors['speed_mae'].append(speed_mae)
        time_errors['u_r2'].append(u_r2)
        time_errors['v_r2'].append(v_r2)
        time_errors['speed_r2'].append(speed_r2)
    
    return time_errors

def analyze_temporal_performance(model, norm_params):
    """分析时间性能（简化版，用于训练总结图）"""
    # 加载时序数据
    x_data, y_data, t_data, u_data, v_data = load_temporal_data_range(0.0, 3.0)
    
    if x_data is None or len(x_data) == 0:
        print("⚠️ 无时序数据，跳过时间性能分析")
        return None
    
    # 使用analyze_temporal_evolution函数进行分析
    temporal_analysis = analyze_temporal_evolution(model, x_data, y_data, t_data, u_data, v_data)
    
    if temporal_analysis:
        print(f"✅ 时间性能分析完成，共{len(temporal_analysis['time'])}个时间点")
        
        # 打印前3个时间点的结果（R²计算已禁用）
        for i in range(min(3, len(temporal_analysis['time']))):
            time_val = temporal_analysis['time'][i]
            print(f"  t={time_val:.1f}s: speed_MAE={temporal_analysis['speed_mae'][i]:.6f}")
        
        if len(temporal_analysis['time']) > 3:
            print(f"  ... 还有{len(temporal_analysis['time'])-3}个时间点结果已保存到文件")
    
    return temporal_analysis

def plot_temporal_evolution_analysis(temporal_analysis, save_dir):
    """绘制时间演化分析4子图"""
    if not temporal_analysis or len(temporal_analysis['time']) == 0:
        print("⚠️ 无时间演化分析数据，跳过绘图")
        return
    
    plt.figure(figsize=(12, 8))
    
    # 子图1: R²随时间变化（已禁用）
    plt.subplot(2, 2, 1)
    plt.text(0.5, 0.5, 'R²计算已禁用', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
    plt.xlabel('时间 (s)')
    plt.ylabel('$R^2$')
    plt.title('$R^2$随时间变化')
    plt.grid(True, alpha=0.3)
    
    # 子图2: MAE随时间变化（u/v/speed MAE）
    plt.subplot(2, 2, 2)
    plt.plot(temporal_analysis['time'], temporal_analysis['u_mae'], 'b-o', label='u MAE', markersize=4)
    plt.plot(temporal_analysis['time'], temporal_analysis['v_mae'], 'r-s', label='v MAE', markersize=4)
    plt.plot(temporal_analysis['time'], temporal_analysis['speed_mae'], 'g-^', label='speed MAE', markersize=4)
    plt.xlabel('时间 (s)')
    plt.ylabel('MAE')
    plt.title('MAE随时间变化')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 子图3: 速度R²随时间变化（单独speed R²）
    plt.subplot(2, 2, 3)
    plt.plot(temporal_analysis['time'], temporal_analysis['speed_r2'], 'k-o', linewidth=2, markersize=4)
    plt.xlabel('时间 (s)')
    plt.ylabel('Speed $R^2$')
    plt.title('速度$R^2$随时间变化')
    plt.grid(True, alpha=0.3)
    
    # 子图4: 速度MAE随时间变化（单独speed MAE）
    plt.subplot(2, 2, 4)
    plt.plot(temporal_analysis['time'], temporal_analysis['speed_mae'], 'r-o', linewidth=2, markersize=4)
    plt.xlabel('时间 (s)')
    plt.ylabel('Speed MAE')
    plt.title('速度MAE随时间变化')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图像
    analysis_path = os.path.join(save_dir, 'temporal_evolution_analysis.png')
    plt.savefig(analysis_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 时间演化分析图已保存: {analysis_path}")


def get_temporal_stage_config(stage_idx):
    """获取指定阶段的时序配置"""
    if not config.TEMPORAL_EXTENSION['enable']:
        return None, None
    
    stage_times = config.TEMPORAL_EXTENSION['stage_times']
    stage_epochs = config.TEMPORAL_EXTENSION['stage_epochs']
    
    if stage_idx >= len(stage_times):
        return None, None
    
    stage_end_time = stage_times[stage_idx]
    stage_epochs_num = stage_epochs[stage_idx]
    
    return stage_end_time, stage_epochs_num


def plot_velocity_difference_heatmap_4subplot(model, save_path, epoch, grid_size=100):
    """绘制t=0.0/1.25/2.50/4.99s时刻的速度差异热图（4子图布局）"""
    import pandas as pd
    from utils import normalize_data, denormalize_velocity
    import matplotlib.pyplot as plt
    from scipy.interpolate import griddata
    
    # 时间点和对应的子图位置 - 改为用户要求的4个时间点 t=0.0/1.25/2.50/4.99s
    time_points = [0.0, 1.25, 2.50, 4.99]
    
    # 设备设置
    device = next(model.parameters()).device
    
    # 创建4子图布局（2x2）
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'速度差异热图对比 (epoch={epoch})', fontsize=16, fontweight='bold')
    
    # 设置固定色标范围 0-4.5
    vmin = 0.0
    vmax = 4.5
    
    # 选择4个时间点用于子图（t=5.0,7.5,10.0,15.0）
    selected_times = [0, 1, 2, 3]  # 对应time_points中的索引
    
    # 存储统计数据
    stats = []
    
    # 为每个选中的时间点创建子图
    for idx, time_idx in enumerate(selected_times):
        t_value = time_points[time_idx]  # 获取实际时间值（5.0, 7.5, 12.5, 15.0）
        
        # 为当前时间点加载对应的真实数据文件
        try:
            # 格式化时间字符串，确保小数点格式正确（支持0.00, 1.25, 2.50, 4.99）
            time_str = f"{t_value:.2f}"
            data_file = f'data2/sigle_t={time_str}s.txt'
            data = pd.read_csv(data_file, sep=r'\s+')
            # 将mm单位转换为m
            x_orig = data['x(mm)'].values / 1000.0
            y_orig = data['y(mm)'].values / 1000.0
            u_orig = data['u(m/s)'].values
            v_orig = data['v(m/s)'].values
        except Exception as e:
            print(f"读取时刻{t_value:.2f}s的真实数据失败: {e}")
            continue
        
        # 归一化输入 - 使用实际时间值
        x_norm, y_norm, t_norm = normalize_data(x_orig, y_orig, t=np.full_like(x_orig, t_value))
        
        # 转换为tensor
        x_tensor = torch.tensor(x_norm, dtype=torch.float32, device=device).unsqueeze(1)
        y_tensor = torch.tensor(y_norm, dtype=torch.float32, device=device).unsqueeze(1)
        t_tensor = torch.tensor(t_norm, dtype=torch.float32, device=device).unsqueeze(1)
        
        # 模型预测
        model.eval()
        with torch.no_grad():
            # 将x, y, t拼接成一个输入张量，模型只接受一个参数
            inputs = torch.cat([x_tensor, y_tensor, t_tensor], dim=1)
            output = model(inputs)
            u_pred_norm = output[:, 0].cpu().numpy()
            v_pred_norm = output[:, 1].cpu().numpy()
            
            # 反归一化速度
            u_pred, v_pred = denormalize_velocity(u_pred_norm, v_pred_norm)
        
        # 计算速度差（使用绝对值）
        u_diff = np.abs(u_pred - u_orig)
        v_diff = np.abs(v_pred - v_orig)
        speed_diff = np.sqrt(u_diff**2 + v_diff**2)
        
        # 创建规则网格用于热图（基于当前时间点的数据范围）
        x_min, x_max = x_orig.min(), x_orig.max()
        y_min, y_max = y_orig.min(), y_orig.max()
        xi = np.linspace(x_min, x_max, grid_size)
        yi = np.linspace(y_min, y_max, grid_size)
        Xi, Yi = np.meshgrid(xi, yi)
        
        # 插值到规则网格
        speed_diff_grid = griddata((x_orig, y_orig), speed_diff, (Xi, Yi), method='linear')
        
        # 确定子图位置
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]
        
        # 绘制热图
        contour = ax.contourf(Xi, Yi, speed_diff_grid, 50, cmap='jet', 
                             vmin=vmin, vmax=vmax, extend='both')
        
        # 添加等值线
        levels = np.linspace(vmin, vmax, 11)
        ax.contour(Xi, Yi, speed_diff_grid, levels=levels[::2], colors='black', alpha=0.3, linewidths=0.5)
        
        # 设置坐标轴
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel('x坐标 (m)')
        ax.set_ylabel('y坐标 (m)')
        ax.set_title(f't={t_value:.2f}s')
        ax.set_aspect('equal', adjustable='box')
        
        # 添加统计信息
        mean_diff = np.nanmean(speed_diff_grid)
        std_diff = np.nanstd(speed_diff_grid)
        max_diff = np.nanmax(speed_diff_grid)
        ax.text(0.02, 0.98, f'平均差值: {mean_diff:.4f}\n标准差: {std_diff:.4f}', 
               transform=ax.transAxes, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
               fontsize=9)
        
        # 为每个子图创建独立的colorbar
        cbar = fig.colorbar(contour, ax=ax, shrink=0.6, aspect=20)
        if idx == 1 or idx == 3:  # 第2和第4个子图
            cbar.set_label('速度差值 (m/s)')
        
        # 存储统计数据
        stats.append({
            'time': t_value,
            'mean_diff': mean_diff,
            'std_diff': std_diff,
            'max_diff': max_diff
        })
    
    plt.tight_layout()
    
    # 保存图片
    filename = f'velocity_difference_4subplot_epoch{epoch}.png'
    filepath = os.path.join(save_path, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"保存4子图速度差热图 (t=0.00/1.25/2.50/4.99s): {filepath}")
    
    return stats

def predict_time_evolution(model, n_steps=10, time_range=(0.0, 2.0)):
    """预测指定时间范围内的流场演化"""
    
    print("预测时间演化...")
    
    # 创建时间演化子目录
    evolution_dir = os.path.join(args.outdir, 'time_evolution')
    os.makedirs(evolution_dir, exist_ok=True)
    
    # 时间步
    time_steps = np.linspace(time_range[0], time_range[1], n_steps)
    
    for i, t_val in enumerate(time_steps):
        print(f"预测时间步 {i+1}/{n_steps}: t={t_val:.2f}s")
        plot_2d_flow_field(model, t_val, evolution_dir, i)
    
    print("时间演化预测完成!")
def train_dual_model_system():
    print("="*80)
    print("启动 PINN 驱动型协同训练系统 (Phase3专属优选版)")
    print("配置: 权重 Data=0.715 | Mamba=0.285")
    print("策略: Phase2冲刺(不保存最佳) -> Phase3精修(保存最佳R1)")
    print("规则: Phase2跳转(R2>=0.99) | Phase3结束(TotalLoss<0.001)")
    print("="*80)
    
    # ------------------------------------------------------------------
    # 0. 全局超参数
    # ------------------------------------------------------------------
    W_DATA_FUSED = 0.715
    W_MAMBA_FUSED = 0.285
    N_SAMPLE_REAL = 2860
    N_SAMPLE_MAMBA = 1140
    
    # [目标] Phase 2 跳转阈值
    MAMBA_R2_THRESHOLD = 0.99
    
    # [物理策略]: 0.01s 极精细
    DT_PHYS = 0.01          
    DATA_DT = 0.01          
    STRIDE = int(DT_PHYS / DATA_DT) # Stride = 1
    PRED_STEPS = 5          
    
    # Mamba 训练频率
    MAMBA_TRAIN_FREQ = 2
    
    # 最佳模型记录器 (初始化)
    best_extrap_r1 = -float('inf')
    best_model_path = os.path.join(args.outdir, 'best_model.pth')
    
    # ------------------------------------------------------------------
    # 1. 优化器与混合精度
    # ------------------------------------------------------------------
    optimizer_pinn = torch.optim.Adam(model.parameters(), lr=args.adamLR)
    optimizer_mamba = torch.optim.Adam(mamba_teacher.parameters(), lr=0.0001) 
    scheduler_pinn = torch.optim.lr_scheduler.StepLR(optimizer_pinn, step_size=2000, gamma=0.8)
    scheduler_mamba = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_mamba, T_max=20000, eta_min=1e-7)
    scaler = torch.cuda.amp.GradScaler()
    
    # ------------------------------------------------------------------
    # 2. 数据加载 (全量点模式)
    # ------------------------------------------------------------------
    try:
        data_path = 'data2/sigle_t=0.00s.txt'
        if not os.path.exists(data_path): data_path = 'sigle_t=0.00s.txt' 
        df_initial = pd.read_csv(data_path, sep=r'\s+')
        
        # [全量]: 使用所有点作为 Mamba 锚点
        fixed_probes_phys = df_initial[['x(mm)', 'y(mm)']].values / 1000.0
        fixed_indices = np.arange(len(df_initial))
        
        print(f"✅ Mamba 空间点设置: 全流域覆盖，共 {len(fixed_indices)} 个点")
        
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        fixed_probes_phys = np.zeros((100, 2))
        fixed_indices = np.arange(100)
        
    x_n = 2.0*(fixed_probes_phys[:,0] - config.norm_params['x_min'])/(config.norm_params['x_max'] - config.norm_params['x_min']) - 1.0
    y_n = 2.0*(fixed_probes_phys[:,1] - config.norm_params['y_min'])/(config.norm_params['y_max'] - config.norm_params['y_min']) - 1.0
    fixed_probes_tensor = torch.tensor(np.column_stack([x_n, y_n]), dtype=torch.float32, device=device)
    num_fixed_points = len(fixed_probes_tensor)
    
    # 加载 0-5s 真实数据
    x_05, y_05, t_05, u_05, v_05 = load_temporal_data_range(0.0, 5.0, 4000)
    
    def to_tensor_list(x, y, t, u, v):
        xn, yn, tn = normalize_data(x, y, t=t)[:3]
        un = 2.0*(u - config.norm_params['u_min'])/(config.norm_params['u_max'] - config.norm_params['u_min']) - 1.0
        vn = 2.0*(v - config.norm_params['v_min'])/(config.norm_params['v_max'] - config.norm_params['v_min']) - 1.0
        return [torch.tensor(k, dtype=torch.float32).view(-1,1).to(device) for k in [xn, yn, tn, un, vn]]

    real_dataset_tensors = to_tensor_list(x_05, y_05, t_05, u_05, v_05)
    
    val_data_05 = {'x_norm': real_dataset_tensors[0], 'y_norm': real_dataset_tensors[1], 
                   't_norm': real_dataset_tensors[2], 'u_phys': u_05, 'v_phys': v_05}
    
    # 外推验证集
    x_57, y_57, t_57, u_57, v_57 = load_temporal_data_range(5.01, 7.0, 2000)
    val_data_57 = None
    if len(x_57) > 0:
        xn_57, yn_57, tn_57 = normalize_data(x_57, y_57, t=t_57)[:3]
        val_data_57 = {
            'x_norm': torch.tensor(xn_57, dtype=torch.float32).view(-1,1).to(device),
            'y_norm': torch.tensor(yn_57, dtype=torch.float32).view(-1,1).to(device),
            't_norm': torch.tensor(tn_57, dtype=torch.float32).view(-1,1).to(device),
            'u_phys': u_57, 'v_phys': v_57
        }

    # ------------------------------------------------------------------
    # 3. Mamba 数据准备 (全量读取)
    # ------------------------------------------------------------------
    def load_full_index_tensor(t_start, t_end):
        import pandas as pd
        data_list = []
        steps = int((t_end - t_start) / 0.01) + 1
        target_ts = [t_start + i*0.01 for i in range(steps)]
        
        print(f"正在加载全量 Mamba 训练数据 ({t_start}-{t_end}s)...")
        
        for t_val in target_ts:
            t_str = f"{t_val:.2f}"
            f_path = f'data2/sigle_t={t_str}s.txt'
            if not os.path.exists(f_path): f_path = f'sigle_t={t_str}s.txt'
            
            try:
                df_temp = pd.read_csv(f_path, sep=r'\s+')
                u_vals = df_temp['u(m/s)'].values[fixed_indices]
                v_vals = df_temp['v(m/s)'].values[fixed_indices]
                
                u_n = 2.0*(u_vals - config.norm_params['u_min'])/(config.norm_params['u_max'] - config.norm_params['u_min']) - 1.0
                v_n = 2.0*(v_vals - config.norm_params['v_min'])/(config.norm_params['v_max'] - config.norm_params['v_min']) - 1.0
                data_list.append(np.column_stack((u_n, v_n)))
            except Exception:
                if len(data_list) > 0: data_list.append(data_list[-1])
                else: data_list.append(np.zeros((num_fixed_points, 2)))
                    
        return torch.tensor(np.array(data_list), dtype=torch.float32, device=device).permute(1, 0, 2)
    
    mamba_train_cache = load_full_index_tensor(0.0, 5.0) 
    mamba_extrap_gt = load_full_index_tensor(5.01, 7.0)

    # ------------------------------------------------------------------
    # 考核函数
    # ------------------------------------------------------------------
    def check_mamba_accuracy():
        if mamba_train_cache is None or mamba_extrap_gt is None: return 0.0
        mamba_teacher.eval()
        with torch.no_grad():
            indices = torch.arange(-4, 1, 1, device=device)
            warmup_uv = mamba_train_cache[:, indices, :] 
            
            pred_uv = mamba_teacher.predict_sequence(None, fixed_probes_tensor, 200, 0.01, warmup_uv)
            
            def denorm(val):
                u = val[:,:,0] * (config.norm_params['u_max'] - config.norm_params['u_min'])/2 + (config.norm_params['u_max'] + config.norm_params['u_min'])/2
                v = val[:,:,1] * (config.norm_params['v_max'] - config.norm_params['v_min'])/2 + (config.norm_params['v_max'] + config.norm_params['v_min'])/2
                return torch.sqrt(u**2 + v**2).cpu().numpy().flatten()

            true_speed = denorm(mamba_extrap_gt)
            pred_speed = denorm(pred_uv)
            return r2_score(true_speed, pred_speed)

    # ------------------------------------------------------------------
    # 4. 状态变量
    # ------------------------------------------------------------------
    current_phase = 1
    phase2_start_epoch = 99999
    
    samples_cache = {
        'x_p': None, 'y_p': None, 't_p': None, 
        'real_x': None, 'real_y': None, 'real_t': None, 'real_u': None, 'real_v': None,
        'mamba_x': None, 'mamba_y': None, 'mamba_t': None, 'mamba_u': None, 'mamba_v': None
    }
    
    start_time = time.time()
    history = {'total_loss': [], 'epoch_r2_speed': [], 'epoch_r2_extrap_speed': []}

    # =================================================================
    # Phase 1 & 2: Adam Training
    # =================================================================
    adam_epochs = 20000 
    
    for epoch in range(adam_epochs):
        model.train(); mamba_teacher.train()
        
        # ---------------------------------------------------------
        # 采样逻辑
        # ---------------------------------------------------------
        if epoch % 100 == 0:
            x_p, y_p, t_p, _ = sample_spatiotemporal_points_uniform(20000, device, (0.0, 7.0))
            samples_cache['x_p'], samples_cache['y_p'], samples_cache['t_p'] = x_p, y_p, t_p
            
            if current_phase == 1:
                idx = torch.randint(0, len(real_dataset_tensors[0]), (4000,), device=device)
                samples_cache['real_x'] = real_dataset_tensors[0][idx]
                samples_cache['real_y'] = real_dataset_tensors[1][idx]
                samples_cache['real_t'] = real_dataset_tensors[2][idx]
                samples_cache['real_u'] = real_dataset_tensors[3][idx]
                samples_cache['real_v'] = real_dataset_tensors[4][idx]
                samples_cache['mamba_x'] = None 
                
                res = calculate_r2_and_mae(model, val_data_05, use_cached_tensors=True)
                pinn_r2 = res.get('speed_r2', 0.0)
                if pinn_r2 >= 0.985: 
                    print(f"\n🚀 [阶段切换] PINN R2={pinn_r2:.4f} | 进入 Phase 2")
                    current_phase = 2
                    phase2_start_epoch = epoch

            if current_phase == 2:
                idx_r = torch.randint(0, len(real_dataset_tensors[0]), (N_SAMPLE_REAL,), device=device)
                samples_cache['real_x'] = real_dataset_tensors[0][idx_r]
                samples_cache['real_y'] = real_dataset_tensors[1][idx_r]
                samples_cache['real_t'] = real_dataset_tensors[2][idx_r]
                samples_cache['real_u'] = real_dataset_tensors[3][idx_r]
                samples_cache['real_v'] = real_dataset_tensors[4][idx_r]
                
                mamba_teacher.eval()
                with torch.no_grad():
                    indices = torch.arange(-4, 1, 1, device=device)
                    warmup_uv = mamba_train_cache[:, indices, :]
                    mamba_pred = mamba_teacher.predict_sequence(None, fixed_probes_tensor, 200, 0.01, warmup_uv)
                    
                    t_vals = torch.linspace(5.01, 7.0, 200, device=device)
                    t_norm_57 = 2.0 * (t_vals - config.norm_params['t_min']) / (config.norm_params['t_max'] - config.norm_params['t_min']) - 1.0
                    
                    m_x_flat = fixed_probes_tensor[:, 0].repeat_interleave(200).view(-1,1)
                    m_y_flat = fixed_probes_tensor[:, 1].repeat_interleave(200).view(-1,1)
                    m_t_flat = t_norm_57.repeat(num_fixed_points).view(-1,1)
                    m_u_flat = mamba_pred[:,:,0].flatten().view(-1,1)
                    m_v_flat = mamba_pred[:,:,1].flatten().view(-1,1)
                    
                    idx_m = torch.randint(0, len(m_x_flat), (N_SAMPLE_MAMBA,), device=device)
                    samples_cache['mamba_x'] = m_x_flat[idx_m]
                    samples_cache['mamba_y'] = m_y_flat[idx_m]
                    samples_cache['mamba_t'] = m_t_flat[idx_m]
                    samples_cache['mamba_u'] = m_u_flat[idx_m]
                    samples_cache['mamba_v'] = m_v_flat[idx_m]
                    
                    # 考核 (Phase 2 -> 3 跳转条件)
                    mamba_r2 = check_mamba_accuracy()
                    if mamba_r2 >= MAMBA_R2_THRESHOLD:
                        print(f"\n🎉 [条件满足] Mamba R2={mamba_r2:.4f} >= {MAMBA_R2_THRESHOLD}")
                        print("🚀 锁定当前混合数据集，进入 Phase 3 (L-BFGS)")
                        current_phase = 3
                        break 

        # ---------------------------------------------------------
        # PINN Update
        # ---------------------------------------------------------
        optimizer_pinn.zero_grad()
        with torch.cuda.amp.autocast():
            l_p, _, _, _, _ = physics_loss_2d(model, samples_cache['x_p'], samples_cache['y_p'], samples_cache['t_p'], config.REYNOLDS_NUMBER, config.norm_params)
            
            out_r = model(torch.cat([samples_cache['real_x'], samples_cache['real_y'], samples_cache['real_t']], dim=1))
            l_real = torch.mean((out_r[:,0:1]-samples_cache['real_u'])**2 + (out_r[:,1:2]-samples_cache['real_v'])**2)
            
            if current_phase == 1:
                loss = l_p + l_real
                l_mamba_loss = torch.tensor(0.0)
            else:
                out_m = model(torch.cat([samples_cache['mamba_x'], samples_cache['mamba_y'], samples_cache['mamba_t']], dim=1))
                l_mamba_loss = torch.mean((out_m[:,0:1]-samples_cache['mamba_u'])**2 + (out_m[:,1:2]-samples_cache['mamba_v'])**2)
                loss = l_p + (W_DATA_FUSED * l_real + W_MAMBA_FUSED * l_mamba_loss)
            
        scaler.scale(loss).backward()
        scaler.step(optimizer_pinn)
        scaler.update()
        scheduler_pinn.step()
        
        # ---------------------------------------------------------
        # Mamba Update
        # ---------------------------------------------------------
        if 'l_mamba_train' not in locals():
            l_mamba_train = torch.tensor(1.0, device=device)

        if epoch % MAMBA_TRAIN_FREQ == 0 and mamba_train_cache is not None:
            mamba_loops = 10
            current_mamba_loss_sum = 0.0
            
            for _ in range(mamba_loops): 
                optimizer_mamba.zero_grad()
                total_span = 4 + 200
                max_start = mamba_train_cache.shape[1] - total_span - 5 
                if max_start > 0:
                    start = np.random.randint(0, max_start)
                    input_indices = torch.arange(0, 5, 1, device=device) + start
                    warmup_uv = mamba_train_cache[:, input_indices, :]
                    target_start = start + 5
                    target_indices = torch.arange(target_start, target_start + 200, 1, device=device)
                    target_uv = mamba_train_cache[:, target_indices, :]
                    
                    xy_ex = fixed_probes_tensor.unsqueeze(1).expand(-1, 5, -1)
                    inp = torch.cat([xy_ex, warmup_uv], dim=2)
                    pred = mamba_teacher(inp, use_direct_head=True)
                    loss_m = torch.mean((pred - target_uv)**2)
                    loss_m.backward()
                    optimizer_mamba.step()
                    current_mamba_loss_sum += loss_m.detach()
            
            l_mamba_train = current_mamba_loss_sum / mamba_loops
            scheduler_mamba.step()

        # ---------------------------------------------------------
        # Logging (Phase 2 不保存最佳模型)
        # ---------------------------------------------------------
        if epoch % 100 == 0:
            with torch.no_grad():
                res = calculate_r2_and_mae(model, val_data_05, use_cached_tensors=True)
                curr_r2 = res.get('speed_r2', 0.0)
                res_ex = calculate_r2_and_mae(model, val_data_57, use_cached_tensors=True)
                curr_r1 = res_ex.get('speed_r2', 0.0)
                curr_mamba_r2 = check_mamba_accuracy()

            print(f"Iter {epoch} [Phase {current_phase}] | Loss: {loss.item():.5f} | Phys: {l_p.item():.5f}")
            print(f"   Data(Real): {l_real.item():.5f} | Mamba_Train: {l_mamba_train:.5f}")
            print(f"   Metrics: PINN_R2(0-5s)={curr_r2:.4f} | PINN_R1(5-7s)={curr_r1:.4f} | Mamba_R2(5-7s)={curr_mamba_r2:.4f}")
            
            history['total_loss'].append(loss.item())
            history['epoch_r2_speed'].append(curr_r2)
            history['epoch_r2_extrap_speed'].append(curr_r1)

    # ==============================================================================
    # Phase 3: L-BFGS (Inherit Samples) + Early Stopping + Best Save
    # ==============================================================================
    class LBFGSEarlyStop(Exception):
        pass

    if current_phase >= 2:
        print("\n" + "#"*80)
        print(">>> 正式进入 Phase 3: L-BFGS 精细优化阶段 (Force 30000 Steps) <<<")
        print(">>> [操作] 直接锁定 Phase 2 最后一轮的数据集")
        print(">>> [规则] Total Loss < 0.001 提前结束 | 始终保存 R1 最高模型")
        print(f"锁定状态: Phys=20000, Real={N_SAMPLE_REAL}, Mamba={N_SAMPLE_MAMBA}")
        print("#"*80)
        
        optimizer_lbfgs = torch.optim.LBFGS(
            model.parameters(), 
            lr=1.0, 
            max_iter=30000,      
            max_eval=30000, 
            history_size=100,    
            tolerance_grad=1e-9, 
            tolerance_change=1e-11,
            line_search_fn="strong_wolfe"
        )
        
        lbfgs_iter_count = 0
        lbfgs_start_time = time.time()
        
        def closure():
            nonlocal lbfgs_iter_count, lbfgs_start_time, best_extrap_r1 
            optimizer_lbfgs.zero_grad()
            
            l_p, _, _, _, _ = physics_loss_2d(model, samples_cache['x_p'], samples_cache['y_p'], samples_cache['t_p'], config.REYNOLDS_NUMBER, config.norm_params)
            out_r = model(torch.cat([samples_cache['real_x'], samples_cache['real_y'], samples_cache['real_t']], dim=1))
            l_real = torch.mean((out_r[:,0:1]-samples_cache['real_u'])**2 + (out_r[:,1:2]-samples_cache['real_v'])**2)
            out_m = model(torch.cat([samples_cache['mamba_x'], samples_cache['mamba_y'], samples_cache['mamba_t']], dim=1))
            l_mamba_loss = torch.mean((out_m[:,0:1]-samples_cache['mamba_u'])**2 + (out_m[:,1:2]-samples_cache['mamba_v'])**2)
            
            loss = l_p + (W_DATA_FUSED * l_real + W_MAMBA_FUSED * l_mamba_loss)
            loss.backward()
            
            # [核心新增] Phase 3 早停判断
            if loss.item() < 0.001:
                print(f"\n🚀 [L-BFGS 提前结束] Total Loss={loss.item():.6f} < 0.001")
                raise LBFGSEarlyStop
            
            if lbfgs_iter_count % 100 == 0:
                elapsed = time.time() - lbfgs_start_time
                it_per_sec = 100 / elapsed if (lbfgs_iter_count > 0 and elapsed > 0) else 0.0
                lbfgs_start_time = time.time()
                
                with torch.no_grad():
                    res = calculate_r2_and_mae(model, val_data_05, use_cached_tensors=True)
                    curr_r2 = res.get('speed_r2', 0.0)
                    res_ex = calculate_r2_and_mae(model, val_data_57, use_cached_tensors=True)
                    curr_r1 = res_ex.get('speed_r2', 0.0)
                    curr_mamba_r2 = check_mamba_accuracy()
                    
                    # [仅在 Phase 3 开启最佳模型保存]
                    if curr_r1 > best_extrap_r1:
                        best_extrap_r1 = curr_r1
                        torch.save(model.state_dict(), best_model_path)
                        print(f"🔥 [L-BFGS] New Best Extrap R1: {best_extrap_r1:.4f} | Model Saved.")
                
                print(f"Iter {adam_epochs + lbfgs_iter_count} [L-BFGS] | Loss: {loss.item():.6f} | Phys: {l_p.item():.6f}")
                print(f"   Data: Real({W_DATA_FUSED})={l_real.item():.6f} | Mamba({W_MAMBA_FUSED})={l_mamba_loss.item():.6f} | Mamba_Train=Locked")
                print(f"   Metrics: PINN_R2(0-5s)={curr_r2:.4f} | PINN_R1(5-7s)={curr_r1:.4f} | Mamba_R2(5-7s)={curr_mamba_r2:.4f}")
                
            lbfgs_iter_count += 1
            return loss
            
        try:
            optimizer_lbfgs.step(closure)
        except LBFGSEarlyStop:
            print(">>> L-BFGS 触发早停机制，训练结束。")
            
        print(">>> Phase 3: L-BFGS 完成 <<<")

    # 回滚操作
    if os.path.exists(best_model_path):
        print(f"\n🏆 正在回滚至历史最佳模型 (R1={best_extrap_r1:.4f})...")
        model.load_state_dict(torch.load(best_model_path))
        print("✅ 模型参数已替换为最佳状态")
    else:
        print("\n⚠️ 未找到最佳模型文件，保留最后一次迭代状态")

    torch.save(model.state_dict(), os.path.join(args.outdir, 'final_model.pth'))
    import pickle
    with open(os.path.join(args.outdir, 'training_history.pkl'), 'wb') as f: pickle.dump(history, f)
    
    return model, history
def generate_parity_plot(model, save_dir, sample_size=2000, external_metrics=None):
    """
    生成校准图
    Args:
        external_metrics: dict, 包含 'u', 'v', 'speed', 'r1' 等指标
    """
    try:
        print(f"\n[生成] 校准图（Parity Plot）...")
        
        x_ten, y_ten, t_ten, u_ten, v_ten = sample_spatiotemporal_data_points_uniform(sample_size, (0.0, 5.0))
        if x_ten is None: return

        model.eval()
        with torch.no_grad():
            inputs = torch.cat([x_ten, y_ten, t_ten], dim=1)
            outputs = model(inputs)
            u_pred_norm = outputs[:, 0:1].cpu().numpy().flatten()
            v_pred_norm = outputs[:, 1:2].cpu().numpy().flatten()
            
        u_true_norm = u_ten.cpu().numpy().flatten()
        v_true_norm = v_ten.cpu().numpy().flatten()
        
        u_min, u_max = config.norm_params['u_min'], config.norm_params['u_max']
        v_min, v_max = config.norm_params['v_min'], config.norm_params['v_max']
        
        def denorm_np(val, v_min, v_max):
            return 0.5 * (val + 1.0) * (v_max - v_min) + v_min

        u_pred = denorm_np(u_pred_norm, u_min, u_max)
        v_pred = denorm_np(v_pred_norm, v_min, v_max)
        u_true = denorm_np(u_true_norm, u_min, u_max)
        v_true = denorm_np(v_true_norm, v_min, v_max)
        
        speed_true = np.sqrt(u_true**2 + v_true**2)
        speed_pred = np.sqrt(u_pred**2 + v_pred**2)
        
        # 指标获取
        r2_u, r2_v, r2_spd, r1_spd = 0, 0, 0, 0
        if external_metrics:
            r2_u = external_metrics.get('u', 0.0)
            r2_v = external_metrics.get('v', 0.0)
            r2_spd = external_metrics.get('speed', 0.0)
            r1_spd = external_metrics.get('r1', 0.0) # 获取 R1
            print(f"✅ 使用训练日志指标: R2={r2_spd:.4f}, R1={r1_spd:.4f}")
        else:
            from sklearn.metrics import r2_score
            r2_u = r2_score(u_true, u_pred)
            r2_v = r2_score(v_true, v_pred)
            r2_spd = r2_score(speed_true, speed_pred)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # 标题中加入 R1 信息
        main_title = f"Calibration (Train R2={r2_spd:.4f} | Extrap R1={r1_spd:.4f})"
        fig.suptitle(main_title, fontsize=14)
        
        axes[0].scatter(u_true, u_pred, alpha=0.5, s=10, c='blue')
        axes[0].plot([u_true.min(), u_true.max()], [u_true.min(), u_true.max()], 'r--')
        axes[0].set_title(f'u (R2={r2_u:.4f})'); axes[0].set_xlabel('True u'); axes[0].set_ylabel('Pred u')
        
        axes[1].scatter(v_true, v_pred, alpha=0.5, s=10, c='green')
        axes[1].plot([v_true.min(), v_true.max()], [v_true.min(), v_true.max()], 'r--')
        axes[2].set_title(f'v (R2={r2_v:.4f})'); axes[1].set_xlabel('True v')
        
        axes[2].scatter(speed_true, speed_pred, alpha=0.5, s=10, c='orange')
        axes[2].plot([speed_true.min(), speed_true.max()], [speed_true.min(), speed_true.max()], 'r--')
        axes[2].set_title(f'Speed (R2={r2_spd:.4f})'); axes[2].set_xlabel('True Speed')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'parity_plot.png'), dpi=150)
        plt.close()
        print(f"✅ 校准图已保存.")
        
    except Exception as e:
        print(f"❌ 校准图生成失败: {e}")
# 主函数
if __name__ == "__main__":
    
    print("="*60)
    print("2D PINN 流场预测程序 (Mamba 自监督外推版 0-7s)")
    print("="*60)
    
    # 1. 训练模型
    # 注意：train_2d_pinn() 是入口函数，它会调用 train_2d_pinn_temporal_extension()
    model, history = train_2d_pinn()
    
    # 2. 预测时间演化 (生成流场快照)
    # 范围改为 0.0 - 7.0s，每 0.2s 一帧，共 36 帧
    print("\n[生成] 时间演化快照 (0-7s)...")
    predict_time_evolution(model, n_steps=36, time_range=(0.0, 7.0))
    
    print("="*60)
    print("训练与基础预测完成!")
    print("="*60)
    
    # 3. 生成综合评价图
    print("\n[生成] 模型综合评价图...")
    try:
        # 确保 visualize 模块中有此函数，如果没有，请使用下面的备份绘图逻辑
        from visualize import generate_comprehensive_evaluation_plots
        evaluation_plots = generate_comprehensive_evaluation_plots(
            model=model,
            history=history, 
            save_path=os.path.join(args.outdir, 'evaluation_plots')
        )
        print(f"✅ 成功生成 {len(evaluation_plots)} 张综合评价图")
    except Exception as e:
        print(f"⚠️ 生成综合评价图时出错 (可能缺少visualize模块对应函数): {e}")
    
    # 4. 生成时间性能分析图 (R² 和 MAE 随时间变化)
    print("\n[生成] 时间性能分析图 (0-7s)...")
    try:
        # 加载 0-7s 的全量数据进行最终评估
        # 注意：这里需要确保 load_temporal_data_range 支持跨度加载
        x_all, y_all, t_all, u_all, v_all = load_temporal_data_range(0.0, 5.0) # 只有0-5s有真实数据
        
        if x_all is not None and len(x_all) > 0:
            # 分析 0-5s (真实数据区) 的性能
            temporal_analysis = analyze_temporal_evolution(model, x_all, y_all, t_all, u_all, v_all)
            
            # 绘制分析图
            plot_temporal_evolution_analysis(temporal_analysis, args.outdir)
            print("✅ 时间性能分析图已保存")
        else:
            print("⚠️ 未加载到验证数据，跳过时间性能分析")
            
    except Exception as e:
        print(f"❌ 生成时间性能分析图出错: {e}")
    
    # 5. 生成校准图 (Parity Plot)
    print("\n[生成] 最终校准图...")
    try:
        # 提取 R2 和 R1 指标 - 安全获取，避免空列表
        final_r2 = history.get('epoch_r2_speed', [0.0])[-1] if history.get('epoch_r2_speed') else 0.0
        final_r1 = history.get('epoch_r2_extrap_speed', [0.0])[-1] if history.get('epoch_r2_extrap_speed') else 0.0
        final_r2_u = history.get('epoch_r2_u', [0.0])[-1] if history.get('epoch_r2_u') else 0.0
        final_r2_v = history.get('epoch_r2_v', [0.0])[-1] if history.get('epoch_r2_v') else 0.0
        
        final_metrics = {
            'u': final_r2_u,
            'v': final_r2_v,
            'speed': final_r2,
            'r1': final_r1
        }
        generate_parity_plot(model, args.outdir, sample_size=2000, external_metrics=final_metrics)
        
        # 保存指标到 txt
        with open(os.path.join(args.outdir, 'final_metrics.txt'), 'w') as f:
            f.write(f"R2 (0-5s): {final_r2:.6f}\n")
            f.write(f"R1 (5-7s): {final_r1:.6f}\n")
        print(f"✅ 最终指标已保存: R2={final_r2:.4f}, R1={final_r1:.4f}")
        
    except Exception as e:
        print(f"❌ 生成校准图出错: {e}")

    # 打印最终文件清单
    print(f"\n📁 所有结果已保存到: {args.outdir}")
    print(f"  - final_model.pth (模型权重)")
    print(f"  - training_history.pkl (损失记录)")
    print(f"  - evaluation_results.json (评估指标)")
    print(f"  - time_evolution/ (0-10s 流场动画帧)")
    
    # =========================================================
    # 5. 生成训练总结与分段指标分析图
    # =========================================================
    print("\n[生成] 训练总结图与分段指标分析...")
    
    # ---------------------------------------------------------
    # 图表 A: 训练过程 R² vs R1² 对比图 (核心需求)
    # ---------------------------------------------------------
    if history and 'epoch_r2_u' in history and len(history['epoch_r2_u']) > 0:
        print("📈 生成分段 R² 指标分析图 (R² vs R1²)...")
        
        # 准备 x 轴 (Epochs)
        num_points = len(history['epoch_r2_u'])
        actual_epochs = len(history.get('total_loss', []))
        if actual_epochs > 0 and num_points > 1:
            epochs = np.linspace(0, actual_epochs - 1, num_points)
        else:
            epochs = np.arange(num_points)
            
        plt.figure(figsize=(16, 10))
        
        # 子图 1: 速度 R² (训练域 0-5s) vs R1² (外推域 5-7s)
        plt.subplot(2, 2, 1)
        # 绘制 0-5s (训练域)
        plt.semilogx(epochs + 1, history['epoch_r2_speed'], 'b-o', label='Train $R^2$ (0-5s)', markersize=4, alpha=0.8)
        # 绘制 5-7s (外推域) - 如果有数据
        if 'epoch_r2_extrap_speed' in history and len(history['epoch_r2_extrap_speed']) == len(epochs):
            # 过滤掉初始阶段可能的0值，避免log坐标报错或图表难看
            r1_data = np.array(history['epoch_r2_extrap_speed'])
            plt.semilogx(epochs + 1, r1_data, 'r-^', label='Extrap $R_1^2$ (5-7s)', markersize=4, alpha=0.8)
            
        plt.xlabel('Epoch (Log Scale)')
        plt.ylabel('Score')
        plt.title('Performance: Train $R^2$ vs Extrapolation $R_1^2$')
        plt.legend()
        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.ylim([-0.5, 1.05]) # 允许一些负值以便观察早期训练，但主要关注正值
        
        # 子图 2: 各分量 R² 细节 (0-5s)
        plt.subplot(2, 2, 2)
        plt.semilogx(epochs + 1, history['epoch_r2_u'], 'b--', label='u $R^2$', alpha=0.6)
        plt.semilogx(epochs + 1, history['epoch_r2_v'], 'g--', label='v $R^2$', alpha=0.6)
        plt.semilogx(epochs + 1, history['epoch_r2_speed'], 'k-', label='Speed $R^2$', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('$R^2$')
        plt.title('Training Domain Accuracy (0-5s)')
        plt.legend()
        plt.grid(True)
        plt.ylim([0, 1.05])
        
        # 子图 3: 损失函数收敛曲线
        plt.subplot(2, 2, 3)
        plt.loglog(range(1, len(history['total_loss']) + 1), history['total_loss'], 'k-', label='Total Loss', alpha=0.7)
        plt.loglog(range(1, len(history['physics_loss']) + 1), history['physics_loss'], 'b-', label='Physics Loss', alpha=0.5)
        plt.loglog(range(1, len(history['data_loss']) + 1), history['data_loss'], 'g-', label='Data Loss (0-5s)', alpha=0.5)
        if 'mamba_extrap_loss' in history and len(history['mamba_extrap_loss']) > 0:
            plt.loglog(range(1, len(history['mamba_extrap_loss']) + 1), history['mamba_extrap_loss'], 'r-', label='Mamba Loss (5-10s)', alpha=0.5)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss Convergence')
        plt.legend()
        plt.grid(True)
        
        # 子图 4: 外推域 R1² 独立分析
        plt.subplot(2, 2, 4)
        if 'epoch_r2_extrap_speed' in history and len(history['epoch_r2_extrap_speed']) == len(epochs):
            r1_speed = history['epoch_r2_extrap_speed']
            plt.plot(epochs, r1_speed, 'r-o', label='Speed $R_1^2$ (5-10s)', markersize=3)
            
            # 标注最终值
            final_r1 = r1_speed[-1]
            plt.text(0.05, 0.95, f'Final $R_1^2$: {final_r1:.4f}', transform=plt.gca().transAxes, 
                     bbox=dict(facecolor='white', alpha=0.8))
            
            plt.xlabel('Epoch')
            plt.ylabel('$R_1^2$')
            plt.title('Extrapolation Capability Evolution (5-10s)')
            plt.grid(True)
            plt.ylim([0, 1.05])
        else:
            plt.text(0.5, 0.5, 'No Extrapolation Data Available', ha='center')
        
        plt.tight_layout()
        r2_analysis_path = os.path.join(args.outdir, 'segment_r2_analysis.png')
        plt.savefig(r2_analysis_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ 分段指标分析图已保存: {r2_analysis_path}")
    
    # ---------------------------------------------------------
    # 图表 B: 最终全时间域 (0-7s) 性能扫描
    # ---------------------------------------------------------
    print("\n[生成] 全时间域 (0-7s) 最终性能扫描...")
    try:
        # 加载 0-7s 的所有可用数据进行最终扫描
        # 注意：这里我们尝试加载全域数据，用于画出 R2 随时间 t 的变化曲线
        x_scan, y_scan, t_scan, u_scan, v_scan = load_temporal_data_range(0.0, 7.0)
        
        if x_scan is not None and len(x_scan) > 0:
            # 复用已有的 analyze_temporal_evolution 函数
            # 该函数会按时间点计算 MAE (因为只有部分时间点有 ground truth)
            temporal_analysis = analyze_temporal_evolution(model, x_scan, y_scan, t_scan, u_scan, v_scan)
            
            if temporal_analysis:
                plt.figure(figsize=(10, 6))
                
                # 绘制 Speed MAE 随时间变化
                # 0-5s 区域背景色
                plt.axvspan(0, 5.0, color='blue', alpha=0.1, label='Training Domain (0-5s)')
                # 5-7s 区域背景色
                plt.axvspan(5.0, 7.0, color='red', alpha=0.1, label='Extrapolation Domain (5-7s)')
                
                plt.plot(temporal_analysis['time'], temporal_analysis['speed_mae'], 'k-o', markersize=4, label='Speed MAE')
                
                plt.xlabel('Time (s)')
                plt.ylabel('MAE Error')
                plt.title('Error Distribution Over Time (0-7s)')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                scan_path = os.path.join(args.outdir, 'final_temporal_error_scan.png')
                plt.savefig(scan_path, dpi=150)
                plt.close()
                print(f"✅ 全时间域误差扫描图已保存: {scan_path}")
        else:
            print("⚠️ 未加载到全域扫描数据，跳过此图。")
            
    except Exception as e:
        print(f"❌ 生成全域扫描图出错: {e}")

    # ---------------------------------------------------------
    # 打印最终文件清单
    # ---------------------------------------------------------
    print(f"\n📁 所有结果已保存到: {args.outdir}")
    print(f"  - segment_r2_analysis.png (R² vs R1² 对比图)")
    print(f"  - final_temporal_error_scan.png (MAE 随时间分布)")
    print(f"  - training_history.pkl")
    print(f"  - final_model.pth")
    print("="*60)

