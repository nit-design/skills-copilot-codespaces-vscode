%% QoS保障的无线视频流媒体组播控制
% 基于马尔可夫决策过程(MDP)的值迭代优化算法
clear all; close all; clc;

%% 系统参数设置
% 用户参数
N = 5;               % 用户数量
n_channel_states = 3; % 信道状态数量（好/中/差）
n_buffer_states = 10; % 离散化的缓冲区状态数量（0-1，步长为0.1）
n_video_layers = 3;  % 视频层数（基础层BL, 增强层EL1, 增强层EL2）
n_mcs_levels = 3;    % 调制编码方案(MCS)级别数量

% 动作空间定义
% l=0: 仅基础层, l=1: BL+EL1, l=2: BL+EL1+EL2
% mcs=1,2,3: 不同调制方案，对应QPSK, 16QAM, 64QAM等
n_layers = 3;
n_actions = n_layers * n_mcs_levels;

% 状态空间离散化
buffer_levels = linspace(0, 1, n_buffer_states); % 缓冲区状态离散化，从0到1
channel_states = 1:n_channel_states; % 1=差, 2=中, 3=好

% QoS参数
B_min = 0.2;         % 最小缓冲区阈值，低于此值可能导致播放卡顿
epsilon = 0.1;       % 允许的缓冲区不足概率上限

% MDP算法参数
gamma = 0.9;         % 折扣因子，决定未来奖励的权重
max_iter = 1000;     % 值迭代的最大迭代次数
convergence_threshold = 1e-6; % 收敛判断阈值

%% 信道模型参数
% Gilbert-Elliott信道模型转移概率矩阵
% 保持在同一状态的概率较高（时间相关性）
P_channel = zeros(n_channel_states, n_channel_states, N);
for i = 1:N
    % 为每个用户设置不同的信道特性
    P_channel(:,:,i) = [
        0.7, 0.2, 0.1;  % 差状态的转移概率
        0.3, 0.4, 0.3;  % 中状态的转移概率
        0.1, 0.3, 0.6   % 好状态的转移概率
    ];
    % 确保每行概率和为1
    P_channel(:,:,i) = P_channel(:,:,i) ./ sum(P_channel(:,:,i), 2);
end

%% 传输速率和误块率(BLER)参数
% 每种层和MCS组合的传输速率(Mbps)
R = zeros(n_layers, n_mcs_levels);
R(1,:) = [2, 3, 4];     % 基础层在MCS 1,2,3下的速率
R(2,:) = [3.5, 5, 6.5]; % BL+EL1在MCS 1,2,3下的速率
R(3,:) = [5, 7, 9];     % BL+EL1+EL2在MCS 1,2,3下的速率

% 每种信道状态和MCS组合的块错误率(BLER)
BLER = zeros(n_channel_states, n_mcs_levels);
BLER(1,:) = [0.2, 0.4, 0.6];   % 差信道下MCS 1,2,3的BLER
BLER(2,:) = [0.05, 0.15, 0.3]; % 中信道下MCS 1,2,3的BLER
BLER(3,:) = [0.01, 0.05, 0.1]; % 好信道下MCS 1,2,3的BLER

%% 功率消耗模型
p_base = 1;        % 基础功率消耗
p_layer = 0.2;     % 每增加一层的额外功率
p_mcs = [0, 0.1, 0.2]; % 每种MCS级别的额外功率

%% 视频质量参数
% 不同层的PSNR值（视频质量度量）
Q = [30, 35, 40]; % BL, BL+EL1, BL+EL1+EL2的质量

%% 奖励函数权重（来自AHP层次分析法）
alpha = 0.6;     % 视频质量权重
beta = 0.3;      % 缓冲区不足惩罚权重
gamma_power = 0.1; % 功率消耗权重

%% 初始化状态-值函数
% 为简化计算，使用代表性用户进行状态值计算
% 状态索引: (信道状态, 视频层, 缓冲区状态)
V = zeros(n_channel_states, n_video_layers, n_buffer_states);
V_next = V;

%% 值迭代算法实现
converged = false;
iter = 0;
delta_history = zeros(1, max_iter);

while ~converged && iter < max_iter
    iter = iter + 1;
    
    % 遍历每个状态
    for c = 1:n_channel_states
        for l = 1:n_video_layers
            for b_idx = 1:n_buffer_states
                b = buffer_levels(b_idx);
                max_value = -inf;
                
                % 遍历每个动作
                for action_l = 1:n_layers
                    for action_mcs = 1:n_mcs_levels
                        % 检查带宽约束
                        % 为最差用户计算香农容量
                        SNR_dB = [5, 15, 25]; % 差/中/好信道对应的信噪比(dB)
                        SNR_linear = 10.^(SNR_dB/10); % 转换为线性SNR
                        C = log2(1 + SNR_linear(c)); % 简化的容量计算
                        
                        % 如果速率超过容量，跳过此动作
                        if R(action_l, action_mcs) > C
                            continue;
                        end
                        
                        % 计算期望值
                        expected_value = 0;
                        
                        % 计算奖励
                        % 视频质量奖励
                        quality_reward = alpha * Q(action_l);
                        
                        % 缓冲区不足惩罚
                        buffer_penalty = beta * (1 - b);
                        
                        % 功率消耗惩罚
                        power_penalty = gamma_power * (p_base + action_l * p_layer + p_mcs(action_mcs));
                        
                        % 总奖励
                        reward = quality_reward - buffer_penalty - power_penalty;
                        
                        % 基于动作计算下一个缓冲区水平
                        consumption_rate = 1; % 视频播放速率(归一化)
                        delta_t = 1; % 时间步长(归一化)
                        
                        % 遍历可能的下一个信道状态
                        for next_c = 1:n_channel_states
                            p_channel_transition = P_channel(c, next_c, 1); % 使用用户1作为代表
                            
                            % 使用缓冲区演化公式计算下一个缓冲区水平
                            next_b = max(0, min(1, b + (R(action_l, action_mcs) * delta_t * (1 - BLER(c, action_mcs)) - consumption_rate * delta_t)));
                            
                            % 找到最接近的离散缓冲区水平
                            [~, next_b_idx] = min(abs(buffer_levels - next_b));
                            
                            % 遍历可能的下一个视频层
                            for next_l = 1:n_video_layers
                                % 计算到下一个层的转移概率
                                if next_l == action_l
                                    p_layer_transition = 0.8; % 保持在同一层的高概率
                                else
                                    p_layer_transition = 0.2 / (n_video_layers - 1); % 切换层的低概率
                                end
                                
                                % 更新期望值
                                expected_value = expected_value + p_channel_transition * p_layer_transition * (reward + gamma * V(next_c, next_l, next_b_idx));
                            end
                        end
                        
                        % 如果找到更好的动作值，则更新最大值
                        if expected_value > max_value
                            max_value = expected_value;
                        end
                    end
                end
                
                % 更新状态-值函数
                V_next(c, l, b_idx) = max_value;
            end
        end
    end
    
    % 检查收敛性
    delta = max(abs(V_next(:) - V(:)));
    delta_history(iter) = delta;
    
    if delta < convergence_threshold
        converged = true;
    end
    
    V = V_next;
end

fprintf('值迭代在%d次迭代后收敛，delta = %.6f\n', iter, delta);

%% 提取最优策略
% 对于每个状态，找到使值函数最大化的动作
optimal_policy = zeros(n_channel_states, n_video_layers, n_buffer_states, 2); % 2维表示(layer, mcs)

for c = 1:n_channel_states
    for l = 1:n_video_layers
        for b_idx = 1:n_buffer_states
            b = buffer_levels(b_idx);
            max_value = -inf;
            best_action_l = 1;
            best_action_mcs = 1;
            
            % 遍历每个动作
            for action_l = 1:n_layers
                for action_mcs = 1:n_mcs_levels
                    % 信道容量约束
                    SNR_dB = [5, 15, 25];
                    SNR_linear = 10.^(SNR_dB/10);
                    C = log2(1 + SNR_linear(c));
                    
                    if R(action_l, action_mcs) > C
                        continue;
                    end
                    
                    % 计算期望值
                    expected_value = 0;
                    
                    % 计算奖励
                    quality_reward = alpha * Q(action_l);
                    buffer_penalty = beta * (1 - b);
                    power_penalty = gamma_power * (p_base + action_l * p_layer + p_mcs(action_mcs));
                    reward = quality_reward - buffer_penalty - power_penalty;
                    
                    % 计算状态转移
                    for next_c = 1:n_channel_states
                        p_channel_transition = P_channel(c, next_c, 1);
                        
                        next_b = max(0, min(1, b + (R(action_l, action_mcs) * (1 - BLER(c, action_mcs)) - 1)));
                        [~, next_b_idx] = min(abs(buffer_levels - next_b));
                        
                        for next_l = 1:n_video_layers
                            if next_l == action_l
                                p_layer_transition = 0.8;
                            else
                                p_layer_transition = 0.2 / (n_video_layers - 1);
                            end
                            
                            expected_value = expected_value + p_channel_transition * p_layer_transition * (reward + gamma * V(next_c, next_l, next_b_idx));
                        end
                    end
                    
                    % 如果找到更好的动作，则更新最佳动作
                    if expected_value > max_value
                        max_value = expected_value;
                        best_action_l = action_l;
                        best_action_mcs = action_mcs;
                    end
                end
            end
            
            % 存储最优动作
            optimal_policy(c, l, b_idx, 1) = best_action_l;
            optimal_policy(c, l, b_idx, 2) = best_action_mcs;
        end
    end
end

%% 系统演化模拟
T = 1000; % 时间步数
selected_user = 1; % 选择一个用户作为可视化代表

% 初始化模拟
sim_channel_state = randi(n_channel_states); % 初始信道状态
sim_video_layer = 1; % 从基础层开始
sim_buffer_level = 0.5; % 从半满缓冲区开始
sim_buffer_idx = find(buffer_levels >= sim_buffer_level, 1);

% 历史跟踪
channel_history = zeros(1, T);
layer_history = zeros(1, T);
mcs_history = zeros(1, T);
buffer_history = zeros(1, T);
quality_history = zeros(1, T);
power_history = zeros(1, T);
reward_history = zeros(1, T);

% 运行模拟
for t = 1:T
    % 记录当前状态
    channel_history(t) = sim_channel_state;
    buffer_history(t) = buffer_levels(sim_buffer_idx);
    
    % 获取当前状态的最优动作
    action_l = optimal_policy(sim_channel_state, sim_video_layer, sim_buffer_idx, 1);
    action_mcs = optimal_policy(sim_channel_state, sim_video_layer, sim_buffer_idx, 2);
    
    layer_history(t) = action_l;
    mcs_history(t) = action_mcs;
    
    % 计算奖励组成部分
    quality = Q(action_l);
    quality_history(t) = quality;
    
    power = p_base + action_l * p_layer + p_mcs(action_mcs);
    power_history(t) = power;
    
    reward = alpha * quality - beta * (1 - buffer_levels(sim_buffer_idx)) - gamma_power * power;
    reward_history(t) = reward;
    
    % 模拟状态转移
    % 信道转移
    sim_channel_state = randsample(1:n_channel_states, 1, true, P_channel(sim_channel_state, :, selected_user));
    
    % 缓冲区转移
    if rand() > BLER(sim_channel_state, action_mcs)
        % 成功传输
        buffer_update = buffer_levels(sim_buffer_idx) + (R(action_l, action_mcs) - 1);
    else
        % 传输失败，只有消耗
        buffer_update = buffer_levels(sim_buffer_idx) - 1;
    end
    
    buffer_update = max(0, min(1, buffer_update));
    [~, sim_buffer_idx] = min(abs(buffer_levels - buffer_update));
    
    % 视频层转移
    if rand() < 0.8
        sim_video_layer = action_l;
    else
        % 随机层变化
        other_layers = setdiff(1:n_video_layers, action_l);
        sim_video_layer = other_layers(randi(length(other_layers)));
    end
end

%% 计算性能指标
% 服务质量(QoS)
avg_quality = mean(quality_history);
quality_variance = var(quality_history);
layer_switches = sum(diff(layer_history) ~= 0);

% 缓冲区不足事件
underflow_events = sum(buffer_history < B_min);
underflow_probability = underflow_events / T;

% 平均功率消耗
avg_power = mean(power_history);

% 平均奖励
avg_reward = mean(reward_history);

%% 显示结果
fprintf('=== 性能指标 ===\n');
fprintf('平均视频质量(PSNR): %.2f dB\n', avg_quality);
fprintf('质量方差: %.4f\n', quality_variance);
fprintf('层切换频率: %.4f%%\n', 100 * layer_switches / T);
fprintf('缓冲区不足概率: %.4f%%\n', 100 * underflow_probability);
fprintf('平均功率消耗: %.4f\n', avg_power);
fprintf('平均奖励: %.4f\n', avg_reward);

%% 可视化
figure;

% 图1: 信道状态演化
subplot(5, 1, 1);
plot(1:T, channel_history, 'b-');
ylabel('信道状态');
title('系统随时间的演化');
ylim([0.5 3.5]);
yticks(1:3);
yticklabels({'差', '中', '好'});
grid on;

% 图2: 视频层选择
subplot(5, 1, 2);
plot(1:T, layer_history, 'r-');
ylabel('视频层');
ylim([0.5 3.5]);
yticks(1:3);
yticklabels({'基础层', 'BL+EL1', 'BL+EL1+EL2'});
grid on;

% 图3: MCS级别选择
subplot(5, 1, 3);
plot(1:T, mcs_history, 'g-');
ylabel('MCS级别');
ylim([0.5 3.5]);
yticks(1:3);
yticklabels({'MCS1', 'MCS2', 'MCS3'});
grid on;

% 图4: 缓冲区水平
subplot(5, 1, 4);
plot(1:T, buffer_history, 'm-');
hold on;
plot([1 T], [B_min B_min], 'k--', 'LineWidth', 1);
ylabel('缓冲区水平');
ylim([0 1]);
grid on;
legend('缓冲区水平', '阈值');

% 图5: 奖励
subplot(5, 1, 5);
plot(1:T, reward_history, 'k-');
xlabel('时间步');
ylabel('奖励');
grid on;

% 值迭代收敛性
figure;
semilogy(1:iter, delta_history(1:iter), 'b-', 'LineWidth', 2);
xlabel('迭代次数');
ylabel('Delta值（对数尺度）');
title('值迭代算法收敛性');
grid on;

% 策略可视化
figure;
for c = 1:n_channel_states
    subplot(1, 3, c);
    
    % 创建一个矩阵来可视化此信道状态下的策略
    policy_matrix = zeros(n_buffer_states, n_video_layers);
    
    for l = 1:n_video_layers
        for b = 1:n_buffer_states
            % 将动作编码为单一值: 3*(l-1) + mcs
            action_l = optimal_policy(c, l, b, 1);
            action_mcs = optimal_policy(c, l, b, 2);
            policy_matrix(b, l) = 3*(action_l-1) + action_mcs;
        end
    end
    
    imagesc(policy_matrix);
    
    % 标签
    xlabel('当前视频层');
    ylabel('缓冲区状态');
    title(['信道状态 ' num2str(c) ' 的最优策略']);
    
    % 自定义x轴
    xticks(1:3);
    xticklabels({'基础层', 'BL+EL1', 'BL+EL1+EL2'});
    
    % 自定义y轴
    yticks(1:2:n_buffer_states);
    y_labels = {};
    for i = 1:2:n_buffer_states
        y_labels{end+1} = num2str(buffer_levels(i), '%.1f');
    end
    yticklabels(y_labels);
    
    colormap(jet);
    colorbar;
end

% QoS性能与功率消耗的权衡
figure;
scatter(power_history, quality_history, 15, buffer_history, 'filled');
xlabel('功率消耗');
ylabel('视频质量(PSNR)');
title('QoS与功率消耗的权衡关系');
colormap(jet);
c = colorbar;
c.Label.String = '缓冲区水平';
grid on;