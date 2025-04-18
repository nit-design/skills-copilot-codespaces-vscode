%% QoS保障的无线视频流媒体组播控制 - 基于值迭代算法的策略优化 (增强版)
% 作者：nit-design
% 日期：2025-04-18

clear all;
close all;
clc;

%% ==================== 主程序 ====================
% 初始化系统参数
params = init_system_params();

% 初始化状态空间和动作空间
[S, A, state_mapping, action_mapping] = init_state_action_space(params);
fprintf('状态空间维度: %d, 动作空间维度: %d\n', length(S), length(A));

% 计算转移概率矩阵和奖励矩阵
fprintf('正在计算转移概率矩阵和奖励矩阵...\n');
[P, R] = compute_matrices(S, A, params, state_mapping, action_mapping);

% 值迭代算法求解最优策略 (增强版)
fprintf('正在通过增强版值迭代算法求解最优策略...\n');
[V, policy, deltas] = value_iteration_enhanced(S, A, P, R, params.gamma, params.epsilon_vi, params.max_iterations);

% 进行系统性能评估
fprintf('正在对最优策略进行性能评估...\n');
sim_results = simulate_policy(policy, S, A, P, R, params, state_mapping, action_mapping);

% 结果可视化
plot_results(sim_results, params);

fprintf('仿真完成!\n');

%% ==================== 系统参数初始化函数 ====================
function params = init_system_params()
    % 初始化系统参数
    % 返回包含所有系统参数的结构体
    
    %% 基本系统参数
    params.N = 5;                        % 组播组中的用户数
    params.channel_states = 3;           % 信道状态数量: Good(3), Medium(2), Bad(1)
    params.buffer_levels = 10;           % 缓冲区状态离散化级别
    params.video_layers = 3;             % 视频层数: BL(1), BL+EL1(2), BL+EL1+EL2(3)
    params.mcs_levels = 3;               % 调制编码方案级别数
    
    %% 信道状态转移参数 (Gilbert-Elliott模型)
    % 信道状态转移矩阵，行表示当前状态，列表示下一状态
    params.channel_transition = [
        0.7, 0.2, 0.1;  % Bad -> [Bad, Medium, Good]
        0.3, 0.4, 0.3;  % Medium -> [Bad, Medium, Good]
        0.1, 0.2, 0.7   % Good -> [Bad, Medium, Good]
    ];
    
    %% 缓冲区参数
    params.buffer_max = 1.0;             % 缓冲区最大占用率（归一化）
    params.buffer_min = 0.2;             % 缓冲区最小占用率阈值
    params.buffer_drain_rate = 0.1;      % 缓冲区消耗速率（每时间步）
    
    %% 视频码率参数 (Mbps)
    % 不同层级的视频码率
    params.video_rates = [1.5, 3.0, 6.0];  % BL, BL+EL1, BL+EL1+EL2
    
    %% MCS参数
    % 不同MCS级别的传输容量 (Mbps)
    params.mcs_capacity = [
        [2.0, 1.0, 0.5];    % MCS1 在 [Good, Medium, Bad] 信道下的容量
        [4.0, 2.0, 0.8];    % MCS2 在 [Good, Medium, Bad] 信道下的容量
        [8.0, 4.0, 1.2]     % MCS3 在 [Good, Medium, Bad] 信道下的容量
    ];
    
    % 不同MCS在不同信道状态下的误块率 (BLER)
    params.mcs_bler = [
        [0.01, 0.05, 0.30];  % MCS1 在 [Good, Medium, Bad] 信道下的BLER
        [0.05, 0.15, 0.50];  % MCS2 在 [Good, Medium, Bad] 信道下的BLER
        [0.10, 0.30, 0.80]   % MCS3 在 [Good, Medium, Bad] 信道下的BLER
    ];
    
    %% 功耗参数
    params.power_base = 1.0;             % 基础功耗
    params.power_layer = 0.2;            % 每增加一层的额外功耗
    params.power_mcs = 0.3;              % 每增加一级MCS的额外功耗
    
    %% 奖励函数权重
    params.alpha = 0.6;                  % 视频质量权重
    params.beta = 0.3;                   % 缓冲区惩罚权重
    params.gamma_power = 0.1;            % 功耗惩罚权重
    
    %% 视频质量参数 (PSNR)
    params.video_quality = [30, 35, 40]; % 各层的视频质量
    
    %% 算法参数
    params.gamma = 0.9;                  % 折扣因子
    params.epsilon_vi = 1e-6;            % 值迭代收敛阈值
    params.max_iterations = 5000;        % 最大迭代次数 (增加到5000确保充分收敛)
    params.simulation_steps = 1000;      % 仿真时间步数
end

%% ==================== 状态和动作空间初始化函数 ====================
function [S, A, state_mapping, action_mapping] = init_state_action_space(params)
    % 初始化状态空间和动作空间
    % 输入:
    %   params - 系统参数
    % 输出:
    %   S - 状态空间索引向量
    %   A - 动作空间索引向量
    %   state_mapping - 状态索引到实际状态的映射
    %   action_mapping - 动作索引到实际动作的映射

    % 状态空间维度计算
    % 状态空间 = 信道状态 × 缓冲区状态 × 视频层状态
    state_dim = params.N * params.channel_states * params.buffer_levels * params.video_layers;
    S = 1:state_dim;
    
    % 创建状态映射
    state_mapping = cell(state_dim, 1);
    idx = 1;
    
    % 遍历每种可能的状态组合
    for v = 1:params.video_layers
        for c = 1:params.N
            for ch_state = 1:params.channel_states
                for b = 1:params.buffer_levels
                    % 创建状态向量
                    channel_vec = ones(params.N, 1);
                    channel_vec(c) = ch_state;
                    
                    buffer_vec = ones(params.N, 1) * 0.5;  % 默认缓冲区填充一半
                    buffer_vec(c) = (b - 1) / (params.buffer_levels - 1);  % 归一化到 [0, 1]
                    
                    state_mapping{idx} = struct('channel', channel_vec, ...
                                                'buffer', buffer_vec, ...
                                                'video_layer', v);
                    idx = idx + 1;
                end
            end
        end
    end
    
    % 动作空间初始化
    % 动作空间 = 视频层 × MCS级别
    action_dim = params.video_layers * params.mcs_levels;
    A = 1:action_dim;
    
    % 创建动作映射
    action_mapping = cell(action_dim, 1);
    idx = 1;
    for l = 1:params.video_layers
        for m = 1:params.mcs_levels
            action_mapping{idx} = struct('layer', l, 'mcs', m);
            idx = idx + 1;
        end
    end
end

%% ==================== 转移概率和奖励矩阵计算函数 ====================
function [P, R] = compute_matrices(S, A, params, state_mapping, action_mapping)
    % 计算转移概率矩阵和奖励矩阵
    % 输入:
    %   S - 状态空间索引向量
    %   A - 动作空间索引向量
    %   params - 系统参数
    %   state_mapping - 状态索引到实际状态的映射
    %   action_mapping - 动作索引到实际动作的映射
    % 输出:
    %   P - 转移概率矩阵，P(s,a,s')表示从状态s执行动作a转移到状态s'的概率
    %   R - 奖励矩阵，R(s,a)表示在状态s执行动作a获得的即时奖励

    n_states = length(S);
    n_actions = length(A);
    
    % 初始化转移概率矩阵和奖励矩阵
    P = zeros(n_states, n_actions, n_states);
    R = zeros(n_states, n_actions);
    
    % 计算每个状态-动作对的转移概率和奖励
    fprintf('总计算量: %d 个状态-动作组合\n', n_states * n_actions);
    progress_step = floor(n_states / 10);  % 每10%显示一次进度
    
    for s_idx = 1:n_states
        % 显示计算进度
        if mod(s_idx, progress_step) == 0
            fprintf('计算进度: %.1f%%\n', 100 * s_idx / n_states);
        end
        
        s = state_mapping{s_idx};
        
        for a_idx = 1:n_actions
            a = action_mapping{a_idx};
            
            % 计算奖励 R(s,a)
            R(s_idx, a_idx) = compute_reward(s, a, params);
            
            % 计算转移概率 P(s'|s,a)
            for s_next_idx = 1:n_states
                s_next = state_mapping{s_next_idx};
                P(s_idx, a_idx, s_next_idx) = compute_transition_prob(s, a, s_next, params);
            end
            
            % 归一化转移概率，确保对每个(s,a)，sum(P(s,a,:)) = 1
            prob_sum = sum(P(s_idx, a_idx, :));
            if prob_sum > 0
                P(s_idx, a_idx, :) = P(s_idx, a_idx, :) / prob_sum;
            else
                % 如果所有转移概率都是0，设置均匀分布
                P(s_idx, a_idx, :) = 1 / n_states;
            end
        end
    end
    fprintf('矩阵计算完成！\n');
end

function reward = compute_reward(state, action, params)
    % 计算在状态s采取动作a的奖励值
    % 输入:
    %   state - 当前状态
    %   action - 采取的动作
    %   params - 系统参数
    % 输出:
    %   reward - 奖励值

    % 提取视频层、缓冲状态和MCS
    video_layer = state.video_layer;
    buffer_state = mean(state.buffer);  % 使用平均缓冲状态用于奖励计算
    mcs_level = action.mcs;
    layer = action.layer;
    
    % 计算视频质量奖励
    quality_reward = params.video_quality(layer);
    
    % 计算缓冲区惩罚
    buffer_penalty = params.beta * (params.buffer_max - buffer_state);
    
    % 计算功耗惩罚
    power_consumption = params.power_base + layer * params.power_layer + mcs_level * params.power_mcs;
    power_penalty = params.gamma_power * power_consumption;
    
    % 计算总奖励
    reward = params.alpha * quality_reward - buffer_penalty - power_penalty;
    
    % 添加QoS约束惩罚
    % 如果选择的层和MCS不满足带宽约束，给予严重惩罚
    min_channel_state = min(state.channel);
    required_rate = params.video_rates(layer);
    available_capacity = params.mcs_capacity(mcs_level, min_channel_state);
    
    % 带宽约束检查
    if required_rate > available_capacity
        reward = reward - 100;  % 严重惩罚
    end
    
    % 时延约束检查（基于缓冲区最小用户）
    min_buffer = min(state.buffer);
    if min_buffer < params.buffer_min
        reward = reward - 50 * (params.buffer_min - min_buffer);  % 缓冲区过低惩罚
    end
end

function prob = compute_transition_prob(state, action, next_state, params)
    % 计算从state通过action转移到next_state的概率
    % 输入:
    %   state - 当前状态
    %   action - 采取的动作
    %   next_state - 下一状态
    %   params - 系统参数
    % 输出:
    %   prob - 转移概率

    % 分解为三个独立的转移概率
    p_channel = compute_channel_transition(state.channel, next_state.channel, params);
    p_buffer = compute_buffer_transition(state.buffer, next_state.buffer, action, state.channel, params);
    p_layer = compute_layer_transition(state.video_layer, next_state.video_layer, action.layer);
    
    % 联合概率（假设转移独立）
    prob = p_channel * p_buffer * p_layer;
end

function prob = compute_channel_transition(current_channel, next_channel, params)
    % 计算信道状态转移概率
    % 输入:
    %   current_channel - 当前信道状态向量
    %   next_channel - 下一信道状态向量
    %   params - 系统参数
    % 输出:
    %   prob - 转移概率

    prob = 1.0;
    for i = 1:length(current_channel)
        curr_state = current_channel(i);
        next_state = next_channel(i);
        prob = prob * params.channel_transition(curr_state, next_state);
    end
end

function prob = compute_buffer_transition(current_buffer, next_buffer, action, channel_state, params)
    % 计算缓冲区状态转移概率
    % 输入:
    %   current_buffer - 当前缓冲区状态向量
    %   next_buffer - 下一缓冲区状态向量
    %   action - 采取的动作
    %   channel_state - 当前信道状态向量
    %   params - 系统参数
    % 输出:
    %   prob - 转移概率

    prob = 1.0;
    layer = action.layer;
    mcs = action.mcs;
    
    for i = 1:length(current_buffer)
        % 计算当前用户的数据接收速率
        ch_state = channel_state(i);
        rate = params.video_rates(layer);
        bler = params.mcs_bler(mcs, ch_state);
        
        % 计算缓冲区的预期下一个状态
        expected_buffer = max(0, min(1, current_buffer(i) + rate * (1 - bler) * 0.1 - params.buffer_drain_rate));
        
        % 简化：如果预期状态接近实际下一状态，给予较高概率
        if abs(expected_buffer - next_buffer(i)) < 0.1
            user_prob = 0.8;
        elseif abs(expected_buffer - next_buffer(i)) < 0.2
            user_prob = 0.15;
        else
            user_prob = 0.05;
        end
        
        prob = prob * user_prob;
    end
end

function prob = compute_layer_transition(current_layer, next_layer, action_layer)
    % 计算视频层转移概率
    % 输入:
    %   current_layer - 当前视频层
    %   next_layer - 下一视频层
    %   action_layer - 动作选择的视频层
    % 输出:
    %   prob - 转移概率

    % 简化视频层转移：动作决定的层就是下一层
    if next_layer == action_layer
        prob = 1.0;
    else
        prob = 0.0;
    end
end

%% ==================== 增强版值迭代求解最优策略函数 ====================
function [V, policy, deltas] = value_iteration_enhanced(S, A, P, R, gamma, epsilon, max_iter)
    % 增强版值迭代算法求解最优策略（带收敛可视化）
    % 输入:
    %   S - 状态空间索引向量
    %   A - 动作空间索引向量
    %   P - 转移概率矩阵
    %   R - 奖励矩阵
    %   gamma - 折扣因子
    %   epsilon - 收敛阈值
    %   max_iter - 最大迭代次数
    % 输出:
    %   V - 最优值函数
    %   policy - 最优策略
    %   deltas - 每次迭代的值函数变化记录

    n_states = length(S);
    n_actions = length(A);
    
    % 初始化值函数和策略
    V = zeros(n_states, 1);
    policy = ones(n_states, 1);
    deltas = zeros(max_iter, 1);  % 记录每次迭代的delta值
    
    % 值迭代
    iteration = 0;
    converged = false;
    
    fprintf('开始值迭代算法...\n');
    while ~converged && iteration < max_iter
        iteration = iteration + 1;
        delta = 0;
        
        % 对每个状态计算新的值函数
        for s = 1:n_states
            v_old = V(s);
            
            % 对所有动作计算Q值
            Q = zeros(1, n_actions);
            for a = 1:n_actions
                % 计算期望奖励
                Q(a) = R(s, a);
                
                % 加上折扣未来奖励
                for s_next = 1:n_states
                    Q(a) = Q(a) + gamma * P(s, a, s_next) * V(s_next);
                end
            end
            
            % 更新值函数和策略
            [V(s), policy(s)] = max(Q);
            
            % 更新收敛度量
            delta = max(delta, abs(v_old - V(s)));
        end
        
        % 记录当前迭代的delta值
        deltas(iteration) = delta;
        
        % 检查是否收敛
        if delta < epsilon
            converged = true;
        end
        
        % 打印进度
        if mod(iteration, 20) == 0
            fprintf('迭代 %d, 最大值变化: %.6f\n', iteration, delta);
        end
    end
    
    % 裁剪未使用的delta记录
    deltas = deltas(1:iteration);
    
    % 使用if-else代替三元运算符
    if converged
        status = '收敛';
    else
        status = '未收敛';
    end
    fprintf('值迭代在 %d 次迭代后%s, 最终最大值变化: %.6f\n', iteration, status, delta);
    
    % 可视化收敛过程
    figure('Name', '值迭代收敛曲线', 'Position', [200, 200, 800, 400]);
    
    % 普通尺度
    subplot(1, 2, 1);
    plot(1:iteration, deltas, 'b-', 'LineWidth', 2);
    title('值迭代收敛曲线 (普通尺度)');
    xlabel('迭代次数');
    ylabel('最大值变化');
    grid on;
    
    % 对数尺度
    subplot(1, 2, 2);
    semilogy(1:iteration, deltas, 'r-', 'LineWidth', 2);
    title('值迭代收敛曲线 (对数尺度)');
    xlabel('迭代次数');
    ylabel('最大值变化 (log)');
    grid on;
    
    saveas(gcf, 'value_iteration_convergence.png');
    fprintf('收敛曲线已保存为: value_iteration_convergence.png\n');
end

%% ==================== 策略仿真评估函数 ====================
function results = simulate_policy(policy, S, A, P, R, params, state_mapping, action_mapping)
    % 模拟最优策略的性能
    % 输入:
    %   policy - 最优策略
    %   S - 状态空间索引向量
    %   A - 动作空间索引向量
    %   P - 转移概率矩阵
    %   R - 奖励矩阵
    %   params - 系统参数
    %   state_mapping - 状态索引到实际状态的映射
    %   action_mapping - 动作索引到实际动作的映射
    % 输出:
    %   results - 包含仿真结果的结构体

    % 初始化结果存储
    T = params.simulation_steps;
    results = struct();
    results.rewards = zeros(1, T);
    results.video_quality = zeros(1, T);
    results.buffer_levels = zeros(params.N, T);
    results.selected_layers = zeros(1, T);
    results.selected_mcs = zeros(1, T);
    results.channel_states = zeros(params.N, T);
    results.outage_ratio = 0;
    results.avg_buffer = zeros(1, T);
    
    % 随机初始化状态
    s_idx = randi(length(S));
    s = state_mapping{s_idx};
    
    % 记录缓冲区低于阈值的次数
    buffer_outage_count = 0;
    
    % 仿真循环
    fprintf('开始策略仿真评估，总步数: %d\n', T);
    progress_step = floor(T / 10);  % 每10%显示一次进度
    
    for t = 1:T
        % 显示仿真进度
        if mod(t, progress_step) == 0
            fprintf('仿真进度: %.1f%%\n', 100 * t / T);
        end
        
        % 获取当前策略下的动作
        a_idx = policy(s_idx);
        a = action_mapping{a_idx};
        
        % 记录当前步骤的结果
        results.rewards(t) = R(s_idx, a_idx);
        results.video_quality(t) = params.video_quality(a.layer);
        results.buffer_levels(:, t) = s.buffer;
        results.avg_buffer(t) = mean(s.buffer);
        results.selected_layers(t) = a.layer;
        results.selected_mcs(t) = a.mcs;
        results.channel_states(:, t) = s.channel;
        
        % 检查缓冲区是否低于阈值
        if min(s.buffer) < params.buffer_min
            buffer_outage_count = buffer_outage_count + 1;
        end
        
        % 确定下一状态
        next_s_idx = sample_next_state(s_idx, a_idx, P);
        
        % 更新状态
        s_idx = next_s_idx;
        s = state_mapping{s_idx};
    end
    
    % 计算平均指标
    results.avg_reward = mean(results.rewards);
    results.avg_quality = mean(results.video_quality);
    results.outage_ratio = buffer_outage_count / T;
    results.avg_layer = mean(results.selected_layers);
    results.avg_mcs = mean(results.selected_mcs);
    
    fprintf('仿真评估完成！\n');
end

function next_state = sample_next_state(s, a, P)
    % 根据转移概率采样下一个状态
    % 输入:
    %   s - 当前状态索引
    %   a - 当前动作索引
    %   P - 转移概率矩阵
    % 输出:
    %   next_state - 下一状态索引

    probs = squeeze(P(s, a, :));
    cdf = cumsum(probs);
    sample = rand();
    next_state = find(cdf >= sample, 1, 'first');
    
    % 如果概率和不为1（浮点误差），选择最后一个状态
    if isempty(next_state)
        next_state = length(probs);
    end
end

%% ==================== 结果可视化函数 ====================
function plot_results(results, params)
    % 可视化仿真结果
    % 输入:
    %   results - 仿真结果结构体
    %   params - 系统参数
    
    T = length(results.rewards);
    time = 1:T;
    
    % 创建图表
    figure('Name', 'QoS保障的无线视频流媒体组播控制性能评估', 'Position', [100, 100, 1200, 800]);
    
    % 1. 累积奖励图
    subplot(3, 2, 1);
    cumreward = cumsum(results.rewards) ./ (1:T);
    plot(time, cumreward, 'b-', 'LineWidth', 2);
    title('平均累积奖励');
    xlabel('时间步');
    ylabel('累积奖励');
    grid on;
    
    % 2. 视频质量和选择的层数图
    subplot(3, 2, 2);
    % 左Y轴
    yyaxis left;
    plot(time, results.video_quality, 'b-', 'LineWidth', 1.5);
    ylabel('视频质量 (PSNR)');
    % 右Y轴
    yyaxis right;
    plot(time, results.selected_layers, 'r--', 'LineWidth', 1.5);
    ylabel('选择的层数');
    title('视频质量和层选择');
    xlabel('时间步');
    grid on;
    legend('视频质量', '选择的层数');
    
    % 3. 平均缓冲区水平图
    subplot(3, 2, 3);
    plot(time, results.avg_buffer, 'g-', 'LineWidth', 2);
    hold on;
    plot([1, T], [params.buffer_min, params.buffer_min], 'r--', 'LineWidth', 1.5);
    hold off;
    title('平均缓冲区水平');
    xlabel('时间步');
    ylabel('缓冲区占用率');
    ylim([0, 1]);
    grid on;
    legend('平均缓冲水平', '缓冲区阈值');
    
    % 4. 选择的MCS图
    subplot(3, 2, 4);
    plot(time, results.selected_mcs, 'm-', 'LineWidth', 2);
    title('调制编码方案选择');
    xlabel('时间步');
    ylabel('MCS级别');
    ylim([0.5, params.mcs_levels+0.5]);
    yticks(1:params.mcs_levels);
    grid on;
    
    % 5. 信道状态热图
    subplot(3, 2, 5);
    imagesc(results.channel_states');
    title('用户信道状态');
    xlabel('时间步');
    ylabel('用户编号');
    yticks(1:params.N);
    colorbar('Ticks', 1:params.channel_states, ...
             'TickLabels', {'差', '中', '好'});
    
    % 6. 性能指标汇总
    subplot(3, 2, 6);
    performance = [
        results.avg_reward;
        results.avg_quality;
        results.avg_layer;
        results.avg_mcs;
        results.outage_ratio * 100
    ];
    
    bar(performance, 'FaceColor', [0.3, 0.6, 0.9]);
    title('性能指标汇总');
    xticks(1:5);
    xticklabels({'平均奖励', '平均质量', '平均层数', '平均MCS', '缓冲区中断率(%)'});
    xtickangle(45);
    grid on;
    
    % 添加总标题
    sgtitle('基于MDP的无线视频流媒体组播控制性能评估', 'FontSize', 14, 'FontWeight', 'bold');
    
    % 调整布局
    set(gcf, 'Color', 'w');
    
    % 保存图表
    saveas(gcf, 'qos_multicast_control_results.png');
    fprintf('性能评估结果已保存为: qos_multicast_control_results.png\n');
    
    % 输出关键性能指标
    fprintf('\n性能指标汇总：\n');
    fprintf('平均奖励: %.4f\n', results.avg_reward);
    fprintf('平均视频质量 (PSNR): %.2f dB\n', results.avg_quality);
    fprintf('平均选择层数: %.2f\n', results.avg_layer);
    fprintf('平均选择MCS级别: %.2f\n', results.avg_mcs);
    fprintf('缓冲区中断率: %.2f%%\n', results.outage_ratio * 100);
end