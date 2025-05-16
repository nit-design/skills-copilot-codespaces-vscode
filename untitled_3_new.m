%% QoS保障的无线视频流媒体组播控制 - 基于值迭代算法的策略优化
% 作者：nit-design
% 日期：2025-05-14 (修改版，支持多场景仿真)

clear all; % 清除工作空间所有变量
close all; % 关闭所有打开的图形窗口
clc;       % 清空命令行窗口

%% ==================== 主程序 ====================
% 初始化系统参数
params = init_system_params();

% 初始化状态空间和动作空间
[S, A, state_mapping, action_mapping] = init_state_action_space(params);
fprintf('状态空间维度: %d, 动作空间维度: %d\n', length(S), length(A));

% 计算转移概率矩阵P和奖励矩阵R
fprintf('正在计算转移概率矩阵和奖励矩阵...\n');
[P, R] = compute_matrices(S, A, params, state_mapping, action_mapping);

% 值迭代算法求解最优策略 - 这部分只运行一次，为场景一和后续可能的比较做准备
fprintf('正在通过值迭代算法求解最优策略 (MDP Optimal)...\n');
[V, optimal_policy_mdp, deltas_mdp] = value_iteration_enhanced(S, A, P, R, params.gamma, params.epsilon_vi, params.max_iterations);

% --- 场景选择与执行 ---
num_scenarios = 3; % 1: MDP 最优策略, 2: 静态高质量策略, 3: 信道反应式策略
all_sim_results = cell(num_scenarios, 1); % 用于存储所有场景的仿真结果
scenario_names = {'MDP 最优策略', '静态高质量策略 (Static-HQ)', '信道反应式策略 (CR)'};

for scenario_idx = 1:num_scenarios
    fprintf('\n\n===== 正在运行仿真场景: %s =====\n', scenario_names{scenario_idx});

    current_policy_to_simulate_vector = []; % MDP策略向量 (如果适用)
    policy_type_str_for_sim = ''; % 当前场景的策略类型字符串

    if scenario_idx == 1 % 场景一: MDP 最优策略
        current_policy_to_simulate_vector = optimal_policy_mdp;
        policy_type_str_for_sim = 'mdp';
        fprintf('使用 MDP 最优策略.\n');
    elseif scenario_idx == 2 % 场景二: Static-HQ 策略
        policy_type_str_for_sim = 'static_hq';
        fprintf('使用静态高质量 (Static-HQ) 策略.\n');
    elseif scenario_idx == 3 % 场景二: Channel-Reactive 策略
        policy_type_str_for_sim = 'channel_reactive';
        fprintf('使用信道反应式 (CR) 策略.\n');
    end

    % 进行系统性能评估 (使用修改后的灵活仿真函数)
    fprintf('正在对策略 [%s] 进行性能评估...\n', scenario_names{scenario_idx});
    sim_results_current_scenario = simulate_policy_flexible(current_policy_to_simulate_vector, S, A, P, R, params, state_mapping, action_mapping, policy_type_str_for_sim);
    all_sim_results{scenario_idx} = sim_results_current_scenario;

    % 结果可视化 (为每个场景单独生成图表)
    plot_results_flexible(sim_results_current_scenario, params, scenario_names{scenario_idx});
end

fprintf('\n\n===== 所有场景仿真完成! =====\n');

% 可选：进行跨场景的比较结果可视化
plot_comparison_results(all_sim_results, scenario_names, params);

fprintf('仿真主程序完成!\n');

%% ==================== 系统参数初始化函数 ====================
function params = init_system_params()
    % 初始化系统参数结构体
    % 返回: params - 包含所有系统参数的结构体
    
    %% 基本系统参数
    params.N = 5;                        % 组播组中的用户数
    params.channel_states = 3;           % 信道状态数量: 好(3), 中(2), 差(1) - 假设索引越大越好
    params.buffer_levels = 10;           % 缓冲区状态离散化级别数
    params.video_layers = 3;             % 视频层数: 基础层(1), 基础层+增强层1(2), 基础层+增强层1+增强层2(3)
    params.mcs_levels = 3;               % 调制编码方案(MCS)的级别数
    
    %% 信道状态转移参数 (基于Gilbert-Elliott模型)
    % 信道状态转移矩阵 P(下一状态 | 当前状态)
    % 行: 差, 中, 好; 列: 差, 中, 好
    params.channel_transition = [
        0.7, 0.2, 0.1;  % 差 -> [差, 中, 好]
        0.3, 0.4, 0.3;  % 中 -> [差, 中, 好]
        0.1, 0.2, 0.7   % 好 -> [差, 中, 好]
    ];
    
    %% 缓冲区参数
    params.buffer_max = 1.0;             % 缓冲区最大占用率 (归一化)
    params.buffer_min = 0.2;             % 缓冲区最小占用率阈值 (低于此值可能卡顿)
    params.buffer_drain_rate = 0.1;      % 缓冲区消耗速率 (每个时间步消耗的单位)
    
    %% 视频码率参数 (单位: Mbps)
    % 不同视频层的码率
    params.video_rates = [1.5, 3.0, 6.0];  % 对应层1, 层2, 层3
    
    %% MCS参数
    % 不同MCS级别在不同信道状态下的传输容量 (单位: Mbps)
    % params.mcs_capacity(MCS级别, 信道状态值)
    % 信道状态值: 1=差, 2=中, 3=好
    params.mcs_capacity = [ % MCS行, 信道列 (差, 中, 好)
        0.5, 1.0, 2.0;    % MCS1 在 [差, 中, 好] 信道下的容量
        0.8, 2.0, 4.0;    % MCS2 在 [差, 中, 好] 信道下的容量
        1.2, 4.0, 8.0     % MCS3 在 [差, 中, 好] 信道下的容量
    ];
    % 注意: 原始代码中 mcs_capacity 的列顺序可能是 [好, 中, 差]。
    % 此处已调整为 [差, 中, 好] 以匹配信道状态值索引 (1=差等)。
    % 如果原始代码的P,R矩阵基于旧顺序，则需仔细检查。
    % 为保持一致性，假设信道状态1为差，2为中，3为好用于索引。

    % 不同MCS在不同信道状态下的误块率 (BLER)
    % params.mcs_bler(MCS级别, 信道状态值)
    params.mcs_bler = [ % MCS行, 信道列 (差, 中, 好)
        0.30, 0.05, 0.01;  % MCS1 在 [差, 中, 好] 信道下的BLER
        0.50, 0.15, 0.05;  % MCS2 在 [差, 中, 好] 信道下的BLER
        0.80, 0.30, 0.10   % MCS3 在 [差, 中, 好] 信道下的BLER
    ];
    % 与mcs_capacity的顺序调整说明相同。
    
    %% 功耗参数
    params.power_base = 1.0;             % 基础功耗 (单位)
    params.power_layer = 0.2;            % 每增加一个视频层的额外功耗
    params.power_mcs = 0.3;              % 每增加一级MCS的额外功耗
    
    %% 奖励函数权重系数
    params.alpha = 0.6;                  % 视频质量的权重
    params.beta = 0.3;                   % 缓冲区惩罚的权重 (用于惩罚偏离理想缓冲区或低缓冲区)
    params.gamma_power = 0.1;            % 功耗惩罚的权重
    
    %% 视频质量参数 (例如 PSNR 或等效质量评分)
    params.video_quality = [30, 35, 40]; % 对应层1, 层2, 层3的视频质量
    
    %% 算法参数
    params.gamma = 0.9;                  % 折扣因子 (discount factor for future rewards)
    params.epsilon_vi = 1e-6;            % 值迭代算法的收敛阈值
    params.max_iterations = 5000;        % 值迭代算法的最大迭代次数
    params.simulation_steps = 1000;      % 策略仿真评估的时间步数
end

%% ==================== 状态和动作空间初始化函数 ====================
function [S, A, state_mapping, action_mapping] = init_state_action_space(params)
    % 初始化状态空间S和动作空间A，以及它们到实际状态/动作结构的映射
    % 输入: params - 系统参数结构体
    % 输出:
    %   S - 状态索引向量 (1, 2, ..., num_states)
    %   A - 动作索引向量 (1, 2, ..., num_actions)
    %   state_mapping - cell数组，state_mapping{i}是第i个状态的结构体描述
    %   action_mapping - cell数组，action_mapping{j}是第j个动作的结构体描述

    % 状态空间初始化
    % 注意: 原始代码片段中状态空间的定义方式比较特殊。
    % 此处尝试遵循原始片段中的循环结构来定义状态，这可能是一个简化的状态表示。
    % 一个更标准的MDP状态可能是 (所有用户的信道状态向量, 所有用户的缓冲区水平向量)。
    % 如果 '当前传输的视频层' 也是状态的一部分，则状态空间会更复杂。
    % 此处的状态定义基于 `state.video_layer` 表示当前正在服务的视频层。
    
    state_indices_counter = 1; % 状态索引计数器
    temp_state_mapping_cell = {}; % 临时的cell数组用于存储状态映射

    % 根据原始代码中的循环结构创建状态映射
    % 这暗示了一种特定的，可能简化的状态表示方法。
    % 如果此状态表示不是MDP的预期表示，那么`compute_matrices`也将基于此潜在有问题的假设。
    for v_current_layer_in_state = 1:params.video_layers % 假设状态包含当前正在传输的视频层
        for user_focus_idx = 1:params.N % 这个循环对于全局状态而言不寻常，可能表示一种简化
            for ch_val = 1:params.channel_states % 焦点用户的信道状态值
                for buf_val_idx = 1:params.buffer_levels % 焦点用户的缓冲区级别索引
                    
                    % 构建当前迭代下的状态结构体
                    % 这是基于原始代码的简化状态表示
                    temp_channel_vec = ones(params.N, 1) * 2; % 其他用户默认为中等信道 (状态值2)
                    temp_channel_vec(user_focus_idx) = ch_val; % 设置焦点用户的信道状态

                    temp_buffer_vec = ones(params.N, 1) * (params.buffer_max / 2); % 其他用户默认为半满缓冲区
                    % 将缓冲区索引转换为归一化的缓冲区水平
                    temp_buffer_vec(user_focus_idx) = (buf_val_idx -1) / (params.buffer_levels -1) * params.buffer_max;
                    
                    current_s_struct = struct(...
                        'channel', temp_channel_vec, ... % N个用户的信道状态向量
                        'buffer', temp_buffer_vec, ...  % N个用户的缓冲区水平向量
                        'video_layer', v_current_layer_in_state); % 当前系统中正在传输的视频层 (作为状态的一部分)
                    
                    temp_state_mapping_cell{state_indices_counter} = current_s_struct;
                    state_indices_counter = state_indices_counter + 1;
                end
            end
        end
    end
    state_mapping = temp_state_mapping_cell'; % 转换为列cell数组
    S = 1:(state_indices_counter-1); % 状态索引向量
    
    % 动作空间初始化
    % 动作 a = (选择传输的视频层, 选择的MCS级别)
    action_indices_counter = 1; % 动作索引计数器
    temp_action_mapping_cell = {}; % 临时的cell数组用于存储动作映射
    for l_chosen = 1:params.video_layers % 选择传输的视频层
        for m_chosen = 1:params.mcs_levels % 选择的MCS级别
            temp_action_mapping_cell{action_indices_counter} = struct('layer', l_chosen, 'mcs', m_chosen);
            action_indices_counter = action_indices_counter + 1;
        end
    end
    action_mapping = temp_action_mapping_cell'; % 转换为列cell数组
    A = 1:(action_indices_counter-1); % 动作索引向量
    
    if isempty(S) || isempty(A)
        error('状态空间或动作空间为空。请检查初始化参数和循环逻辑。');
    end
end


%% ==================== 转移概率P和奖励矩阵R计算函数 ====================
function [P, R] = compute_matrices(S, A, params, state_mapping, action_mapping)
    % 计算转移概率矩阵P P(s'|s,a) 和奖励矩阵R R(s,a)
    % 输入: (如上定义)
    % 输出:
    %   P - 转移概率三维矩阵 P(当前状态索引, 当前动作索引, 下一状态索引)
    %   R - 奖励二维矩阵 R(当前状态索引, 当前动作索引)

    n_states = length(S);   % 状态总数
    n_actions = length(A); % 动作总数
    
    P = zeros(n_states, n_actions, n_states); % 初始化P矩阵
    R = zeros(n_states, n_actions);           % 初始化R矩阵
    
    fprintf('计算P和R矩阵的总状态-动作对数量: %d\n', n_states * n_actions);
    progress_interval = max(1, floor(n_states / 20)); % 大约显示20次进度

    for s_idx = 1:n_states % 遍历所有当前状态
        if mod(s_idx, progress_interval) == 0
            fprintf('计算P和R矩阵: 当前状态 %d / %d (%.1f%%)\n', s_idx, n_states, (s_idx/n_states)*100);
        end
        current_s_struct = state_mapping{s_idx}; % 获取当前状态的详细结构
        
        for a_idx = 1:n_actions % 遍历所有可能的动作
            current_a_struct = action_mapping{a_idx}; % 获取当前动作的详细结构
            
            % 计算即时奖励 R(s,a)
            R(s_idx, a_idx) = compute_reward(current_s_struct, current_a_struct, params);
            
            % 计算转移到所有下一状态 s' 的概率 P(s'|s,a)
            temp_probs_for_s_next = zeros(1, n_states); % 临时存储到各下一状态的概率

            for s_next_idx = 1:n_states % 遍历所有可能的下一状态
                next_s_struct = state_mapping{s_next_idx}; % 获取下一状态的详细结构
                temp_probs_for_s_next(s_next_idx) = compute_transition_prob(current_s_struct, current_a_struct, next_s_struct, params);
            end
            
            % 归一化转移概率，确保 sum_{s'} P(s'|s,a) = 1
            sum_probs = sum(temp_probs_for_s_next);
            if sum_probs > 1e-9 % 如果存在任何转移的可能性 (避免除以零)
                P(s_idx, a_idx, :) = temp_probs_for_s_next / sum_probs;
            else
                % 如果根据转移逻辑，从(s,a)无法转移到任何s' (即所有转移概率为0)
                % 这可能表示该动作在状态s下不可行，或者转移逻辑有缺陷。
                % 一种处理方式是分配均匀概率到所有状态（作为回退）。
                % 另一种可能是该(s,a)对的奖励应该非常低。
                % 当前采用均匀分布。这可能需要根据具体问题进行调整。
                % fprintf('警告: 对于 s_idx=%d, a_idx=%d, 到所有下一状态的概率和约等于0。采用均匀分布。\n', s_idx, a_idx);
                P(s_idx, a_idx, :) = 1 / n_states;
            end
        end
    end
    fprintf('P和R矩阵计算完成。\n');
end

function reward = compute_reward(state_struct, action_struct, params)
    % 计算在状态 state_struct 下执行动作 action_struct 所获得的即时奖励
    % 输入:
    %   state_struct - 当前状态的结构体
    %   action_struct - 当前采取的动作的结构体
    %   params - 系统参数
    % 输出:
    %   reward - 计算得到的即时奖励值

    chosen_layer = action_struct.layer; % 动作选择的视频层
    chosen_mcs = action_struct.mcs;     % 动作选择的MCS级别

    % 1. 视频质量相关的奖励部分 (基于选择传输的层)
    quality_component = params.video_quality(chosen_layer);

    % 2. 缓冲区相关的惩罚部分
    % 可以考虑平均缓冲区水平或最差用户的缓冲区水平。此处使用平均值。
    avg_buffer_level = mean(state_struct.buffer);
    buffer_penalty_val = 0;
    % 对缓冲区过低进行惩罚
    if avg_buffer_level < params.buffer_min
        buffer_penalty_val = params.beta * (params.buffer_min - avg_buffer_level) * 5; % 较强的惩罚
    end
    % 对缓冲区溢出进行惩罚 (虽然有消耗速率，但仍可能发生)
    if avg_buffer_level > params.buffer_max 
        buffer_penalty_val = buffer_penalty_val + params.beta * (avg_buffer_level - params.buffer_max) * 2;
    end
    
    % 3. 功耗相关的惩罚部分
    power_consumed = params.power_base + params.power_layer * chosen_layer + params.power_mcs * chosen_mcs;
    power_penalty_val = params.gamma_power * power_consumed;

    % 4. 传输可行性/成功率相关的惩罚 (关键部分)
    transmission_penalty = 0;
    required_rate_for_chosen_layer = params.video_rates(chosen_layer); % 选择层所需的码率
    
    % 检查所有用户的接收情况。如果任何一个用户无法接收，则存在问题。
    % 为简化，考虑最差用户的信道状态下，所选MCS的性能。
    worst_user_channel_state_val = min(state_struct.channel); % 1=差, 2=中, 3=好
    
    % 获取所选MCS在最差信道下的容量和BLER
    % 注意索引: params.mcs_capacity(MCS级别, 信道状态值)
    capacity_at_chosen_mcs_worst_channel = params.mcs_capacity(chosen_mcs, worst_user_channel_state_val);
    bler_at_chosen_mcs_worst_channel = params.mcs_bler(chosen_mcs, worst_user_channel_state_val);

    % 如果有效速率 (考虑BLER后) 低于所需速率
    if required_rate_for_chosen_layer > capacity_at_chosen_mcs_worst_channel * (1 - bler_at_chosen_mcs_worst_channel)
        transmission_penalty = 2; % 对选择不可持续的动作给予严重惩罚
    % 或者，如果名义容量 (不考虑BLER) 就低于所需速率
    elseif required_rate_for_chosen_layer > capacity_at_chosen_mcs_worst_channel
        transmission_penalty = 1; % 仍然是显著的惩罚
    end
    
    % 总奖励 = 质量奖励 - 缓冲区惩罚 - 功耗惩罚 - 传输惩罚
    reward = params.alpha * quality_component - buffer_penalty_val - power_penalty_val - transmission_penalty;
end

function prob = compute_transition_prob(current_s_struct, current_a_struct, next_s_struct, params)
    % 计算从 current_s_struct 状态，通过 current_a_struct 动作，转移到 next_s_struct 状态的概率
    % P(s'|s,a) = P(channel'|channel) * P(buffer'|buffer,channel,action) * P(video_layer'|video_layer_s,action)
    % 假设信道、缓冲区、视频层状态的转移是相互独立的。

    % 1. 信道状态转移概率 P(channel'|channel)
    prob_channel_transition = 1.0;
    for i = 1:params.N % 遍历每个用户
        % params.channel_transition(当前信道状态, 下一信道状态)
        prob_channel_transition = prob_channel_transition * params.channel_transition(current_s_struct.channel(i), next_s_struct.channel(i));
    end

    % 2. 缓冲区状态转移概率 P(buffer'|buffer,channel,action)
    prob_buffer_transition = 1.0;
    chosen_layer_for_tx = current_a_struct.layer; % 动作选择的传输层
    chosen_mcs_for_tx = current_a_struct.mcs;     % 动作选择的MCS
    data_rate_chosen_layer = params.video_rates(chosen_layer_for_tx); % 该层的数据率

    for i = 1:params.N % 遍历每个用户
        current_buffer_user = current_s_struct.buffer(i);       % 用户i当前缓冲区水平
        current_channel_user = current_s_struct.channel(i);     % 用户i当前信道状态 (在状态s中)
        
        % 用户i的有效数据接收速率
        % params.mcs_capacity(MCS级别, 信道状态值)
        capacity_user_mcs = params.mcs_capacity(chosen_mcs_for_tx, current_channel_user);
        bler_user_mcs = params.mcs_bler(chosen_mcs_for_tx, current_channel_user);
        
        data_arrival_rate_user = 0; % 用户i的数据到达速率 (单位/时间步)
        % 只有当MCS的名义容量能支持所选层的数据率时，才认为有数据到达
        if data_rate_chosen_layer <= capacity_user_mcs 
            data_arrival_rate_user = data_rate_chosen_layer * (1 - bler_user_mcs);
        end
        % 注意: 此处假设 video_rates 和 buffer_drain_rate 的单位是兼容的 (例如 "单位/时间步")。
        % 如果 video_rates 是 Mbps，需要一个转换因子。当前直接使用。
        
        % 预期的下一缓冲区水平 (连续值)
        expected_next_buffer_user = current_buffer_user + data_arrival_rate_user - params.buffer_drain_rate;
        expected_next_buffer_user = max(0, min(params.buffer_max, expected_next_buffer_user)); % 限制在[0, max]
        
        % 将预期的下一缓冲区水平（连续）映射到离散的缓冲区级别，以便与 next_s_struct.buffer(i) 比较。
        % next_s_struct.buffer(i) 是 params.buffer_levels 个离散级别之一。
        buffer_discretization_step = params.buffer_max / (params.buffer_levels -1); % 每个离散级别代表的缓冲区范围
        
        % 预期下一缓冲区对应的离散索引
        expected_discrete_idx = round(expected_next_buffer_user / buffer_discretization_step) + 1;
        expected_discrete_idx = max(1, min(params.buffer_levels, expected_discrete_idx)); % 确保索引在范围内

        % 实际下一状态中用户i的缓冲区对应的离散索引
        actual_next_discrete_idx = round(next_s_struct.buffer(i) / buffer_discretization_step) + 1;
        actual_next_discrete_idx = max(1, min(params.buffer_levels, actual_next_discrete_idx));

        % 简化的转移概率：如果实际下一离散缓冲区级别与预期接近，则概率较高。
        % 这是 MDP 中连续到离散状态转移的一种常见简化。
        % 更严谨的方法是使用围绕期望值的概率分布。
        if actual_next_discrete_idx == expected_discrete_idx
            prob_buffer_transition_user = 0.8; % 如果与预期匹配，概率高
        elseif abs(actual_next_discrete_idx - expected_discrete_idx) == 1
            prob_buffer_transition_user = 0.1; % 如果相差一个级别，概率较低
        else
            prob_buffer_transition_user = 0.0; % 如果相差较远，概率非常低或为0
        end
        prob_buffer_transition = prob_buffer_transition * prob_buffer_transition_user;
    end

    % 3. (状态中的)视频层转移概率 P(video_layer_s' | video_layer_s, action)
    % 这取决于状态结构中 `video_layer` 的确切含义。
    % 如果 state.video_layer 表示“系统在上一步决定传输的层”，
    % 那么 next_state.video_layer 应该精确地等于 current_a_struct.layer (当前动作选择的层)。
    prob_layer_transition = 0.0;
    if next_s_struct.video_layer == current_a_struct.layer
        prob_layer_transition = 1.0;
    end
    
    % 总转移概率
    prob = prob_channel_transition * prob_buffer_transition * prob_layer_transition;
    
    % 确保概率不是 NaN (例如由于参数索引错误导致组件为NaN)
    if isnan(prob)
        prob = 0;
    end
end

%% ==================== 值迭代求解最优策略函数 ====================
function [V, policy, deltas] = value_iteration_enhanced(S, A, P, R, gamma_discount, epsilon_conv, max_iter)
    % 使用值迭代算法求解最优值函数V和最优策略policy
    % 输入:
    %   S, A, P, R - 状态空间，动作空间，转移概率，奖励矩阵
    %   gamma_discount - 折扣因子
    %   epsilon_conv - 收敛阈值
    %   max_iter - 最大迭代次数
    % 输出:
    %   V - 最优值函数向量 (每个状态一个值)
    %   policy - 最优策略向量 (每个状态一个最优动作索引)
    %   deltas - 每次迭代的值函数最大变化量记录

    n_states = length(S);
    n_actions = length(A);
    
    V = zeros(n_states, 1);          % 初始化值函数 (通常为0)
    policy = ones(n_states, 1);      % 初始化策略 (默认为动作1)
    deltas = zeros(max_iter, 1);     % 记录每次迭代的delta值
    
    fprintf('开始值迭代算法...\n');
    for iteration = 1:max_iter % 迭代循环
        delta_max_this_iter = 0; % 本次迭代中V(s)的最大变化量
        V_old = V; % 保存上一轮的值函数，用于计算delta

        for s_idx = 1:n_states % 遍历所有状态 s
            % 计算当前状态 s_idx下，所有动作 a 的Q值: Q(s,a)
            Q_s_a = zeros(1, n_actions);
            for a_idx = 1:n_actions % 遍历所有动作 a
                expected_future_reward = 0;
                % 计算 sum_{s'} P(s'|s,a) * V_old(s')
                % 这个内层循环可以被向量化，如果P的形状合适，或者使用pagefun (如果可用)
                for s_next_idx = 1:n_states % 遍历所有可能的下一状态 s'
                    expected_future_reward = expected_future_reward + P(s_idx, a_idx, s_next_idx) * V_old(s_next_idx);
                end
                Q_s_a(a_idx) = R(s_idx, a_idx) + gamma_discount * expected_future_reward;
            end
            
            % 找到最大的Q值及其对应的动作，更新V(s)和policy(s)
            [max_q_val, best_action_idx] = max(Q_s_a);
            V(s_idx) = max_q_val;
            policy(s_idx) = best_action_idx;
            
            % 更新本次迭代的最大变化量
            delta_max_this_iter = max(delta_max_this_iter, abs(V(s_idx) - V_old(s_idx)));
        end
        
        deltas(iteration) = delta_max_this_iter; % 记录本次迭代的delta
        
        % 打印迭代进度
        if mod(iteration, 50) == 0 || iteration == 1
            fprintf('迭代次数 %d, 最大值变化 (Delta): %.8f\n', iteration, delta_max_this_iter);
        end
        
        % 检查是否收敛
        if delta_max_this_iter < epsilon_conv
            fprintf('值迭代算法在 %d 次迭代后收敛。最终Delta: %.8f\n', iteration, delta_max_this_iter);
            deltas = deltas(1:iteration); % 截断未使用的部分
            break; % 跳出迭代循环
        end
        if iteration == max_iter % 达到最大迭代次数
             fprintf('值迭代算法达到最大迭代次数 (%d) 未完全收敛。最终Delta: %.8f\n', max_iter, delta_max_this_iter);
        end
    end
    
    % 可视化收敛过程
    figure('Name', '值迭代算法收敛曲线', 'Position', [300, 300, 900, 400]);
    subplot(1,2,1); % 普通尺度
    plot(deltas, 'b-', 'LineWidth', 1.5);
    title('值迭代收敛曲线 (普通尺度)');
    xlabel('迭代次数');
    ylabel('最大值变化 (Delta)');
    grid on;
    
    subplot(1,2,2); % 对数尺度
    semilogy(deltas, 'r-', 'LineWidth', 1.5);
    title('值迭代收敛曲线 (对数尺度)');
    xlabel('迭代次数');
    ylabel('最大值变化 (Delta) - Log尺度');
    grid on;
    saveas(gcf, 'value_iteration_convergence.png'); % 保存收敛曲线图
end

%% ==================== 灵活的策略仿真评估函数 ====================
function results = simulate_policy_flexible(policy_vector_mdp, S, A, P, R, params, state_mapping, action_mapping, policy_type_str)
    % 模拟不同类型策略的性能
    % 输入:
    %   policy_vector_mdp - MDP最优策略向量 (仅当 policy_type_str == 'mdp' 时使用)
    %   S, A, P, R, params, state_mapping, action_mapping - 同原有函数
    %   policy_type_str - 字符串，指定策略类型: 'mdp', 'static_hq', 'channel_reactive'
    % 输出:
    %   results - 包含仿真结果的结构体

    T = params.simulation_steps; % 仿真总步数
    results = struct(); % 初始化结果结构体
    results.rewards = zeros(1, T);         % 每步的即时奖励
    results.video_quality = zeros(1, T);   % 每步选择的视频质量
    results.buffer_levels = zeros(params.N, T); % 每步各用户的缓冲区水平
    results.selected_layers = zeros(1, T); % 每步选择的视频层
    results.selected_mcs = zeros(1, T);    % 每步选择的MCS级别
    results.channel_states = zeros(params.N, T);% 每步各用户的信道状态
    results.outage_ratio = 0;              % 缓冲区欠载（卡顿）比率
    results.avg_buffer = zeros(1, T);      % 每步所有用户的平均缓冲区水平
    
    s_idx = randi(length(S)); % 随机选择一个初始状态索引
    buffer_outage_count = 0;  % 记录缓冲区低于阈值的次数（所有用户总计）
    
    fprintf('开始仿真策略: %s, 总步数: %d\n', policy_type_str, T);
    progress_print_interval = max(1, floor(T / 10)); % 大约打印10次进度

    for t = 1:T % 时间步循环
        if mod(t, progress_print_interval) == 0
            fprintf('仿真 %s: 第 %d 步 / 共 %d 步 (%.0f%%)\n', policy_type_str, t, T, (t/T)*100);
        end
        
        current_s_struct = state_mapping{s_idx}; % 获取当前状态的详细结构
        a_idx = -1; % 将要确定的动作索引
        action_struct_to_execute = struct(); % 将要执行的动作的详细结构

        % --- 根据策略类型确定当前时间步的动作 ---
        if strcmp(policy_type_str, 'mdp') % 如果是MDP最优策略
            a_idx = policy_vector_mdp(s_idx); % 从策略向量中获取动作索引
            action_struct_to_execute = action_mapping{a_idx}; % 获取动作结构
        elseif strcmp(policy_type_str, 'static_hq') % 如果是静态高质量策略
            target_layer = params.video_layers; % 尝试传输最高质量的视频层
            worst_user_channel = min(current_s_struct.channel); % 获取最差用户的信道状态 (1=差, 3=好)
            
            chosen_mcs_static = -1; % 初始化选择的MCS
            % 从高到低尝试MCS级别，找到第一个能支持目标层的MCS
            for mcs_try = params.mcs_levels:-1:1 
                % params.mcs_capacity(MCS级别, 信道状态值)
                effective_capacity = params.mcs_capacity(mcs_try, worst_user_channel) * (1-params.mcs_bler(mcs_try, worst_user_channel));
                if params.video_rates(target_layer) <= effective_capacity
                    chosen_mcs_static = mcs_try;
                    break; % 找到了合适的MCS
                end
            end
            if chosen_mcs_static == -1 % 如果最高层无法支持，则回退
                action_struct_to_execute.layer = 1; % 选择基础层
                action_struct_to_execute.mcs = 1;   % 选择最鲁棒的MCS
            else
                action_struct_to_execute.layer = target_layer;
                action_struct_to_execute.mcs = chosen_mcs_static;
            end
        elseif strcmp(policy_type_str, 'channel_reactive') % 如果是信道反应式策略
            worst_user_channel = min(current_s_struct.channel); % 获取最差用户的信道状态
            if worst_user_channel == 3 % 信道好
                action_struct_to_execute.layer = 3; action_struct_to_execute.mcs = 3; % 选择最高层和最高MCS
            elseif worst_user_channel == 2 % 信道中
                action_struct_to_execute.layer = 2; action_struct_to_execute.mcs = 2; % 选择中间层和中间MCS
            else % 信道差 (worst_user_channel == 1)
                action_struct_to_execute.layer = 1; action_struct_to_execute.mcs = 1; % 选择最低层和最低MCS
            end
        end

        % 如果动作是由策略逻辑 (非MDP) 确定的结构体，需要找到对应的动作索引 a_idx
        % 以便使用P矩阵和R矩阵 (它们是基于索引的)
        if ~strcmp(policy_type_str, 'mdp')
            found_match_a_idx = -1;
            for temp_idx = 1:length(A) % A 是动作索引向量 1:num_actions
                if action_mapping{temp_idx}.layer == action_struct_to_execute.layer && ...
                   action_mapping{temp_idx}.mcs == action_struct_to_execute.mcs
                    found_match_a_idx = temp_idx;
                    break;
                end
            end
            if found_match_a_idx ~= -1
                a_idx = found_match_a_idx;
            else
                % 如果选择的动作组合不在预定义的 action_mapping 中 (理论上不应发生，除非策略逻辑有误)
                warning('策略 %s 选择的动作在 action_mapping 中未找到。默认使用动作1。', policy_type_str);
                a_idx = 1; % 回退到默认动作
                action_struct_to_execute = action_mapping{a_idx}; % 同时更新动作结构体
            end
        end

        % --- 记录当前时间步 t 的各项结果 ---
        results.rewards(t) = R(s_idx, a_idx); % 即时奖励
        results.video_quality(t) = params.video_quality(action_struct_to_execute.layer); % 视频质量
        results.buffer_levels(:, t) = current_s_struct.buffer; % 各用户缓冲区水平
        results.avg_buffer(t) = mean(current_s_struct.buffer); % 平均缓冲区水平
        results.selected_layers(t) = action_struct_to_execute.layer; % 选择的视频层
        results.selected_mcs(t) = action_struct_to_execute.mcs;       % 选择的MCS
        results.channel_states(:, t) = current_s_struct.channel;     % 各用户信道状态

        % 检查是否有用户的缓冲区低于最小阈值
        if any(current_s_struct.buffer < params.buffer_min) 
            buffer_outage_count = buffer_outage_count + 1; % 增加欠载计数
        end
        
        % --- 根据当前状态 s_idx 和选定动作 a_idx，采样得到下一状态 next_s_idx ---
        try
            next_s_idx = sample_next_state(s_idx, a_idx, P);
        catch ME % 捕获 sample_next_state 中的错误，用于调试
            fprintf('在 sample_next_state 中发生错误: s_idx=%d, a_idx=%d, 策略类型=%s\n', s_idx, a_idx, policy_type_str);
            disp('当前 s_idx 对应的 state_mapping:'); disp(state_mapping{s_idx});
            disp('当前 a_idx 对应的 action_mapping:'); disp(action_mapping{a_idx});
            % 打印P矩阵的相关切片信息
            current_probs_debug = squeeze(P(s_idx, a_idx, :));
            fprintf('P(s_idx, a_idx, :) 的概率和 = %f\n', sum(current_probs_debug));
            if any(isnan(current_probs_debug)); fprintf('在P矩阵切片中发现NaN值\n'); end
            if any(current_probs_debug < 0); fprintf('在P矩阵切片中发现负值\n'); end
            rethrow(ME); % 重新抛出原始错误，以便查看详细信息
        end
        s_idx = next_s_idx; % 更新当前状态索引，进入下一时间步
    end

    % 计算仿真结束后的平均性能指标
    results.avg_reward = mean(results.rewards);
    results.avg_quality = mean(results.video_quality);
    results.outage_ratio = buffer_outage_count / T; % 至少一个用户缓冲区欠载的时间步比例
    results.avg_layer = mean(results.selected_layers);
    results.avg_mcs = mean(results.selected_mcs);
    
    fprintf('策略 %s 的仿真评估完成。平均奖励: %.2f, 平均质量: %.2f, 欠载率: %.2f%%\n', ...
        policy_type_str, results.avg_reward, results.avg_quality, results.outage_ratio*100);
end

function next_state_idx = sample_next_state(current_s_idx, current_a_idx, P_matrix)
    % 根据转移概率 P(s'|s,a) 采样得到下一个状态的索引
    % 输入:
    %   current_s_idx - 当前状态的索引
    %   current_a_idx - 当前动作的索引
    %   P_matrix - 转移概率矩阵 P(s,a,s')
    % 输出:
    %   next_state_idx - 采样得到的下一状态的索引

    % 从P矩阵中提取当前 (s,a) 对到所有下一状态 s' 的转移概率向量
    probabilities_s_prime = squeeze(P_matrix(current_s_idx, current_a_idx, :));
    
    % 检查概率和是否为1 (考虑到浮点误差)
    if abs(sum(probabilities_s_prime) - 1.0) > 1e-5 
        % fprintf('警告: P(%d,%d,:) 的概率和不为1 (和=%.4f)。正在归一化。\n', current_s_idx, current_a_idx, sum(probabilities_s_prime));
        if sum(probabilities_s_prime) == 0 % 如果所有概率都为0 (不应发生如果P矩阵正确构建)
             probabilities_s_prime = ones(size(probabilities_s_prime)) / length(probabilities_s_prime); % 采用均匀分布作为回退
        else
             probabilities_s_prime = probabilities_s_prime / sum(probabilities_s_prime); % 归一化
        end
    end
    
    % 计算累积分布函数 (CDF)
    cumulative_probs = cumsum(probabilities_s_prime);
    
    % 生成一个 (0,1] 之间的随机数
    random_sample = rand();
    
    % 找到第一个累积概率大于等于随机数的索引，即为采样到的下一状态
    next_state_idx = find(random_sample <= cumulative_probs, 1, 'first');
    
    if isempty(next_state_idx) % 如果由于某种原因 (如概率全为0或NaN) 未找到下一状态
        warning('sample_next_state: 无法找到下一状态。默认选择最后一个状态。请检查P矩阵。');
        next_state_idx = length(probabilities_s_prime); % 回退到最后一个可能的状态
    end
end

%% ==================== 灵活的结果可视化函数 ====================
function plot_results_flexible(results, params, scenario_title_str)
    % 可视化仿真结果，并在图表标题中加入场景名称
    % 输入:
    %   results - 包含仿真结果的结构体
    %   params - 系统参数结构体
    %   scenario_title_str - 场景名称字符串，用于图表标题

    T = params.simulation_steps;    % 仿真总步数
    time_steps = 1:T;               % 时间轴
    
    % 创建一个对文件名安全的场景标题 (移除空格和特殊字符)
    safe_scenario_title = strrep(scenario_title_str, ' ', '_');
    safe_scenario_title = regexprep(safe_scenario_title, '[^a-zA-Z0-9_]', '');

    % 创建新的图形窗口
    figure_handle = figure('Name', ['性能评估: ' scenario_title_str], 'Position', [50, 50, 1300, 850], 'NumberTitle', 'off');
    
    % 子图1: 平均累积奖励
    subplot(3,2,1);
    plot(time_steps, cumsum(results.rewards) ./ time_steps, 'b-', 'LineWidth', 1.5);
    title('平均累积奖励'); xlabel('时间步'); ylabel('平均奖励'); grid on;

    % 子图2: 视频质量和选择的视频层
    subplot(3,2,2);
    yyaxis left; % 左Y轴
    plot(time_steps, results.video_quality, 'b-', 'LineWidth', 1.5); 
    ylabel('视频质量 (评分)');
    ylim_left_q = [min(params.video_quality)*0.8, max(params.video_quality)*1.2]; % 动态Y轴范围
    if ylim_left_q(1) < ylim_left_q(2); ylim(ylim_left_q); end % 避免Y轴范围错误
    
    yyaxis right; % 右Y轴
    plot(time_steps, results.selected_layers, 'r--', 'LineWidth', 1.5); 
    ylabel('选择的视频层');
    ylim_right_l = [0.5, params.video_layers + 0.5];
    ylim(ylim_right_l); yticks(1:params.video_layers); % 设置Y轴刻度为整数层数
    
    title('视频质量与选择的层'); xlabel('时间步'); grid on; 
    legend('质量', '层数', 'Location','best');

    % 子图3: 平均缓冲区水平 (所有用户的平均值)
    subplot(3,2,3);
    plot(time_steps, results.avg_buffer, 'g-', 'LineWidth', 1.5); hold on;
    plot(time_steps, ones(1,T)*params.buffer_min, 'r--', 'LineWidth', 1); % 绘制最小缓冲区阈值线
    hold off;
    title('平均缓冲区水平 (用户平均)'); xlabel('时间步'); ylabel('平均缓冲区占用率');
    ylim([0 params.buffer_max*1.1]); grid on; % Y轴范围略大于最大值
    legend('平均缓冲', '最小阈值','Location','best');

    % 子图4: 选择的MCS级别
    subplot(3,2,4);
    plot(time_steps, results.selected_mcs, 'm-', 'LineWidth', 1.5);
    title('选择的MCS级别'); xlabel('时间步'); ylabel('MCS级别');
    ylim([0.5 params.mcs_levels + 0.5]); yticks(1:params.mcs_levels); grid on;

    % 子图5: 用户信道状态热图
    subplot(3,2,5);
    if params.N > 0 % 仅当用户数大于0时绘制
        imagesc(time_steps, 1:params.N, results.channel_states); % (X轴, Y轴, 数据)
        ylabel('用户索引');
        % 定义颜色条的刻度和标签 (1=差, 2=中, 3=好)
        channel_tick_labels = arrayfun(@(x) sprintf('%d',x), 1:params.channel_states, 'UniformOutput', false);
        colorbar('Ticks',1:params.channel_states,'TickLabels', channel_tick_labels);
    else
        text(0.5,0.5,'N=0, 无信道状态可绘制','HorizontalAlignment','center'); % 如果N=0
    end
    title('用户信道状态 (数值: 1=差, 2=中, 3=好)'); xlabel('时间步');
    set(gca, 'YDir', 'normal'); % 确保Y轴（用户索引）从下到上

    % 子图6: 关键性能指标汇总条形图
    subplot(3,2,6);
    metric_names = {'平均奖励', '平均质量', '平均层数', '平均MCS', '欠载率 (%)'};
    metric_values = [results.avg_reward, results.avg_quality, results.avg_layer, results.avg_mcs, results.outage_ratio*100];
    bar_h = bar(metric_values); % 绘制条形图
    title('性能指标汇总'); xticklabels(metric_names); xtickangle(30); grid on; % X轴标签和角度
    ylabel('指标值');
    % 在条形图上显示具体数值
    for k=1:length(metric_values)
        text(k, metric_values(k), sprintf('%.2f', metric_values(k)), ... % 数值标签
            'HorizontalAlignment','center', 'VerticalAlignment','bottom', 'FontSize',8, 'Color','k');
    end
    
    % 图形窗口总标题
    sgtitle(['性能评估: ' scenario_title_str], 'FontSize', 16, 'FontWeight', 'bold');
    % 保存图表
    saveas(figure_handle, ['sim_results_' safe_scenario_title '.png']);
    fprintf('场景 [%s] 的性能图表已保存为 sim_results_%s.png\n', scenario_title_str, safe_scenario_title);
end

%% ==================== 跨场景对比结果可视化函数 ====================
function plot_comparison_results(all_sim_results_cell, scenario_names_cell, params)
    % 可视化不同场景的关键性能指标对比
    % 输入:
    %   all_sim_results_cell - 包含每个场景仿真结果结构体的cell数组
    %   scenario_names_cell - 包含每个场景名称的cell数组
    %   params - 系统参数结构体 (可能用于设置Y轴范围等)

    num_scenarios = length(all_sim_results_cell); % 场景总数
    if num_scenarios < 1; return; end % 如果没有场景结果，则不绘图

    % 初始化用于存储各场景指标的向量
    avg_rewards = zeros(1, num_scenarios);
    avg_qualities = zeros(1, num_scenarios);
    outage_ratios_pct = zeros(1, num_scenarios);
    avg_sel_layers = zeros(1, num_scenarios);
    avg_sel_mcs = zeros(1, num_scenarios);

    % 从 all_sim_results_cell 中提取各场景的关键指标
    for i = 1:num_scenarios
        res = all_sim_results_cell{i}; % 当前场景的结果结构体
        avg_rewards(i) = res.avg_reward;
        avg_qualities(i) = res.avg_quality;
        outage_ratios_pct(i) = res.outage_ratio * 100; % 转换为百分比
        avg_sel_layers(i) = res.avg_layer;
        avg_sel_mcs(i) = res.avg_mcs;
    end

    % 创建新的图形窗口用于对比
    figure_comp = figure('Name', '跨场景性能对比', 'Position', [100,100,1100,750], 'NumberTitle','off');
    
    % 子图1: 平均奖励对比
    subplot(2,3,1); bar(avg_rewards); title('平均奖励对比'); 
    xticks(1:num_scenarios); xticklabels(scenario_names_cell); xtickangle(20); grid on;
    
    % 子图2: 平均视频质量对比
    subplot(2,3,2); bar(avg_qualities); title('平均视频质量对比'); 
    xticks(1:num_scenarios); xticklabels(scenario_names_cell); xtickangle(20); grid on;
    
    % 子图3: 缓冲区欠载率对比
    subplot(2,3,3); bar(outage_ratios_pct); title('缓冲区欠载率 (%) 对比'); 
    xticks(1:num_scenarios); xticklabels(scenario_names_cell); xtickangle(20); grid on;
    
    % 子图4: 平均选择层数对比
    subplot(2,3,4); bar(avg_sel_layers); title('平均选择层数对比'); 
    xticks(1:num_scenarios); xticklabels(scenario_names_cell); xtickangle(20); 
    ylim([0 params.video_layers+1]); grid on; % 设置Y轴范围
    
    % 子图5: 平均选择MCS级别对比
    subplot(2,3,5); bar(avg_sel_mcs); title('平均选择MCS级别对比'); 
    xticks(1:num_scenarios); xticklabels(scenario_names_cell); xtickangle(20); 
    ylim([0 params.mcs_levels+1]); grid on; % 设置Y轴范围
    
    % 子图6: (可选) 放置一个总结文本或留空
    subplot(2,3,6);
    text(0.5,0.5, '跨场景性能对比总结', 'HorizontalAlignment','center','FontSize',14); axis off;

    % 对比图的总标题
    sgtitle('所有场景性能对比总结', 'FontSize',16,'FontWeight','bold');
    % 保存对比图
    saveas(figure_comp, 'comparison_all_scenarios_summary.png');
    fprintf('跨场景对比图已保存为 comparison_all_scenarios_summary.png\n');
end