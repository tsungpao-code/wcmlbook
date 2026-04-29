%% q7a_generate_cost2100_datasets.m
% Q7(a): Generate more than five different channel datasets using COST2100.
%
% Purpose:
% This script uses the COST2100 MATLAB channel model to generate six
% different channel datasets by changing user distributions.
%
% Difference from original demo_model.m:
% 1. The original demo usually runs only one fixed channel scenario.
% 2. This script creates a dataset configuration list.
% 3. It automatically generates six datasets with different user distributions:
%    uniform, center, edge, hotspot, ring, and line.
% 4. It saves complex CSI, normalized CSI, real/imag CSI representation,
%    user positions, user velocities, and metadata for later CsiNet testing.
%
% Put this file inside:
%   cost2100-master/cost2100-master/matlab/
%
% Run in MATLAB:
%   run q7a_generate_cost2100_datasets.m

clear;
clc;
close all;

%% ================================================================
% 1. Add COST2100 MATLAB path
% ================================================================
% If this file is placed inside the COST2100 matlab folder, pwd is enough.
addpath(genpath(pwd));

fprintf('Current folder:\n%s\n', pwd);
fprintf('COST2100 MATLAB path added.\n');

%% ================================================================
% 2. Output folder
% ================================================================
output_folder = fullfile(pwd, 'q7_generated_datasets');

if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

fprintf('Output folder:\n%s\n', output_folder);

%% ================================================================
% 3. Global simulation settings
% ================================================================
% You can increase num_users after confirming the script works.
% For homework testing, 200 users per dataset is enough and faster.

num_users = 200;        % Number of user samples per dataset
snapRate  = 1;          % Snapshot rate
snapNum   = 1;          % One snapshot per user
delta_f   = 7.8125e4;   % Frequency spacing for impulse response conversion

rng(2026);              % Fixed random seed for reproducibility

%% ================================================================
% 4. Define six dataset configurations
% ================================================================
% To avoid unstable SemiUrban generation, this stable version uses
% IndoorHall_5GHz for all six datasets and changes the user distribution.
%
% This still satisfies Q7(a), because the question asks for more than five
% different channel datasets, such as by changing the distribution of users.

dataset_cfgs = {};

% Dataset 1: Indoor + uniform users
dataset_cfgs{end+1} = struct( ...
    'name', 'D1_indoor_uniform', ...
    'network', 'IndoorHall_5GHz', ...
    'scenario', 'LOS', ...
    'Band', 'Wideband', ...
    'freq', [-10e6 10e6] + 5.3e9, ...
    'BSPos', [10 10 0], ...
    'distribution', 'uniform', ...
    'radius', 8, ...
    'speed', 0.001);

% Dataset 2: Indoor + center users
dataset_cfgs{end+1} = struct( ...
    'name', 'D2_indoor_center', ...
    'network', 'IndoorHall_5GHz', ...
    'scenario', 'LOS', ...
    'Band', 'Wideband', ...
    'freq', [-10e6 10e6] + 5.3e9, ...
    'BSPos', [10 10 0], ...
    'distribution', 'center', ...
    'radius', 8, ...
    'speed', 0.001);

% Dataset 3: Indoor + edge users
dataset_cfgs{end+1} = struct( ...
    'name', 'D3_indoor_edge', ...
    'network', 'IndoorHall_5GHz', ...
    'scenario', 'LOS', ...
    'Band', 'Wideband', ...
    'freq', [-10e6 10e6] + 5.3e9, ...
    'BSPos', [10 10 0], ...
    'distribution', 'edge', ...
    'radius', 8, ...
    'speed', 0.001);

% Dataset 4: Indoor + hotspot users
dataset_cfgs{end+1} = struct( ...
    'name', 'D4_indoor_hotspot', ...
    'network', 'IndoorHall_5GHz', ...
    'scenario', 'LOS', ...
    'Band', 'Wideband', ...
    'freq', [-10e6 10e6] + 5.3e9, ...
    'BSPos', [10 10 0], ...
    'distribution', 'hotspot', ...
    'radius', 8, ...
    'speed', 0.001);

% Dataset 5: Indoor + ring users
dataset_cfgs{end+1} = struct( ...
    'name', 'D5_indoor_ring', ...
    'network', 'IndoorHall_5GHz', ...
    'scenario', 'LOS', ...
    'Band', 'Wideband', ...
    'freq', [-10e6 10e6] + 5.3e9, ...
    'BSPos', [10 10 0], ...
    'distribution', 'ring', ...
    'radius', 8, ...
    'speed', 0.001);

% Dataset 6: Indoor + line users
dataset_cfgs{end+1} = struct( ...
    'name', 'D6_indoor_line', ...
    'network', 'IndoorHall_5GHz', ...
    'scenario', 'LOS', ...
    'Band', 'Wideband', ...
    'freq', [-10e6 10e6] + 5.3e9, ...
    'BSPos', [10 10 0], ...
    'distribution', 'line', ...
    'radius', 8, ...
    'speed', 0.001);

%% ================================================================
% 5. Generate datasets
% ================================================================

for d = 1:length(dataset_cfgs)

    cfg = dataset_cfgs{d};

    fprintf('\n====================================================\n');
    fprintf('Generating dataset: %s\n', cfg.name);
    fprintf('Network: %s\n', cfg.network);
    fprintf('Scenario: %s\n', cfg.scenario);
    fprintf('Distribution: %s\n', cfg.distribution);
    fprintf('Frequency: [%g, %g]\n', cfg.freq(1), cfg.freq(2));
    fprintf('====================================================\n');

    % Generate user positions and velocities
    [MSPos_all, MSVelo_all] = generate_user_positions(cfg, num_users);

    H_list = cell(num_users, 1);
    valid_count = 0;

    for u = 1:num_users

        MSPos = MSPos_all(u, :);
        MSVelo = MSVelo_all(u, :);

        try
            %% ------------------------------------------------------------
            % Correct COST2100 function call
            %
            % IMPORTANT:
            % The correct official input order is:
            %
            % cost2100(network, scenario, freq, snapRate, snapNum, ...
            %          BSPosCenter, BSPosSpacing, BSPosNum, MSPos, MSVelo)
            %
            % Do NOT put cfg.Link or cfg.Band into cost2100().
            % cfg.Band is used later in create_IR_omni().
            %% ------------------------------------------------------------

            BSPosSpacing = [0 0 0];
            BSPosNum = 1;

            [paraEx, paraSt, link, env] = cost2100( ...
                cfg.network, ...
                cfg.scenario, ...
                cfg.freq, ...
                snapRate, ...
                snapNum, ...
                cfg.BSPos, ...
                BSPosSpacing, ...
                BSPosNum, ...
                MSPos, ...
                MSVelo);

            %#ok<NASGU>
            % paraEx, paraSt, and env are not saved for training,
            % but they are useful for debugging.

            %% ------------------------------------------------------------
            % Generate omnidirectional impulse response
            %% ------------------------------------------------------------
            h = create_IR_omni(link, cfg.freq, delta_f, cfg.Band);

            %% ------------------------------------------------------------
            % Convert impulse response to frequency response
            %% ------------------------------------------------------------
            H_freq = fft(h, [], 2);
            H_freq = squeeze(H_freq);

            % Convert to row vector
            H_freq = H_freq(:).';

            valid_count = valid_count + 1;
            H_list{valid_count} = H_freq;

        catch ME
            fprintf('User %d failed: %s\n', u, ME.message);

            if ~isempty(ME.stack)
                fprintf('Error file: %s\n', ME.stack(1).file);
                fprintf('Error line: %d\n', ME.stack(1).line);
                fprintf('Error function: %s\n', ME.stack(1).name);
            end
        end

        if mod(u, 50) == 0
            fprintf('Progress: %d / %d users processed.\n', u, num_users);
        end
    end

    %% ------------------------------------------------------------
    % Keep only valid samples
    %% ------------------------------------------------------------
    H_list = H_list(1:valid_count);

    if valid_count == 0
        warning('No valid samples generated for %s. Skip this dataset.', cfg.name);
        continue;
    end

    %% ------------------------------------------------------------
    % Align all channel vectors to the same length
    %% ------------------------------------------------------------
    min_len = inf;

    for i = 1:valid_count
        min_len = min(min_len, length(H_list{i}));
    end

    H_complex = zeros(valid_count, min_len);

    for i = 1:valid_count
        temp = H_list{i};
        H_complex(i, :) = temp(1:min_len);
    end

    %% ------------------------------------------------------------
    % Normalize CSI
    %% ------------------------------------------------------------
    norm_factor = max(abs(H_complex(:))) + eps;
    H_norm = H_complex / norm_factor;

    %% ------------------------------------------------------------
    % Split real and imaginary parts for deep learning input
    % Shape: [num_samples, csi_dimension, 2]
    %% ------------------------------------------------------------
    H_real = real(H_norm);
    H_imag = imag(H_norm);
    H_real_imag = cat(3, H_real, H_imag);

    %% ------------------------------------------------------------
    % Save metadata
    %% ------------------------------------------------------------
    metadata = struct();
    metadata.name = cfg.name;
    metadata.network = cfg.network;
    metadata.scenario = cfg.scenario;
    metadata.Band = cfg.Band;
    metadata.distribution = cfg.distribution;
    metadata.num_users = num_users;
    metadata.valid_count = valid_count;
    metadata.freq = cfg.freq;
    metadata.BSPos = cfg.BSPos;
    metadata.radius = cfg.radius;
    metadata.speed = cfg.speed;
    metadata.norm_factor = norm_factor;
    metadata.description = 'Q7(a) COST2100 dataset generated by changing user distribution.';

    %% ------------------------------------------------------------
    % Save .mat file
    %% ------------------------------------------------------------
    save_path = fullfile(output_folder, [cfg.name '.mat']);

    save(save_path, ...
        'H_complex', ...
        'H_norm', ...
        'H_real_imag', ...
        'MSPos_all', ...
        'MSVelo_all', ...
        'metadata', ...
        '-v7.3');

    fprintf('\nSaved dataset: %s\n', save_path);
    fprintf('Valid samples: %d / %d\n', valid_count, num_users);
    fprintf('H_complex size: ');
    disp(size(H_complex));
    fprintf('H_real_imag size: ');
    disp(size(H_real_imag));
end

fprintf('\n====================================================\n');
fprintf('Q7(a) COST2100 dataset generation completed.\n');
fprintf('Generated datasets are saved in:\n%s\n', output_folder);
fprintf('====================================================\n');

%% ================================================================
% Helper function: generate user position distributions
% ================================================================
function [MSPos_all, MSVelo_all] = generate_user_positions(cfg, num_users)

    BS = cfg.BSPos;
    R = cfg.radius;

    MSPos_all = zeros(num_users, 3);
    MSVelo_all = zeros(num_users, 3);

    switch cfg.distribution

        case 'uniform'
            % Uniform users in a circular area
            theta = 2*pi*rand(num_users, 1);
            radius = R * sqrt(rand(num_users, 1));

            MSPos_all(:, 1) = BS(1) + radius .* cos(theta);
            MSPos_all(:, 2) = BS(2) + radius .* sin(theta);
            MSPos_all(:, 3) = 0;

        case 'center'
            % Users concentrated near the base station
            theta = 2*pi*rand(num_users, 1);
            radius = abs(randn(num_users, 1)) * R / 4;
            radius = min(radius, R);

            MSPos_all(:, 1) = BS(1) + radius .* cos(theta);
            MSPos_all(:, 2) = BS(2) + radius .* sin(theta);
            MSPos_all(:, 3) = 0;

        case 'edge'
            % Users near the edge of the indoor area
            theta = 2*pi*rand(num_users, 1);
            radius = R * (0.75 + 0.25 * rand(num_users, 1));

            MSPos_all(:, 1) = BS(1) + radius .* cos(theta);
            MSPos_all(:, 2) = BS(2) + radius .* sin(theta);
            MSPos_all(:, 3) = 0;

        case 'hotspot'
            % Users clustered around three hotspots
            hotspot_centers = [
                BS(1) + 0.30*R, BS(2) + 0.25*R, 0;
                BS(1) - 0.25*R, BS(2) + 0.20*R, 0;
                BS(1) + 0.10*R, BS(2) - 0.30*R, 0
            ];

            cluster_id = randi(size(hotspot_centers, 1), num_users, 1);

            for i = 1:num_users
                center = hotspot_centers(cluster_id(i), :);
                MSPos_all(i, :) = center + [randn()*0.05*R, randn()*0.05*R, 0];
            end

        case 'ring'
            % Users distributed on a ring
            theta = 2*pi*rand(num_users, 1);
            radius = R * (0.45 + 0.10 * randn(num_users, 1));
            radius = max(0.2*R, min(radius, 0.8*R));

            MSPos_all(:, 1) = BS(1) + radius .* cos(theta);
            MSPos_all(:, 2) = BS(2) + radius .* sin(theta);
            MSPos_all(:, 3) = 0;

        case 'line'
            % Users distributed along a line, like a walking route
            x_line = linspace(-0.6*R, 0.6*R, num_users).';
            y_line = 0.05*R * randn(num_users, 1);

            MSPos_all(:, 1) = BS(1) + x_line;
            MSPos_all(:, 2) = BS(2) + y_line;
            MSPos_all(:, 3) = 0;

        otherwise
            error('Unknown distribution type: %s', cfg.distribution);
    end

    % Generate random velocity vectors
    direction = 2*pi*rand(num_users, 1);
    speed = cfg.speed * (0.5 + rand(num_users, 1));

    MSVelo_all(:, 1) = speed .* cos(direction);
    MSVelo_all(:, 2) = speed .* sin(direction);
    MSVelo_all(:, 3) = 0;
end