% Exercise 2.12: QuaDRiGa Channel Generation for LISTA-CE
%
% This script demonstrates how to:
% 1. Configure 3GPP channel models using QuaDRiGa.
% 2. Generate channel coefficients for channel estimation tasks.
% 3. Format and save the data for training Deep Learning models.
%
% Scenario Configuration:
% - Scenario: 3GPP 38.901 UMi_NLOS (Outdoor) or Indoor Open
% - Frequency: 2.655 GHz
% - Bandwidth: 10 MHz
% - BS Antennas: 32 (Omni/3GPP-3D)
% - UE Antennas: 1
%
% TODO:
% 1. Configure simulation parameters (QD simulation parameters).
% 2. Create Layout and Tracks (User movement).
% 3. Extract and normalize Channel State Information (CSI).

clear
clc

id = 1
set(0,'defaultTextFontSize', 12)                        % Default Font Size
set(0,'defaultAxesFontSize', 12)                        % Default Font Size
set(0,'defaultAxesFontName','Times')                    % Default Font Type
set(0,'defaultTextFontName','Times')                    % Default Font Type
set(0,'defaultFigurePaperPositionMode','auto')          % Default Plot position
set(0,'DefaultFigurePaperType','<custom>')              % Default Paper Type
set(0,'DefaultFigurePaperSize',[14.5 7.3])            	% Default Paper Size

test = '';
switch test % train:1, test: 9999
    case 'test'
        randn('seed',9999);  
        rand('seed',9999);
    case 'valid'
        randn('seed',1234);  
        rand('seed',1234);
    case ''
        randn('seed',1);  
        rand('seed',1);
end

domain = 'angle_delay';

%% frequency related parameters
centerFrequency = 2.655e9;            % Center frequency in Hz
bandwidth = 10 * 1e6;                     % Bandwidth in Hz
numSubcarriers = 32;               % Number of sub-carriers
cutout = 32;                        % Reserve 32 non-zero rows
subSampling = 1;                     % Only take every subsampling's sub-carriers
lambda = 3e8/centerFrequency;
num_sc_used = 72;

%% BS related parameters
BSantType = 'omni';
vertical = false;
antennaHeight = 25;                  % Antenna height of the bse station in m
antennaSpacing = 1/2;                % Antenna spacing in multiples of the wave length
M = 32;                              % Total number of antennas
region = 1;  %save 1 region's datasets in one-time generation (due to limited Matlab memory)
% id = 1;  % region id

train_num = 10;
vali_num = 1;
test_num = 2;           % 用户数
numSamples = train_num + vali_num + test_num;
numSamplesSave = numSamples;
% 
% numSamplesSave = 10000;

%% User related parameters
num_rx = region * numSamples;        % number of single-antenna users
dis_range = 100;                      % maximum distance among users, meter of radius
userHeight = 1.5;                    % Antenna height of the users
DisWithBSx = [50, 0];       % distance between the district center and BS in x-axis, meter
DisWithBSy = [0, 0];     % distance between the district center and BS in y-axis, meter
speed = 100;  % m/s
coherenceTime = lambda/4/speed
num_symbols_coherence_time = coherenceTime/1e-3*14*subSampling  % assume 1ms slot duration, 14 symbols, subcarrier spacing*4->symbol duration/4 
sequenceLength = 14;                % Number of steps in a track to simulate for each drop
time_grid = 66.66e-6;               % time interval to sample = ofdm symbol duration = 1/subcarrier spacing 
sequenceLengthSave = 1;

%% randomly generate 2D user positions
Location = zeros(3, num_rx);
Location(3,:) = userHeight;
for i = id:region+id-1     %for genetating task id, use"i = id:region+id-1"
    Location(1,(1*numSamples-numSamples+1):1*numSamples) = DisWithBSx(i) + dis_range*(rand(numSamples,1)).*cos(2*pi*rand(numSamples,1));
    Location(2,(1*numSamples-numSamples+1):1*numSamples) = DisWithBSy(i) + dis_range*(rand(numSamples,1)).*sin(2*pi*rand(numSamples,1));
end
label = Location(1:2, :).';

figure;
scatter(Location(1,:),Location(2,:),'+');
hold on;
plot(0,0,'ro','MarkerFaceColor','r');
axis([-200 200 -200 200]);
grid on
grid on;
set(gca, 'GridLineStyle', ':');  
set(gca, 'GridAlpha', 1);  
set(gca, 'XTick', -200:20:200);  
set(gca, 'YTick', -200:20:200);
xlabel('x-cord [m]');
ylabel('y-cord [m]');
box on

%% Build 3GPP standard simulation environment (TODO)

% TODO: Set up simulation parameters using qd_simulation_parameters
% 1. Create a simulation parameter object 's'
% 2. Set center_frequency, sample_density, etc.

% TODO: Create QuaDRiGa layout
% 1. Create a layout object 'L' using 's'
% 2. Configure number of transmitters (no_tx) and receivers (no_rx)
% 3. Set standard 3GPP scenario (e.g., '3GPP_38.901_UMi_NLOS')

% TODO: Configure Antennas
% 1. Create transmit antenna array (e.g., omni-directional, M elements)
% 2. Create receive antenna array (omni-directional, Nr elements)
% 3. Assign arrays to L.tx_array and L.rx_array

%% Create tracks (TODO)

% TODO: Create user tracks (movement)
% 1. Iterate through all receivers (L.no_rx)
% 2. Create qd_track objects with linear motion
% 3. Set track name, speed, and scenario
% 4. Assign tracks to the layout

L.rx_position = Location;

% Environment related parameters
clusterSet = [5, 40];
b = L.init_builder;  
SClambda = 20;     % SSF decorrelation distance
Nclusters = clusterSet(id); 
Nsubpaths = 10; 
PerClusterAS = 1;     

% Apply settings
b.scenpar.PerClusterDS = 0; 
b.scenpar.NumClusters = Nclusters;     
b.scenpar.NumSubPaths = Nsubpaths; 
b.scenpar.KF_mu = -3;
b.scenpar.KF_sigma = 0.5;
b.scenpar.SC_lambda = SClambda;
b.scenpar.PerClusterAS_A = PerClusterAS; 
b.scenpar.PerClusterAS_D = PerClusterAS;

% Generate small-scale fading
b.gen_ssf_parameters;        
c = get_channels(b); 


%% generate CSI data (TODO)
switch domain
    case 'angle_delay'
        H = zeros(numSamplesSave, Nr, M, cutout, sequenceLengthSave);
        normalization_factor = zeros(numSamplesSave, 1);
    case 'spatial_fre'
        H = zeros(numSamplesSave, Nr, M, num_sc_used, sequenceLengthSave);
        normalization_factor = zeros(numSamplesSave, M);
end
label = label(1:numSamplesSave, :);

for i = 1: numSamplesSave
    % TODO: Extract Channel Coefficients
    % 1. Get channel builder/coefficients for sample 'i'
    % 2. Compute frequency response using 'fr' method
    % 3. Perform FFT if domain is 'angle_delay'
    % 4. Normalize the coefficients
    % 5. Store in H matrix
    
    disp(['This is the ',num2str(i),'-th sample'])
end
output_h=squeeze(H);

region = 1;
for i = 1 : numSamplesSave
    H_data(:,:,i) = reshape(output_h(i,:,:), [cutout*M, region]);
end

H_data = reshape(H_data,[cutout*M, region*numSamplesSave]); 

train_data = zeros(train_num, M*cutout*2);
vali_data = zeros(vali_num, M*cutout*2);
test_data = zeros(test_num, M*cutout*2);
 
for kk = 1: train_num
% for kk = 1
    H_data_temp = reshape(H_data(:,kk), [M,cutout]);
    H_data_all = [real(H_data_temp);imag(H_data_temp)];
    train_data(kk,:) = reshape(H_data_all, [1, M*cutout*2]);
end
for kk = 1: vali_num
    H_data_temp = reshape(H_data(:,kk+train_num), [M,cutout]);
    H_data_all = [real(H_data_temp);imag(H_data_temp)];
    vali_data(kk,:) = reshape(H_data_all, [1, M*cutout*2]);
end
for kk = 1: test_num
    H_data_temp = reshape(H_data(:,kk+train_num+vali_num), [M,cutout]);
    H_data_all = [real(H_data_temp);imag(H_data_temp)];
    test_data(kk,:) = reshape(H_data_all, [1, M*cutout*2]);
end

% 进行统一的归一化后存储
train_data = (train_data)./30;
vali_data = (vali_data)./30;
test_data = (test_data)./30;

save('D:\Users\20532\Compressive_Sensing\Train_data_LISTA_CE_BeamFreq_Path1.mat','train_data','-v7');
save('D:\Users\20532\Compressive_Sensing\Vali_data_LISTA_CE_BeamFreq_Path1.mat','vali_data','-v7');
save('D:\Users\20532\Compressive_Sensing\Test_data_LISTA_CE_BeamFreq_Path1.mat','test_data','-v7');
