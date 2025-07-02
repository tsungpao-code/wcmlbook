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

%% Build 3GPP standard simulation environment
% scenario
s = qd_simulation_parameters;                           % Set up simulation parameters (matlab class)
% s.show_progress_bars = 0;                               % Disable progress bars
s.center_frequency = centerFrequency;                   % Set center frequency
s.sample_density = 2;                                   % 2 samples per half-wavelength
s.use_absolute_delays = 1;                              % Include delay of the LOS path
s.use_random_initial_phase = 0;

% layout
L = qd_layout(s);                                       % Create new QuaDRiGa layout
L.no_tx = 1;
L.tx_position = [0; 0; antennaHeight];                       
L.no_rx = num_rx;
Nr = 1; 

LosType = {'UMi_NLOS', 'indoor_open', 'UMi_LOS', 'UMi_LOS', 'UMi_NLOS', 'UMa_LOS'} ;   
scenario = sprintf('3GPP_38.901_%s', LosType{id});
L.set_scenario(scenario);  %scenarios can be selected
switch BSantType
    case '3gpp-3d' % Directional BS antenna
        a_2655_Mhz = qd_arrayant('3gpp-3d',1,M,s.center_frequency,1); % V polarization only
        L.tx_array = a_2655_Mhz;  
    case 'omni' % Omni-directional BS antenna
        L.tx_array = qd_arrayant('omni');  
        L.tx_array.copy_element(1,M);
        L.tx_array.element_position(2,:) = (0:M-1)*3*10^8/s.center_frequency/2; % half-wavelength distance
end
L.rx_array = qd_arrayant('omni');   
if Nr>1 % change one rx_array and all changes
    L.rx_array(1,1).copy_element(1,Nr);
    L.rx_array(1,1).element_position(2,:) = (0:Nr-1)*antennaSpacing*lambda - (Nr/2-0.5)*antennaSpacing*lambda;
end

%% Create tracks
% speed_all = 3 + rand(numSamples, 1) * (200 - 3);
speed_all = speed * ones(numSamples, 1);
label = [label speed_all]; 

for i=1:L.no_rx
    speed = speed_all(i);
    name = L.track(1,i).name;
    L.track(1,i) = qd_track('linear', time_grid*(sequenceLength-1)*speed);  
    L.track(1,i).name = name;
    L.track(1,i).set_speed(speed)
    L.track(1,i).interpolate('time', time_grid, [], [], 1);
    L.track(1,i).scenario = scenario;
end
L.rx_position = Location;

% Interpolate positions to get spacial samples
% interpolate_positions(L.track, s.samples_per_meter) 

% Environment related parameters
clusterSet = [5, 40];
b = L.init_builder;  
SClambda = 20;     % SSF decorrelation distance, settings for spatial consistency
Nclusters = clusterSet(id); % Number of clusters, no smaller than 2, 1st is LOS 
Nsubpaths = 10;             % Number of subpaths in each NLOS cluster
PerClusterAS = 1;           % Per cluster AS (azimuth spread), degree
b.scenpar.PerClusterDS = 0; % Disable per-cluster delay spread, enforced to be 0 for spatial consistency
% if strcmp(scenario, 'LOS')
    b.scenpar.NumClusters = Nclusters;     
    b.scenpar.NumSubPaths = Nsubpaths; 
    b.scenpar.KF_mu = -3;        % Rician K factor;
    b.scenpar.KF_sigma = 0.5;
% end
b.scenpar.SC_lambda = SClambda;
b.scenpar.PerClusterAS_A = PerClusterAS; 
b.scenpar.PerClusterAS_D = PerClusterAS;
b.scenpar.PerClusterES_A = 0;  % (elevation spread)
b.scenpar.PerClusterES_D = 0;
b.gen_ssf_parameters;        % Generate small-scale-fading parameters
visualize_clusters(b,1);    %view the channal layout of sample 1
c = get_channels(b); 


%% generate CSI data
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
    temp = c(i);
    temp.individual_delays = 0;           % Remove per-antenna delays on each MIMO link
    fcoef = temp.fr(bandwidth,numSubcarriers,1:sequenceLength);  %.function:fr(bandwidth, carriers, i_snapshot), output freq_response (Rx-Antenna , Tx-Antenna , Carrier-Index , Snapshot ], i.e.,(1. 32,512,1)
    fcoef = fcoef(:,:,:,1:sequenceLengthSave);
    switch domain
        case 'angle_delay'
            fcoef = fft(fft(fcoef,[],2),[],3);    % angle-delay domain
            normalization_factor(i, :) = max(abs(fcoef(:)))./2;
            fcoef = fcoef/normalization_factor(i, :);  % normalized over all time
        %     fcoef = fcoef./max(max(abs(fcoef), [], 2), [], 3)./2;  % normalized in [-0.5,0.5]
            fcoef = fcoef(:,:,numSubcarriers-cutout+1:numSubcarriers,:);    % Reserve the last 32 non-zero rows
        case 'spatial_fre'
            fcoef = fcoef(:,:,numSubcarriers-num_sc_used+1:numSubcarriers,:);
            normalization_factor(i, :) = sqrt(mean(mean(abs(fcoef).^2, 3),4));
            fcoef = fcoef ./ normalization_factor(i, :);
    end
    H(i,:,:,:,:) = fcoef;  
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
