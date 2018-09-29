%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%	Code of S3RC (with single labeled sample per person) described in
%	"Semi-Supervised Sparse Representation Based Classification for Face 
%    Recognition with Insufficient Labeled Samples", 
%   Yuan Gao, Jiayi Ma, and Alan Yuille. 
%   IEEE Transactions on Image Processing, 2017.
%
%	e-mail: Ethan.Y.Gao@gmail.com, 
%
% ********************************************************************%%%%%%
% Multi-PIE database can be purchased from:
% http://www.cs.cmu.edu/afs/cs/project/PIE/MultiPie/Multi-Pie/Home.html               
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear;  
close all;

addpath L1_homotopy_v2.0
addpath L1_homotopy_v2.0/utils
addpath Dataset
addpath utilities
addpath EM
addpath SRC


%%%%%%%%%%%%%%%%%%%%%%%%%%
TrainDataSet = 'S_1_Ca_051_Re_01.mat';
TestDataSet = 'S_2_Ca_051_Re_01.mat';
nPCs = 90;
nSamplesPerSubject = 20;
nSubject = 100;
GalleryIndex = 8;
%%%%%%%%%%%%%%%%%%%%%%%%%%
lambda = 0.001;
EM_MaxIter = 5;
%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('************************\n');
fprintf('Train Subset: %s\n', TrainDataSet);
fprintf('Test Subset: %s\n', TestDataSet);
fprintf('nSamplesPerSubject = %d\n', nSamplesPerSubject);
fprintf('Number of PCs = %d\n', nPCs);
fprintf('lambda = %d\n', lambda);
fprintf('************************\n');
    
load(TrainDataSet)
Data = double(DAT);
Label(Label>213) = Label(Label>213) -1;

N = size(Data, 2); 

UniLabel = 1:nSubject;
K = length(UniLabel);


%% Gallary Data
TrainData = Data(:,GalleryIndex:nSamplesPerSubject:end);
TrainLabel = Label(GalleryIndex:nSamplesPerSubject:end);

TrainData = TrainData(:, TrainLabel<=nSubject);
TrainLabel = TrainLabel(TrainLabel<=nSubject);
NTrain = length(TrainLabel);

D = size(TrainData, 1);

%% Generic Training and Reference Datasets
OtherTrainData = Data(:, Label > nSubject);
OtherTrainLabel = Label(Label > nSubject);

%% Initialize V from ESRC.
% Note V can be replaced by the result of other variational
% Dictionary learning method.
V = [];
for iter = nSubject+1:OtherTrainLabel(end)
    Samples = OtherTrainData(:,OtherTrainLabel == iter);
    mu = mean(Samples, 2);
    V = [V, bsxfun(@minus, Samples, mu)];
end

%% PCA
[coeff, ~, ~] = compute_PCA(TrainData');
coeff = coeff(:, 1: nPCs);

TrainData   = coeff' * TrainData;
TrainData   = TrainData./( repmat(sqrt(sum(TrainData.*TrainData)), [size(TrainData, 1),1]) ); % normalize

V   = coeff' * V;
V   =  V./( repmat(sqrt(sum(V.*V)), [size(V, 1),1]) ); % normalize




%% Testing Dataset
load(TestDataSet)
TestData = double(DAT);
Label(Label>213) = Label(Label>213) -1;
TestLabel = Label;


TestData = coeff' * TestData;
TestData = TestData(:, TestLabel<=nSubject);
TestLabel = TestLabel(TestLabel<=nSubject);

TestData   =  TestData./( repmat(sqrt(sum(TestData.*TestData)), [size(TestData, 1),1]) );

[D,N2] = size(TestData);


%% ESRC Initialization
fprintf('ESRC_Mean, first mean then normalize: \n');
[ESRC_Accu, X_ESRC] = Fun_ESRC(TrainData,TrainLabel,TestData,TestLabel,V,lambda);
fprintf('Accuracy for ESRC_Mean = %f; \n', ESRC_Accu);

%% rectify Data by the results of ESRC3
beta = X_ESRC(K+1:end,:);        

Test_tmp = TestData - V * beta;
DataRec_ESRC3 = [TrainData, Test_tmp];
DataRec_ESRC3 =  DataRec_ESRC3./( repmat(sqrt(sum(DataRec_ESRC3.*DataRec_ESRC3)), [size(DataRec_ESRC3, 1),1]) );

TrainIndex_ESRC = 1:size(TrainData, 2);    

%% prior for GMM
n = ones(nSubject, 1);
Pi = n / NTrain;

%% initialize GMM-EM using the mean and variance from K-means 
conf_ESRC.mu = TrainData;
conf_ESRC.SIGMA = ones(D,K);
conf_ESRC.Pi = Pi;
conf_ESRC.MaxIter = EM_MaxIter;
conf_ESRC = conf_init(conf_ESRC, K);

fprintf('*****************  EM  ********************\n')
Output = GmmEm(DataRec_ESRC3, TrainLabel, TrainIndex_ESRC, TestLabel, conf_ESRC);      % GMM-EM Clustering
fprintf('************  EM Finished  ****************\n')

%% Construct the new dictionary with the learned gallery from GMM-EM
mu_S3RC = Output.mu;
mu_S3RC = mu_S3RC./( repmat(sqrt(sum(mu_S3RC.*mu_S3RC)), [size(mu_S3RC, 1),1]) );

A = [mu_S3RC, V];

%% Final S3RC Classification
FinalLabel = [];

parfor i = 1:size(TestData,2)
    y  = TestData(:,i);
    x  = SolveL1Min(A, y, lambda*2);
    y_add = V * x(K+1:end,1);
    r = [];
    for j = 1:K
        cdat = mu_S3RC(:, j);
        er   = y - cdat * x(j) - y_add;
        r(j) = er(:)'*er(:);
    end
    index = find(r == min(r));
    FinalLabel(i,1) = UniLabel(index(1));
end

S3RC_Accu_ESRC_3 = sum(FinalLabel==TestLabel)/length(TestLabel);
fprintf('Accuracy for S3RC_ESRC_3 = %f \n', S3RC_Accu_ESRC_3);
