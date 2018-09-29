%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%	Code of S3RC (with insufficient labeled samples) described in
%	"Semi-Supervised Sparse Representation Based Classification for Face 
%    Recognition with Insufficient Labeled Samples", 
%   Yuan Gao, Jiayi Ma, and Alan Yuille. 
%   IEEE Transactions on Image Processing, 2017.
%
%	e-mail: Ethan.Y.Gao@gmail.com, 
%
% ********************************************************************%%%%%%
% AR database can be requested from:
% http://cbcsl.ece.ohio-state.edu/protected-dir/AR_warp_zip.zip.                
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
DataSet = 'AR_Data.mat';
nPCs = 300;
nSamplesPerSubject = 26;
nLabeledEachClass = 2;
%%%%%%%%%%%%%%%%%%%%%%%%%%
lambda = 0.005;
EM_MaxIter = 5;
%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('************************\n');
fprintf('Dataset: %s\n', DataSet);
fprintf('nSamplesPerSubject = %d\n', nSamplesPerSubject);
fprintf('Number of PCs = %d\n', nPCs);
fprintf('lambda = %d\n', lambda);
fprintf('************************\n');

load(DataSet)
N = size(Data, 2);


%% Determine training and testing
Index = 1:N;
UniLabel = unique(Label);
K = length(UniLabel);
TrainIndex = [];
for i = 1:K
    r = randperm(nSamplesPerSubject);
    r = sort(r(1:nLabeledEachClass))';
    TrainIndex = [TrainIndex; r + nSamplesPerSubject*(i - 1)];
end


LabelOrg = Label;
Label = zeros(length(LabelOrg), 1);
for i = 1:length(Label)
    Label(i) = find(UniLabel == LabelOrg(i));
end
UniLabelOrg = UniLabel;
UniLabel = 1:length(UniLabel);

TestIndex = Index;
TestIndex(TrainIndex) = [];

TrainData = Data(:,TrainIndex);
TrainLabel = Label(TrainIndex);
NTrain = length(TrainIndex);

TestData = Data(:,TestIndex);
NTest = length(TestIndex);
TestLabel = Label(TestIndex);



%% Initialize P and V from SSRC.
% Note V can be replaced by the result of other variational
% Dictionary learning method.
V = [];
mu = zeros(size(TrainData, 1), K);
n = zeros(K, 1);
for iter = 1:K
    Samples = TrainData(:, TrainLabel == UniLabel(iter));
    mu(:,iter) = mean(Samples, 2);
    V = [V, bsxfun(@minus, Samples, mu(:,iter))];
    n(iter) = sum(TrainLabel == UniLabel(iter));
end

%% PCA
[coeff, ~, ~] = compute_PCA(TrainData');
coeff = coeff(:, 1:nPCs);

TrainData   = coeff' * TrainData;
TrainData   =  TrainData./( repmat(sqrt(sum(TrainData.*TrainData)), [size(TrainData, 1),1]) );

TestData   = coeff' * TestData;
TestData   =  TestData./( repmat(sqrt(sum(TestData.*TestData)), [size(TestData, 1),1]) );

V   = coeff' * TestData;
V   =  V./( repmat(sqrt(sum(V.*V)), [size(V, 1),1]) ); % normalize 

mu   = coeff' * ,u;
mu   =  mu./( repmat(sqrt(sum(mu.*mu)), [size(mu, 1),1]) ); % normalize



%% SSRC Initialization
fprintf('SSRC: \n');
[SSRC_Accu, X_SSRC] = Fun_SSRC(mu, [1:size(mu, 2)]', TestData, TestLabel, V, lambda);
MU_SSRC = mu;
fprintf('Accuracy for SSRC = %f; \n', SSRC_Accu);

%% rectify Data
beta = X(K+1:end, :);        

Test_tmp = TestData - V * beta;
iter = 1;
for i = 1:N
    if ismember(i, TrainIndex)
        DataRec(:,i) = mu(:, UniLabel == (TrainLabel(TrainIndex == i)));
    else
        DataRec(:,i) = Test_tmp(:,iter);
        iter = iter + 1;
    end
end

DataRec   =  DataRec./( repmat(sqrt(sum(DataRec.*DataRec)), [size(DataRec, 1),1]) );

D = size(DataRec, 1);

%% prior for GMM
Pi = n / NTrain;

%% initialize GMM-EM using the mean and variance from K-means 
conf.mu = mu;
conf.SIGMA = ones(D,K);
conf.Pi = Pi;
conf.MaxIter = EM_MaxIter;
conf = conf_init(conf, K);

fprintf('*****************  EM  ********************\n')
Output = GmmEm(DataRec, TrainLabel, TrainIndex, TestLabel, conf);      % GMM-EM Clustering
fprintf('************  EM Finished  ****************\n')

%% Construct the new dictionary with the learned gallery from GMM-EM
mu_S3RC = Output.mu;
mu_S3RC  =  mu_S3RC./( repmat(sqrt(sum(mu_S3RC.*mu_S3RC)), [size(mu_S3RC, 1),1]) );
A = [mu_S3RC, V];

%% Final S3RC Classification
FinalLabel = [];

parfor i = 1:size(TestData,2)
    y  = TestData(:,i);
    x  = SolveL1Min(A, y, lambda);
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
S3RC_Accu = sum(FinalLabel==TestLabel)/length(TestLabel);

fprintf('Accuracy for S3RC = %f; \n', S3RC_Accu);
fprintf('======================================== \n');