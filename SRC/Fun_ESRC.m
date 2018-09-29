% Author: Yuan Gao
% Date: 2016-02-12

function [correct_rate, X] = Fun_ESRC(TrainData,TrainLabel,TestData,TestLabel,V,lambda)
% lambda is the parameter of the coefficient's regularization, e.g., lambda = 1e-3
% TrainData (D \times N1) is the training data, where D is the feature dimension and N1 is the number of training samples
% TrainLabel (N1 \times 1) is a set of numbers from 1 to #class, where N1 is the number of training samples
% TestData (D \times N2) is the testing data, where D is the feature dimension and N2 is the number of testing samples
% TestLabel (N2 \times 1) is the groundtruth label for evaluation, where N2 is the number of testing samples
% V (D \times N3) is the shared variation dictionary, where D is the feature dimension and N3 is the dictionary size


D        =  size(TrainData,1);
NTrain   =  length(TrainLabel);
UniLabel =  unique(TrainLabel);
K        =  length(UniLabel);

TrainData  =  TrainData./( repmat(sqrt(sum(TrainData.*TrainData)), [D,1]) );
V          = V./( repmat(sqrt(sum(V.*V)), [D,1]) );
TestData   =  TestData./( repmat(sqrt(sum(TestData.*TestData)), [D,1]) );

A = [TrainData, V];
Ind = size(TrainData, 2);

parfor i = 1:size(TestData,2)
    y  = TestData(:,i);
    x  = SolveL1Min(A, y, lambda);
    y_add = V * x(Ind+1:end,1);
    X(:, i) = x;
    r = [];
    for j = 1:K
        class = UniLabel(j);
        cdat = A(:, TrainLabel==class);
        er   = y - cdat*x(TrainLabel==class) - y_add;
        r(j) = er(:)'*er(:);
    end
    index = find(r == min(r));
    FinalLabel(i,1) = UniLabel(index(1));
end
correct_rate = sum(FinalLabel==TestLabel)/length(TestLabel);