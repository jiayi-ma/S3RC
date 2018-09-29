% Author: Yuan Gao
% Date: 2016-02-12

% EM algorithm on Gaussian Mixture Model
% This needs the train label to be continous!~!

function Output = GmmEm(X, TrainLabel, TrainIndex, TestLabel, conf)

Pi = conf.Pi;
mu = conf.mu;
SIGMA = conf.SIGMA;
K = conf.K;

[D,N] = size(X);
M = conf.K;

iter=0; tecr=conf.ecr+10; E=1; 
PP = {};
MU = {};

while iter < conf.MaxIter
    iter=iter+1;
    fprintf('iter = %d; ', iter);
    
    %% E-step
    E_old=E;
    [P, E] = get_P(X, mu, SIGMA ,Pi, conf.minP);
    EE(iter) = E;
    
    %% Check Converge
    tecr=abs((E-E_old)/E);
    
    if (iter > conf.MaxIter) || (tecr < conf.ecr)
        break;
    end
    
    %% Rec P
    for i = 1:length(TrainIndex)
        P(:,TrainIndex(i)) = zeros(size(P, 1),1);
        P(TrainLabel(i),TrainIndex(i)) = 1;
    end
    
    PP{iter} = P;
    
    FinalLabel = [];
    PPP = P;
    PPP(:, TrainIndex) = [];
    for i = 1:size(PPP, 2)
        
        tmp = find(PPP(:,i) == max(PPP(:,i)), 1);
        FinalLabel(i,1) = tmp;
    end

    
    %% M-step
    Np = sum(P, 2);
    Pi = sum(P,2)/N;





   % Update mu
    if conf.UpdateMean
        for i = 1:M
            mu(:,i) = sum(X .* repmat(P(i,:), D, 1), 2)/Np(i);
        end
    end
    MU{iter} = mu;
   

%     
   %Update Variance
    if conf.UpdateVar
        for i = 1:M
%             x = X{i};
            X_Minus_mu = bsxfun(@minus, X, mu(:,i));
            X_Minus_mu2 = X_Minus_mu .^ 2;
            SIGMA(:,i) = sum(bsxfun(@times, X_Minus_mu2, P(i,:)), 2)/Np(i) ;
        end
    end
end

accuracy = sum(FinalLabel == TestLabel)/length(TestLabel);

Output.EE = EE;
Output.PP = PP;
Output.Pi = Pi;
Output.MU = MU;
Output.SIGMA = SIGMA;
Output.mu = mu;
Output.iter = iter;
Output.accuracy = accuracy;

fprintf('\n');