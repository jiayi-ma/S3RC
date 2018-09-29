% Author: Yuan Gao
% Date: 2016-02-12

% E-Step

function [P, E]=get_P(X,mu, sigma2 ,Pi, minP)

pi = 3.1415926;
M = size(mu, 2);
[D,N] = size(X);

Sigma2 = max(sigma2 , minP);

for i = 1:M
    X_Minus_mu = bsxfun(@minus, X, mu(:,i));
    InvSigma = diag(1./Sigma2(:,i));
    Mod2_X_Minus_mu(i,:) = diag(X_Minus_mu' * InvSigma * X_Minus_mu);
    
end

ModLogSigma = sum(log(Sigma2),1)';            % ModLogSigma = log(|Sigma|)
tmp1 = -D/2 * log(2*pi) - ModLogSigma/2;      % tmp1 = -D/2 * log(2*pi) - log(|Sigma|)/2
LLC = log(repmat(Pi,[1,N])) - Mod2_X_Minus_mu/2 + repmat(tmp1,[1,N]);
LStar = max(LLC);
LL  = log(sum(exp(LLC - repmat(LStar,[M,1])),1)) + LStar;
gamma = LLC - repmat(LL,[M,1]);
P = exp(gamma);



E = sum(sum(P .* ( repmat(tmp1, [1,N]) - Mod2_X_Minus_mu/2 + log(repmat(Pi,[1,N])) ) ));