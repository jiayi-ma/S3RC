% Author: Yuan Gao
% Date: 2016-02-12

function [coeff, score, latent] = compute_PCA(X)
% PCA aims to find a transformation of X, coeff, so that column of X*coeff
% is orthogonal with each other. (coeff rotate the coordinate of X).
% i.e. (X*coeff)'* (X*coeff) has a Diagonal Matrix result. 
% X = U * sigma * coeff (by SVD), thus the singlar value of X'X = sigma.^2 = latent.
% coeff is the coefficient matrix.
% score is the selected PCs, i.e. score(:,1:N) is the first N PCs. It is X * coeff.

mu = mean(X);
X = bsxfun(@minus, X, mu);

[~,sigma, coeff] = svd(X);
sigma = diag(sigma);

% Enforce a sign convention on the coefficients -- the largest element in
% each column will have a positive sign.
[~,maxind] = max(abs(coeff), [], 1);
[d1, d2] = size(coeff);
colsign = sign(coeff(maxind + (0:d1:(d2-1)*d1)));
coeff = bsxfun(@times, coeff, colsign);

score = X*coeff;
% score =  bsxfun(@times,U,sigma');
% if nargout > 1
%     score = bsxfun(@times, score, colsign); % scores = score
% end
latent = sigma.^2;


