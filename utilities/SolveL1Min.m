% Author: Yuan Gao
% Date: 2016-02-12

function xh = SolveL1Min(A, y, tau)
% Call l1homotopy of L1_homotopy_v2.0 to solve L1 minimization problem


in = [];
in.tau = tau;
in.delx_mode = 'mil';

out = l1homotopy(A,y,in);

xh = out.x_out;