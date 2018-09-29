function conf = conf_init(conf,K)

if ~isfield(conf,'K'), conf.K = K; end;
if ~isfield(conf,'MaxIter'), conf.MaxIter = 50; end;
if ~isfield(conf,'Pi'), conf.Pi(1:K,1) = 1/K;  end;
if ~isfield(conf,'ecr'), conf.ecr = 1e-10; end;
if ~isfield(conf,'minP'), conf.minP = 1e-10; end;
if ~isfield(conf,'UpdateMean'), conf.UpdateMean = 1; end;
if ~isfield(conf,'UpdateVar'), conf.UpdateVar = 1; end;
if ~isfield(conf,'UpdateX'), conf.UpdateX = 0; end;