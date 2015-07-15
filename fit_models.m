function [results, bms_results] = fit_models(data,models)
    
    % Fit RL models using MFIT.
    %
    % USAGE: [results, bms_results] = fit_models(data,models)
    %
    % INPUTS:
    %   data - [S x 1] structure array of data for S subjects
    %   models (optional) - cell array of model names (default: {'Qlearn1' 'Qlearn2' 'Qlearn1_sticky' 'Qlearn2_sticky'})
    %
    % OUTPUTS:
    %   results - [S x 1] model fits for each subject
    %   bms_results - Bayesian model selection results
    %
    % Sam Gershman, July 2015
    
    if nargin < 2 || isempty(models)
        models = {'Qlearn1' 'Qlearn2' 'Qlearn1_sticky' 'Qlearn2_sticky'};
    end
    
    % create parameter structure
    g = [2 1];  % parameters of the gamma prior
    param(1).name = 'inverse temperature';
    param(1).logpdf = @(x) sum(log(gampdf(x,g(1),g(2))));  % log density function for prior
    param(1).lb = 0;    % lower bound
    param(1).ub = 50;   % upper bound
    
    a = 1.2; b = 1.2;   % parameters of beta prior
    param(2).name = 'learning rate';
    param(2).logpdf = @(x) sum(log(betapdf(x,a,b)));
    param(2).lb = 0;
    param(2).ub = 1;
    
    g = [0 1];  % parameters of the gamma prior
    param(3).name = 'inverse temperature';
    param(3).logpdf = @(x) sum(log(normpdf(x,g(1),g(2))));  % log density function for prior
    param(3).lb = -5;    % lower bound
    param(3).ub = 5;     % upper bound
    
    nstarts = 2;
    
    for m = 1:length(models)
        switch models{m}
            case 'Qlearn1'
                fun = @Qlearn1;
                p = [1 2];
            case 'Qlearn2'
                fun = @Qlearn2;
                p = [1 2 2];
            case 'Qlearn1_sticky'
                fun = @Qlearn1_sticky;
                p = [1 2 3];
            case 'Qlearn2_sticky'
                fun = @Qlearn2_sticky;
                p = [1 2 2 3];
        end
        
        results(m) = mfit_optimize(fun,param(p),data,nstarts);
    end
    
    [alpha,exp_r,xp,pxp,bor] = mfit_bms(results);
    bms_results.alpha = alpha;
    bms_results.exp_r = exp_r;
    bms_results.xp = xp;
    bms_results.pxp = pxp;
    bms_results.bor = bor;