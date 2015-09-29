function [results, bms_results, param] = fit_models(data,models)
    
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
    param(1).name = 'inverse temperature';
    param(1).hp = [3 2];    % hyperparameters of the gamma prior
    param(1).logpdf = @(x) sum(log(gampdf(x,param(1).hp(1),param(1).hp(2))));  % log density function for prior
    param(1).lb = 1e-8; % lower bound
    param(1).ub = 50;   % upper bound
    param(1).fit = @(x) gamfit(x);
    
    param(2).name = 'learning rate';
    param(2).hp = [1.2 1.2];    % hyperparameters of beta prior
    param(2).logpdf = @(x) sum(log(betapdf(x,param(2).hp(1),param(2).hp(2))));
    param(2).lb = 0;
    param(2).ub = 1;
    param(2).fit = @(x) betafit(x);
    
    param(3).name = 'choice stickiness';
    param(3).hp = [0 3]; % hyperparameters of the normal prior
    param(3).logpdf = @(x) sum(log(normpdf(x,param(3).hp(1),param(3).hp(2))));  % log density function for prior
    param(3).lb = -5;    % lower bound
    param(3).ub = 5;     % upper bound
    param(3).fit = @(x) [mean(x) std(x)];
    
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
        
        R = mfit_optimize(fun,param(p),data,nstarts);
        for s = 1:length(data)
            [~,R.data(s)] = fun(R.x(s,:),data(s));
        end
        R.param = mfit_priorfit(R.x,param(p));  % estimate prior
        results(m) = R;
    end
    
    % Bayesian model selection
    if nargout > 1
        [alpha,exp_r,xp,pxp,bor] = mfit_bms(results);
        bms_results.alpha = alpha;
        bms_results.exp_r = exp_r;
        bms_results.xp = xp;
        bms_results.pxp = pxp;
        bms_results.bor = bor;
    end