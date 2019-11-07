function [results, bms_results, empirical_prior] = fit_models(data,paramfun,models)
    
    % Fit RL models using MFIT.
    %
    % USAGE: [results, bms_results] = fit_models(data,paramfun,models)
    %
    % INPUTS:
    %   data - [S x 1] structure array of data for S subjects
    %   paramfun - function handle that takes the model name as input and returns a parameter structure
    %   models (optional) - cell array of model names (default: {'Qlearn1' 'Qlearn2' 'Qlearn1_sticky' 'Qlearn2_sticky'})
    %
    % OUTPUTS:
    %   results - [S x 1] model fits for each subject
    %   bms_results - Bayesian model selection results
    %   empirical_prior - prior fit to parameter estimates
    %
    % Sam Gershman, July 2015
    
    if nargin < 3 || isempty(models)
        models = {'Qlearn1' 'Qlearn2' 'Qlearn1_sticky' 'Qlearn2_sticky'};
    end
    
    if nargin < 2 || isempty(paramfun)
        paramfun = @(model) RL_paramfun(model,'uniform');
    end
    
    for m = 1:length(models)
        
        % get parameter structure
        param = paramfun(models{m});
        
        % fit model
        fun = str2func(models{m});
        R = mfit_optimize(fun,param,data);
        
        % collect latent variables
        for s = 1:length(data)
            [~,R.latents(s)] = fun(R.x(s,:),data(s));
        end
        
        % estimate prior
        empirical_prior.(models{m}) = mfit_priorfit(R.x,param);  % estimate prior
        
        % store results
        results(m) = R;
    end
    
    % Bayesian model selection
    if nargout > 1
        bms_results = mfit_bms(results);
    end