function [results, bms_results] = fit_models(data,opts)
    
    % Fit RL models using MFIT.
    %
    % USAGE: [results, bms_results] = fit_models(data,opts)
    %
    % INPUTS:
    %   data - [S x 1] structure array of data for S subjects
    %   opts - [M x 1] structure of model options (see set_opts.m)
    %
    % OUTPUTS:
    %   results - [M x 1] model fits
    %   bms_results - Bayesian model selection results
    %
    % Sam Gershman, Nov 2015
    
    for m = 1:length(opts)
        
        disp(['... fitting model ',num2str(m),' out of ',num2str(length(opts))]);
        
        % get parameter structure
        [opts1, param] = set_opts(opts(m));
        
        % fit model
        tic
        fun = @(x,data) Qlearn(x,data,opts1);
        R = mfit_optimize(fun,param,data);
        toc
        R.opts = opts1;
        
        % collect latent variables
        if opts1.latents
            for s = 1:length(data)
                [~,R.latents(s)] = fun(R.x(s,:),data(s));
            end
        end
        
        % fit empirical prior
        R.param_empirical = mfit_priorfit(R.x,param);
        
        results(m) = R;
        
    end
    
    % Bayesian model selection
    if nargout > 1
        bms_results = mfit_bms(results);
    end