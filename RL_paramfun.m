function param = RL_paramfun(model,mode)
    
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
    
    switch model
        case 'Qlearn1'
            p = [1 2];
        case 'Qlearn2'
            p = [1 2 2];
        case 'Qlearn1_sticky'
            p = [1 2 3];
        case 'Qlearn2_sticky'
            p = [1 2 2 3];
    end
    
    param = param(p);
    
    switch mode
        case 'uniform'
            for i = 1:length(param); param(i).logpdf = @(x) 0; end
        case 'empirical'
            load results_heuristic
            param = empirical_prior.(model);
    end