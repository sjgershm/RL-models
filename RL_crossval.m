function RL_crossval
    
    load RL_model_results_informative
    
    [logp_informative,results_informative] = mfit_crossval(@Qlearn2_sticky,results(4).param,folds);
    param = results(4).param; param(1).hp=[3 2]; param(2).hp=[1.2 1.2]; param(3).hp=[1.2 1.2]; param(4).hp=[0 3];
    [logp_heuristic, results_heuristic] = mfit_crossval(@Qlearn2_sticky,param,folds);
    for i=1:4; param(i).logpdf = @(x) 0; end
    [logp_uninformative, results_uninformative] = mfit_crossval(@Qlearn2_sticky,param,folds);
    
    n = 0;
    for i = 1:4
        for j = 1:4
            if i ~= j
                n = n + 1;
                r(n,1) = corr(results_informative(i).x(:),results_informative(j).x(:));
                r(n,2) = corr(results_uninformative(i).x(:),results_uninformative(j).x(:));
            end
        end
    end
    
    save crossval_results logp_uninformative logp_informative results_uninformative results_informative r