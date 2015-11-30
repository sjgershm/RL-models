function opts = factorial_models(Opts)
    
    % Create a factorial space of models.
    %
    % USAGE: opts = factorial_models(Opts)
    %
    % INPUTS:
    %   Opts - meta options structure, with the same fields as "opts" (see
    %          set_opts.m) but each field accepts a vector of parameters
    %
    % OUTPUTS:
    %   opts - [1 x M] options structure, where each structure in the array
    %          corresponds to one model (a particular combination of settings from Opts)
    %
    % Sam Gershman, Nov 2015
    
    if nargin < 1 || isempty(Opts)
        Opts.go_bias = [false true];
        Opts.sticky = false;
        Opts.dual_learning_rate = false;
        Opts.lapse = [false true];
        Opts.inverse_temp = [false true];
        Opts.pavlovian_bias = [false true];
        Opts.sensitivity = [0 1 2];
    end
    
    g = CombVec(Opts.go_bias,Opts.sticky,Opts.dual_learning_rate,Opts.lapse,Opts.inverse_temp,Opts.pavlovian_bias,Opts.sensitivity);
    
    for m = 1:size(g,2)
        
        opts(m).go_bias = g(1,m);
        opts(m).sticky = g(2,m);
        opts(m).dual_learning_rate = g(3,m);
        opts(m).lapse = g(4,m);
        opts(m).inverse_temp = g(5,m);
        opts(m).pavlovian_bias = g(6,m);
        opts(m).sensitivity = g(7,m);
        
    end