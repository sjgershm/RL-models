function [opts, param] = set_opts(opts)
    
    % Fill in missing options and create parameter structure
    %
    % USAGE: [opts, param] = set_opts([opts])
    %
    % INPUTS:
    %   opts - options structure with any of the following fields:
    %           .go_bias - bias towards "go" action (default: false)
    %           .sticky - choice stickiness (default: false)
    %           .dual_learning_rate - different learning rates for pos and
    %            neg prediction errors (default: false)
    %           .lapse - lapse rate (default: false)
    %           .inverse_temp - inverse temperature (default: true)
    %           .pavlovian_bias - bias towards "go" action for states with
    %            high average reward (default: false)
    %           .sensitivity - reward/punishment sensitivity. 0 =
    %            sensitivity fixed at 1; 1 = common sensitivity for rewards
    %            and punishments; 2 = different sensitivity for rewards and
    %            punishmesn (default: 0)
    %
    % OUTPUTS:
    %   opts - options structure with missing fields added
    %   param - parameter structure
    %
    % Sam Gershman, Nov 2015
    
    % default options
    def_opts.go_bias = false;
    def_opts.sticky = false;
    def_opts.dual_learning_rate = false;
    def_opts.lapse = false;
    def_opts.inverse_temp = true;
    def_opts.pavlovian_bias = false;
    def_opts.sensitivity = 0;
    def_opts.latents = false;
    
    % fill in missing or empty fields
    if nargin < 1 || isempty(opts)
        opts = def_opts;
    else
        F = fieldnames(def_opts);
        for f = 1:length(F)
            if ~isfield(opts,F{f}) || isempty(opts.(F{f}))
                opts.(F{f}) = def_opts.(F{f});
            end
        end
    end
    
    opts.ix = ones(1,9);
    if ~opts.go_bias; opts.ix(6) = 0; end
    if ~opts.sticky; opts.ix(4) = 0; end
    if ~opts.dual_learning_rate; opts.ix(3) = 0; end
    if ~opts.lapse; opts.ix(9) = 0; end
    if ~opts.inverse_temp; opts.ix(1) = 0; end
    if ~opts.pavlovian_bias; opts.ix(5) = 0; end
    if opts.sensitivity == 0; opts.ix(7:8) = 0; end
    if opts.sensitivity == 1; opts.ix(8) = 0; end
    
    %---------- create parameter structure ---------------%
    
    param(1).name = 'beta';
    param(1).hp = [3 2];    % hyperparameters of the gamma prior
    param(1).logpdf = @(x) sum(log(gampdf(x,param(1).hp(1),param(1).hp(2))));  % log density function for prior
    param(1).lb = 1e-8; % lower bound
    param(1).ub = 50;   % upper bound
    param(1).fit = @(x) gamfit(x);
    
    param(2).name = 'lr_pos';
    param(2).hp = [1.2 1.2];    % hyperparameters of beta prior
    param(2).logpdf = @(x) sum(log(betapdf(x,param(2).hp(1),param(2).hp(2))));
    param(2).lb = 0;
    param(2).ub = 1;
    param(2).fit = @(x) betafit(x);
    
    param(3) = param(2);
    param(3).name = 'lr_neg';
    
    param(4).name = 'choice stickiness';
    param(4).hp = [0 3]; % hyperparameters of the normal prior
    param(4).logpdf = @(x) sum(log(normpdf(x,param(4).hp(1),param(4).hp(2))));  % log density function for prior
    param(4).lb = -5;    % lower bound
    param(4).ub = 5;     % upper bound
    param(4).fit = @(x) [mean(x) std(x)];
    
    param(5) = param(4);
    param(5).name = 'pi';
    
    param(6) = param(4);
    param(6).name = 'b';
    
    param(7) = param(1);
    param(7).name = 'rho_pos';
    
    param(8) = param(1);
    param(8).name = 'rho_neg';
    
    param(9).name = 'epsilon';
    param(9).hp = [1.2 2];    % hyperparameters of beta prior
    param(9).logpdf = @(x) sum(log(betapdf(x,param(9).hp(1),param(9).hp(2))));
    param(9).lb = 0;
    param(9).ub = 0.99;
    param(9).fit = @(x) betafit(x);
    
    param = param(opts.ix==1);