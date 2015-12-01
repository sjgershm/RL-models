function [lik, latents] = Qlearn(x,data,opts)
    
    % Q-learning for a multi-armed bandit.
    
    % USAGE: [lik, latents] = Qlearn2_sticky(x,data,opts)
    %
    % INPUTS:
    %   x - parameters:
    %       x(1) - inverse temperature
    %       x(2) - learning rate (positive prediction errors)
    %       x(3) - learning rate (negative prediction errors)
    %       x(4) - stickiness inverse temperature
    %   data - structure with the following fields (likelihood mode)
    %           .c - [N x 1] choices
    %           .r - [N x 1] rewards
    %           .C - number of choice options
    %           .N - number of trials
    %   opts - options structure (see set_opts.m)
    %
    % OUTPUTS:
    %   lik - log-likelihood
    %   latents - structure with the following fields:
    %           .Q - [N x C] learned values
    %           .W - [N x C] action weights
    %           .P - [N x C] action probabilities
    %           .V - [N x 1] state values
    %           .rpe - [N x 1] reward prediction error for chosen option
    %
    % Sam Gershman, Nov 2015
    
    % set parameters
    y(opts.ix==1) = x;
    
    if ~opts.go_bias; y(6) = 0; end
    if ~opts.sticky; y(4) = 0; end
    if ~opts.dual_learning_rate; y(3) = y(2); end
    if ~opts.lapse; y(9) = 0; end
    if ~opts.inverse_temp; y(1) = 1; end
    if ~opts.pavlovian_bias; y(5) = 0; end
    if opts.sensitivity == 0; y(7:8) = 1; end
    if opts.sensitivity == 1; y(8) = y(7); end
    
    beta = y(1);        % inverse temperature
    lr_pos = y(2);      % learning rate for positive prediction errors
    lr_neg = y(3);      % learning rate for negative prediction errors
    kappa = y(4);       % stickiness coefficient
    pi = y(5);          % Pavlovian bias coefficient
    b = y(6);           % go bias
    rho_pos = y(7);     % reward sensitivity
    rho_neg = y(8);     % punishment sensitivity
    epsilon = y(9);     % lapse rate
    
    lik = 0; C = data.C;
    for n = 1:data.N
        
        % reset values and stickiness for new block or first trial
        if n == 1 || data.block(n)~=data.block(n-1)
            Q = ones(1,C)/C;    % initial action values
            V = 0;              % initial state value
            U = zeros(1,C);     % stickiness
        end
        
        % data for current trial
        c = data.c(n);
        r = data.r(n);
        go = data.go(n);
        
        % compute policy and accumulate log-likelihod
        W = beta*Q + b*go + pi*V*go + kappa*U;              % action weights
        P = (1-epsilon)*(exp(W)./sum(exp(W))) + epsilon/C;  % softmax + lapse
        lik = lik + log(P(c));
        
        % store latent variables and update values
        if r < 0; rho = rho_neg; else rho = rho_pos; end
        rpe = rho*r - Q(c);
        
        if nargout > 1
            latents.rpe(n,1) = rpe;
            latents.Q(n,:) = Q;
            latents.V(n,1) = V;
            latents.W(n,:) = W;
            latents.P(n,:) = P;
        end
        
        if rpe > 0
            Q(c) = Q(c) + lr_pos*rpe;
            V = V + lr_pos*(rho*r - V);
        else
            Q(c) = Q(c) + lr_neg*rpe;
            V = V + lr_neg*(rho*r - V);
        end
        
        % update stickiness
        U = zeros(1,C);
        U(c) = 1;
        
    end