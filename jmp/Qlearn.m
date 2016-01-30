function [lik, latents] = Qlearn(x,data)
    
    % Q-learning on multi-armed bandit with choice stickiness and separate learning rates for positive
    % and negative prediction errors.
    %
    % USAGE: [lik, data] = Qlearn2_sticky(x,data)
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
    %          in simulation mode, the following fields are required:
    %           .R - [1 x C] reward function for each choice option
    %           .C - number of choice options
    %           .N - number of trials
    %
    % OUTPUTS:
    %   lik - log-likelihood
    %   data - structure with the following fields:
    %           .v - [N x C] learned values
    %           .rpe - [N x 1] reward prediction error for chosen option
    %          in simulation mode, the following fields are created:
    %           .c - [N x 1] choices
    %           .r - [N x 1] rewards
    %
    % Sam Gershman, July 2015
    
    beta = x(1);
    lr_pos = x(2);
    lr_neg = x(3);
    kappa = x(4);
    pi = x(5);
    b = x(6);
    
    
    C = data.C;
    
    % fill in missing info
    if ~isfield(data,'block'); data.block = ones(data.N,1); end
    if ~isfield(data,'go'); data.go = zeros(data.N,1); end
    
    lik = 0;
    for n = 1:data.N
        
        % reset values and stickiness for new block or first trial
        if n == 1 || data.block(n)~=data.block(n-1)
            Q = ones(1,C)/C;    % initial action values
            V = 0;              % initial state value
            U = zeros(1,C);     % stickiness
        end
        
        c = data.c(n);
        r = data.r(n);
        go = data.go(n);
        data.v(n,:) = v;
        
        % compute policy and accumulate log-likelihod
        W = beta*Q + b*go + pi*V + kappa*U;                 % action weights
        P = (1-epsilon)*(exp(P)./sum(exp(P))) + epsilon/C;  % softmax + lapse
        lik = lik + log(P(c));
        
        % update values
        if r < 0; rho = rho_neg; else rho = rho_pos; end
        latents.rpe(n,1) = rho*r - Q(c);
        latents.Q(n,:) = Q;
        latents.V(n,1) = V;
        latents.W(n,:) = W;
        latents.P(n,:) = P;
        if rpe > 0
            Q(c) = Q(c) + lr_pos*(rho*r - Q(c));
            V = V + lr_pos*(rho*r - V);
        else
            Q(c) = Q(c) + lr_neg*(rho*r - Q(c));
            V = V + lr_neg*(rho*r - V);
        end
        
        % update stickiness
        U = zeros(1,C);
        U(c) = 1;
        
    end