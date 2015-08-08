function [lik, data] = Qlearn1_sticky(x,data)
    
    % Q-learning on multi-armed bandit with single learning rate and choice stickiness.
    %
    % USAGE: [lik, data] = Qlearn1_sticky(x,data)
    %
    % INPUTS:
    %   x - parameters:
    %       x(1) - inverse temperature
    %       x(2) - learning rate
    %       x(3) - stickiness inverse temperature
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
    
    b = x(1);
    lr = x(2);
    bs = x(3);
    
    if isfield(data,'R') == 1 % simulation mode
        C = data.C;
        v = zeros(1,C);  % initial values
        u = zeros(1,C);  % stickiness
        lik = 0;
        for n = 1:data.N
            data.v(n,:) = v;
            q = b*v + bs*u;
            p = exp(q - logsumexp(q,2));
            c = fastrandsample(p);
            lik = lik + log(p(c));
            r = data.R(c);
            rpe = r-v(c);           % reward prediction error
            v(c) = v(c) + lr*rpe;   % update values
            u = zeros(1,C); u(c) = 1;    % update stickiness
            data.c(n,1) = c;
            data.r(n,1) = r;
            data.rpe(n,1) = rpe;
        end
    else                 % likelihood mode
        C = max(unique(data.c)); % number of options
        v = zeros(1,C);  % initial values
        u = zeros(1,C);  % stickiness
        lik = 0;
        for n = 1:data.N
            c = data.c(n);
            r = data.r(n);
            data.v(n,:) = v;
            q = b*v + bs*u;
            lik = lik + q(c) - logsumexp(q,2);
            rpe = r-v(c);
            v(c) = v(c) + lr*rpe;   % update values
            u = zeros(1,C); u(c) = 1;    % update stickiness
            data.rpe(n,1) = rpe;
        end
    end