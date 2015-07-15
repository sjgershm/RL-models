function [lik, data] = Qlearn1(x,data)
    
    % Q-learning on multi-armed bandit with single learning rate.
    %
    % USAGE: [lik, data] = Qlearn1(x,data)
    %
    % INPUTS:
    %   x - parameters:
    %       x(1) - inverse temperature
    %       x(2) - learning rate
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
    
    if isfield(data,'R') == 1 % simulation mode
        C = data.C;
        v = zeros(1,C);  % initial values
        lik = 0;
        for n = 1:data.N
            p = exp(b*v - logsumexp(b*v,2));
            c = fastrandsample(p);
            lik = lik + log(p(c));
            r = data.R(c);
            rpe = r-v(c);           % reward prediction error
            v(c) = v(c) + lr*rpe;   % update values
            data.c(n,1) = c;
            data.r(n,1) = r;
            data.v(n,:) = v;
            data.rpe(n,1) = rpe;
        end
    else                 % likelihood mode
        C = max(unique(data.c)); % number of options
        v = zeros(1,C);  % initial values
        lik = 0;
        for n = 1:data.N
            c = data.c(n);
            r = data.r(n);
            lik = lik + b*v(c) - logsumexp(b*v,2);
            rpe = r-v(c);
            v(c) = v(c) + lr*rpe;      % update values
            data.v(n,:) = v;
            data.rpe(n,1) = rpe;
        end
    end