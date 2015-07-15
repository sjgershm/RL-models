function [lik, data] = Qlearn2_sticky(x,data)
    
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
    %       x(5) - stickiness decay
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
    lr_pos = x(2);
    lr_neg = x(3);
    bs = x(4);
    bd = x(5);
    
    if isfield(data,'R') == 1 % simulation mode
        C = data.C;
        v = zeros(1,C);  % initial values
        u = zeros(1,C);  % stickiness
        lik = 0;
        for n = 1:data.N
            q = b*v + bs*u;
            p = exp(q - logsumexp(q,2));
            c = fastrandsample(p);
            lik = lik + log(p(c));
            r = data.R(c);
            rpe = r-v(c);           % reward prediction error
            if rpe > 0
                v(c) = v(c) + lr_pos*rpe;   % update values
            else
                v(c) = v(c) + lr_neg*rpe;   % update values
            end
            u = u.*bd; u(c) = 1;    % update stickiness
            data.c(n,1) = c;
            data.r(n,1) = r;
            data.v(n,:) = v;
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
            q = b*v + bs*u;
            lik = lik + q(c) - logsumexp(q,2);
            rpe = r-v(c);
            if rpe > 0
                v(c) = v(c) + lr_pos*rpe;   % update values
            else
                v(c) = v(c) + lr_neg*rpe;   % update values
            end
            u = u.*bd; u(c) = 1;    % update stickiness
            data.v(n,:) = v;
            data.rpe(n,1) = rpe;
        end
    end