function P = policy(Q,beta,epsilon)
    
    % Map Q-values to action policy.
    %
    % USAGE: p = policy(Q,beta,epsilon)
    %
    % INPUTS:
    %   Q - [1 x C] vector of Q-values
    %   beta - inverse temperature
    %   epsilon - lapse rate
    %
    % OUTPUTS:
    %   P - [1 x C] vector of action probabilities
    %
    % Sam Gershman, Nov 2015
    
    C = length(Q);
    P = exp(Q*beta);
    P = (1-epsilon)*(P./sum(P)) + epsilon/C;