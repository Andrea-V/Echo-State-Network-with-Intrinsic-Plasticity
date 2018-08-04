%% HARVEST ECHO STATES FROM RESERVOIR
function [ states ] = esn_states(X, W_in, W_hat, a, b)
    % X: samples on the columns
    % computing reservoir states
    states = [];
    %state =  zeros(size(W_hat, 1), size(X, 2));
    
    for j = 1:size(X, 2)
        if j == 1
            state = W_in * X(:, j);
        else
            state = W_in * X(:, j) + W_hat * states(:, j-1);
            % LI-ESN variant
            % state = (1-leaky) * states(:, j-1) + leaky * tanh(W_in * X(:, j) + W_hat * states(:, j-1));
        end
        
        state = tanh(a .* state + b);
        states = [states state];
    end    
end

