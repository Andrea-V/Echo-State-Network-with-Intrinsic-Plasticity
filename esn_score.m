%% SCORE ESN OVER A TEST SET
function [ output, error ] = esn_score(X, y, W_in, W_hat, W_out, a, b, ntransient)
    [ states ] = esn_states(X, W_in, W_hat, a, b);
    % discard transient
    states = states(:, ntransient:end);     
    % add bias
    states = [ states; ones(1, size(states, 2)) ];
    y = y(:, ntransient:end); 

    output = W_out * states;
    
    % metric: normalized RMSE
    error = zeros(size(y, 1), 1);
    for i = 1:size(y, 1)
        error(i) = sqrt( immse(output(i, :), y(i, :)) / var(y(i, :)) );
    end
end

