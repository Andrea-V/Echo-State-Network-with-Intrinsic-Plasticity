%% TRAIN READOUT OF ESN
function [ W_out, output, error ] = esn_train(X, y, W_in, W_hat, lambda, a, b, ntransient)
    states = esn_states(X, W_in, W_hat, a, b);

    % discard transient
    states = states(:, ntransient:end);     
    y = y(:, ntransient:end); 
    % add bias
    states = [ states; ones(1, size(states, 2)) ];

    % train readout and compute error
    W_out = (y * states') / (states * states'  + lambda * eye(size(W_hat, 1) + 1)); % ridge regression     
    output = W_out * states;
    
    % metric: normalized RMSE
    error = zeros(size(y, 1), 1);
    for i = 1:size(y, 1)
        error(i) = sqrt( immse(output(i, :), y(i, :)) / var(y(i, :)) );
    end
end

