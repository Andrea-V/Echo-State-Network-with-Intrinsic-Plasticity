%% INITIALISE THE ECHO STATE NETWORK
function [W_in, W_hat] = echo_state_network(ni, nr, rho, scaling, connectivity)
    % create W_in e W_hat
    W_in = scaling * 2 * rand(nr, ni + 1) - 1;
    
    % we do not want 0 as max eigenvalue
    max_eigenvalue = 0;
    while max_eigenvalue == 0
        W_hat = 2 * rand(nr, nr) - 1;
        W_hat(rand(nr, nr) > connectivity) = 0;
        % adjusting W_hat for ESP with leaky parameter
        % W_hat = (1-leaky) * eye(nr) + leaky * W_hat;
        max_eigenvalue = max(abs(eig(W_hat)));
    end
    
    % echo state property
    W_hat = W_hat * (rho / max_eigenvalue ); % necessary condition
    % W_hat = W_hat * (rho / norm(W_hat) ); % sufficient condition
end