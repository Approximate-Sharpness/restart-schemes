% RE_RG_BEST_SYNC Renegar and Grimmer's best synchronous restart scheme.
%
%   See ne_NESTA_QCBP_Fourier_compare_RG.m for how to use this function,
%   in particular, how to define the functions 'initialize' and
%   'iteration' to return the appropriate structs.

function [state_ev_values, xout] = re_RG_best_sync(...
    initialize, iteration, f, x0, epsilon, T, eval_fns)

P = max(2,ceil(-log2(epsilon))); % Compute how many processes to use
ret = cell(P,T+1); % Record of minimum objective value seen a each iteration

state_ev_values = zeros(length(eval_fns), P*T);

eps = cell(P,1);    % Accuracy level used by each algorithm instance
states = cell(P,1); % Current state of each parallel algorithm instance
objs = cell(P,1);   % Target objective value of each algorithm instance

for j=1:P
    ret{j,1} = f(x0);
    eps{j} = epsilon*(2.0^(P-j));
    states{j} = initialize(x0, eps{j});
    objs{j} = ret{j,1} - eps{j}; % TODO: this might be buggy, be careful!
end

inbox = states{1};
inbox_obj = f(inbox.x);

minimizing_x = states{1}.x;

for i=1:T
    % Handle first process since it's special
    value = f(states{1}.x);
    state = states{1};

    if value < inbox_obj
        inbox = state;
        inbox_obj = value;
    else
        value = inbox_obj;
        state = inbox;
    end

    if value < objs{1} % if we complete our goal,
        objs{1} = value - eps{1};
    end

    ret{1,i+1} = f(states{1}.x); % min(ret{1,i}, f(states{1}.x)
    states{1} = iteration(states{1}); % do an iteration

    % Handle the middle processes
    for j=2:P-1
        value = f(states{j}.x);
        state = states{j};

        if value < inbox_obj
            inbox = state;
            inbox_obj = value;
        else
            state = inbox;
            value = inbox_obj;
        end

        if value < objs{j} % if we complete our current goal, restart self
            states{j} = initialize(state.x, eps{j});
            objs{j} = value - eps{j};
        end

        ret{j,i+1} = f(states{j}.x); % min(ret{j,i}, F(states{j}.x)
        states{j} = iteration(states{j}); % do an iteration
    end

    % handle last process since it's special
    value = f(states{P}.x);
    state = states{P};

    if value < inbox_obj
        inbox = state;
        inbox_obj = value;
    else
        state = inbox;
        value = inbox_obj;
    end

    if value < objs{P} % if we complete our current goal, restart self
        states{P} = initialize(state.x, eps{P});
        objs{P} = value - eps{P};
    end

    ret{P,i+1} = f(states{P}.x); % min(ret{P,i}, F(states{P}.x)
    states{P} = iteration(states{P}); % do an iteration

    % track eval_fns values on state iterates
    if ~isempty(eval_fns)
        for j=1:P
            if f(states{j}.x) < f(minimizing_x)
                minimizing_x = states{j}.x;
            end
            for fidx=1:length(eval_fns)
                state_ev_values(fidx,(i-1)*P+j) = eval_fns{fidx}(minimizing_x);
            end
        end
    end
end

% return evaluation values and final iterate
xout = states{P}.x;

end