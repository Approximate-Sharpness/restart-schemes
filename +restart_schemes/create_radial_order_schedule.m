function phi = create_radial_order_schedule(t, flags)

% - count solutions of n1*n2*... <= tau to generate array of fixed size
% - maximum number of ni variables is 4, depending on flags
count = 0;
tau = 1;

% limit exponents on sharpness grid search
alpha_lim = 32;
beta_lim = 32;
eta_lim = 32;

while count < t
    count = 0;
    if flags(1); n1end = min(alpha_lim,tau); else; n1end = 1; end
    if flags(2); n2end = min(beta_lim,tau); else; n2end = 1; end
    if flags(3); n3end = min(eta_lim,tau); else; n3end = 1; end
    for n1 = 1:n1end
        for n2 = 1:n2end
            if n1*n2 > tau
                break
            end
            for n3 = 1:n3end
                if n1*n2*n3 > tau
                    break
                end
                for n4 = 1:tau
                    if (n1*n2*n3*n4 > tau)
                        break
                    else
                        count = count+1;
                        if n1 > 1
                            count = count+1;
                        end
                        if n3 > 1
                            count = count+1;
                        end
                        if (n1 > 1) && (n3 > 1)
                            count = count+1;
                        end
                    end
                end
            end
        end
    end
    tau = tau+1;
end

tau = tau-1; % correct off-by-one

% now generate the solutions as in the radial ordering proof
sols = zeros(count,4);
idx = 1;

if flags(1); n1end = min(alpha_lim,tau); else; n1end = 1; end
if flags(2); n2end = min(beta_lim,tau); else; n2end = 1; end
if flags(3); n3end = min(eta_lim,tau); else; n3end = 1; end

for n1 = 1:n1end
    for n2 = 1:n2end
        if n1*n2 > tau
            break
        end
        for n3 = 1:n3end
            if n1*n2*n3 > tau 
                break
            end
            for n4 = 1:tau
                if n1*n2*n3*n4 > tau
                    break
                else
                    sols(idx,:) = [n1-1,n2-1,n3-1,n4];
                    idx = idx+1;
                    if n1 > 1
                        sols(idx,:) = [1-n1,n2-1,n3-1,n4];
                        idx = idx+1;
                    end
                    if n3 > 1
                        sols(idx,:) = [n1-1,n2-1,1-n3,n4];
                        idx = idx+1;
                    end
                    if (n1 > 1) && (n3 > 1)
                        sols(idx,:) = [1-n1,n2-1,1-n3,n4];
                        idx = idx+1;
                    end
                end
            end
        end
    end
end

% finally, sort the solutions according to the values of schedule h
% this ensures phi is an h-schedule (h <-> radial ordering)
schedule_value = (abs(sols(:,1))+1).*(sols(:,2)+1).*(abs(sols(:,3))+1).*sols(:,4);
[~,idxperm] = sort(schedule_value);
sols = sols(idxperm,:);

phi = sols;
    
end