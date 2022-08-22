% Helper method to extract restart values from cells
%
%   The inputs ev, ii correspond to the second and third outputs of the
%   re_radial_search function, respectively. The input num_fn_handles is a
%   positive integer corresponding to length(eval_fns), where eval_fns
%   are one of the optional parameters for re_radial_search.
%
%   The output is given as
%     
%     values    - an array with num_fn_handles rows, and number of columns 
%                 equal to the number of nontrivial restarts performed; the 
%                 entries are precisely the evaluations of each (nontrivial) 
%                 restart iterate using functions defined in eval_fns
%     iter_idxs - a vector with number of entries equal to the number of
%                 nontrivial restarts performed; the entries are cumulative
%                 number of total iterations performed at the end of that
%                 restart indexed by the column

function [values, iter_idxs] = h_extract_re_cell_data(ev, ii, num_fn_handles)

t = length(ev);
not_empty = @(v) ~isempty(v);
ev_exact_len = sum(cellfun(not_empty, ev));
ii_exact_len = sum(cellfun(not_empty, ii));

assert(ev_exact_len == ii_exact_len);

values = zeros(num_fn_handles,ev_exact_len);
iter_idxs = zeros(1,ii_exact_len);

k = 1;
for j=1:t
    if ~isempty(ii{j})
        values(:,k) = ev{j};
        iter_idxs(k) = ii{j};
        k = k+1;
    end
end

iter_idxs = cumsum(iter_idxs);

end
