% Helper method to extract restart values from cells

function out = concatenate_cell_entries(c)

cell_len = length(c);
total_len = 0;

for i=1:cell_len
    total_len = total_len + length(c{i});
end

out = zeros(total_len,1);

k = 1;
for i=1:cell_len
    for j=1:length(c{i})
        out(k) = c{i}{j};
        k = k + 1;
    end
end

end
