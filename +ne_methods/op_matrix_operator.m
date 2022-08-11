% Linear operator as a function handle

function out = op_matrix_operator(M,x,adjoint)

if adjoint; out = M.'*x; else; out = M*x; end

end