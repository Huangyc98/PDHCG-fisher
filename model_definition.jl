mutable struct Restartinfo
    last_res_primal::CuArray{Float64}
    last_res_dual::CuArray{Float64}
    res_primal_distence::Float64
    res_dual_distence::Float64
    
end

mutable struct cuRestartinfo
    last_res_primal::CUDA.CUSPARSE.CuSparseMatrixCSR{Float64, Int32}
    last_res_dual::CuArray{Float64}
    res_primal_distence::Float64
    res_dual_distence::Float64
    last_res_residual::Float64
    last_residual::Float64
end

mutable struct Fisherproblem
    w::Vector{Float64}
    u::SparseMatrixCSC{Float64,Int64}
    beq::Vector{Float64}
    ori_m::Int64
    ori_n::Int64
    x0::SparseMatrixCSC{Float64,Int64}
end

struct cuFisherproblem
    w::CuArray{Float64}          
    u::CUDA.CUSPARSE.CuSparseMatrixCSR{Float64, Int32}  
    beq::CuArray{Float64}       
    m::Int64                       
    n::Int64
    x0::CUDA.CUSPARSE.CuSparseMatrixCSR{Float64, Int32}    
end

mutable struct cuPdhgSolverState
    current_primal_solution::CUDA.CUSPARSE.CuSparseMatrixCSR{Float64, Int32}
    current_primal_sum::CuArray{Float64}
    current_dual_solution::CuArray{Float64}
    prev_primal_solution::CUDA.CUSPARSE.CuSparseMatrixCSR{Float64, Int32}
    prev_primal_sum::CuArray{Float64}
    prev_dual_solution::CuArray{Float64}
    avg_primal_solution::CUDA.CUSPARSE.CuSparseMatrixCSR{Float64, Int32}
    avg_primal_sum::CuArray{Float64}
    avg_dual_solution::CuArray{Float64}
    step_size::Float64
    primal_weight::Float64
    total_number_iterations::Int64
    numerical_error::Bool
    inner_iteration::Int64
end

mutable struct PdhgSolverState
    current_primal_solution::Vector{Float64}
    current_dual_solution::Vector{Float64}
    prev_primal_solution::Vector{Float64}
    prev_dual_solution::Vector{Float64}
    avg_primal_solution::Vector{Float64}
    avg_dual_solution::Vector{Float64}
    avg_dual_product::Vector{Float64}
    current_dual_product::Vector{Float64}
    step_size::Float64
    primal_weight::Float64
    total_number_iterations::Int64
    numerical_error::Bool
    inner_iteration::Int64
end

mutable struct Bufferstate
    primal_solution::Vector{Float64}
    dual_solution::Vector{Float64}
    dual_product::Vector{Float64}
    step_size::Float64
    primal_weight::Float64
end

mutable struct cuBufferstate
    primal_solution::CUDA.CUSPARSE.CuSparseMatrixCSR{Float64, Int32}
    primal_sum::CuArray{Float64}
    dual_solution::CuArray{Float64}
    step_size::Float64
    primal_weight::Float64
    extra_ux::CUDA.CUSPARSE.CuSparseMatrixCSR{Float64, Int32}
    p_array::CuArray{Float64}
    new_p_array::CuArray{Float64}
    Upper_bound::CuArray{Float64}
    Lower_bound::CuArray{Float64}
    residual::Float64
end

mutable struct residual
    current_primal_residual::Float64
    current_dual_residual::Float64
    current_gap::Float64
    avg_primal_residual::Float64
    avg_dual_residual::Float64
    avg_gap::Float64
end