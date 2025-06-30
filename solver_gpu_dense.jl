
function generate_problem(m, n)
    w = rand(m)   
    u = sprand(m, n,0.2)   
    println("Number of nonzeros in u: ", nnz(u))

    x0 = similar(u)   
    x0 .=u
    b = 0.25*m*ones(n)
    vec = b' ./ sum(u,dims=1)
    x0 = u.*vec

    problem = Fisherproblem(
        w,u,b,m,n,x0,
    )
    return problem
end


function fisher_cpu_to_gpu(problem::Fisherproblem)
    
    d_w = CuArray{Float64}(undef, length(problem.w))
    d_beq = CuArray{Float64}(undef, length(problem.beq))
    
    
    copyto!(d_w, problem.w)
    copyto!(d_beq,problem.beq)
    
    d_x0 = CUDA.CUSPARSE.CuSparseMatrixCSR(problem.x0)
    d_u =  CUDA.CUSPARSE.CuSparseMatrixCSR(problem.u)

    return cuFisherproblem(d_w,d_u,d_beq,problem.ori_m,problem.ori_n,d_x0)
end

    function take_step_exact_matrix_gpu!(
        problem::cuFisherproblem,
        solverstate::cuPdhgSolverState,
        bufferstate::cuBufferstate, 
        initial_sttepsize::Float64,    
        )

        step_size = solverstate.step_size
        m = problem.m
        n = problem.n
        done = false
        k = 1
        while !done
            k += 1
            if k>=20
                break
            end
            tau = solverstate.step_size /solverstate.primal_weight
            sigma = solverstate.step_size * solverstate.primal_weight

            primal_exact_total_in_gpu!(problem.w, problem.u, solverstate.current_primal_solution, solverstate.current_dual_solution,tau, problem.m, problem.n,
            bufferstate.primal_solution, bufferstate.extra_ux,bufferstate.new_p_array,bufferstate.p_array,bufferstate.Upper_bound,bufferstate.Lower_bound,bufferstate.residual)

            CSR_sum(bufferstate.primal_solution,bufferstate.primal_sum)

            bufferstate.dual_solution .= solverstate.current_dual_solution - sigma .*((2 .*bufferstate.primal_sum - solverstate.current_primal_sum)' -problem.beq)

            interaction, movement = compute_interaction_and_movement_gpu(
            solverstate,
            bufferstate,
            )

            if interaction > 0
                step_size_limit = movement / interaction
                if movement == 0.0
                    solverstate.numerical_error = true
                    break
                end
            else
                step_size_limit = Inf
            end
        
                   if step_size <= step_size_limit
                update_solution_in_solver_state_gpu!(
                    solverstate, 
                    bufferstate,
                )
                done = true
                   end
                first_term = (1 - 1/(solverstate.inner_iteration + 1)^(0.3)) * step_size_limit
                second_term = (1 + 1/(solverstate.inner_iteration + 1)^(0.6)) * step_size
                step_size = min(first_term, second_term)
                step_size = min(max(step_size,0.01*initial_sttepsize),3.0*initial_sttepsize)

        end
              solverstate.step_size = step_size
        return 
    end

    function primal_exact_total_in_gpu!(
        w::CuVector{Float64},
        u::CUDA.CUSPARSE.CuSparseMatrixCSR{Float64, Int32},
        x_init::CUDA.CUSPARSE.CuSparseMatrixCSR{Float64, Int32},
        y::CuArray{Float64},
        tau::Float64,
        m::Int64,
        n::Int64,
        x_matrix::CUDA.CUSPARSE.CuSparseMatrixCSR{Float64, Int32},
        ux::CUDA.CUSPARSE.CuSparseMatrixCSR{Float64, Int32},
        new_p_array::CuArray{Float64},
        p::CuArray{Float64},
        U::CuArray{Float64},
        L::CuArray{Float64},
        residual::Float64 
        )
        
        iteration = CUDA.zeros(Int64, m)
        tol =1e-9  

        rowptr = u.rowPtr  
        colind = u.colVal  
        nzval = u.nzVal   
        ux.nzVal .= u.nzVal.*x_matrix.nzVal
        p .= CUDA.sum(ux, dims=2)
        size_row = length(rowptr) - 1        
        threads_per_block = 32 
        num_blocks = size_row
        @cuda threads=threads_per_block blocks=num_blocks primal_update_in_cuda(rowptr, colind, size_row, nzval, tau, w, p, x_init.nzVal, y,x_matrix.nzVal,new_p_array,U,L,tol,iteration)

        return x_matrix
    end

function compute_interaction_and_movement_gpu(
    solverstate::cuPdhgSolverState,
    bufferstate::cuBufferstate,
)

    delta_primal = similar(solverstate.current_primal_solution)
    CUDA.sum(solverstate.current_primal_solution, dims=2)
    delta_primal.nzVal .= bufferstate.primal_solution.nzVal .- solverstate.current_primal_solution.nzVal
    delta_dual = bufferstate.dual_solution .- solverstate.current_dual_solution
    sum_delta_primal = bufferstate.primal_sum - solverstate.current_primal_sum
    primal_dual_interaction = CUDA.dot(sum_delta_primal, delta_dual)
    interaction = abs(primal_dual_interaction)
    norm_delta_primal = sqrt(CUDA.dot(delta_primal.nzVal,delta_primal.nzVal))
    norm_delta_dual = sqrt(CUDA.dot(delta_dual,delta_dual))

    movement = 0.5*(solverstate.primal_weight * norm_delta_primal^2 + (1 / solverstate.primal_weight) * norm_delta_dual^2)
    return interaction, movement
end

function update_solution_in_solver_state_gpu!(
    solverstate::cuPdhgSolverState,
    bufferstate::cuBufferstate,
)
    solverstate.current_primal_sum .= bufferstate.primal_sum
    solverstate.current_primal_solution.nzVal .= bufferstate.primal_solution.nzVal
    solverstate.current_dual_solution .= bufferstate.dual_solution

end

function compute_new_primal_weight(
    last_restart_info::cuRestartinfo,
    primal_weight::Float64,
    m::Int64,
    primal_weight_update_smoothing::Float64=0.2,
)
    primal_distance = last_restart_info.res_primal_distence
    dual_distance = last_restart_info.res_dual_distence
    if primal_distance > eps() && dual_distance > eps()
        new_primal_weight_estimate = dual_distance / (primal_distance)
        log_primal_weight =
            primal_weight_update_smoothing * log(new_primal_weight_estimate) +
            (1 - primal_weight_update_smoothing) * log(primal_weight)
        primal_weight = exp(log_primal_weight)

        return primal_weight
    else
        return primal_weight
    end
end

function compute_residual_matrix_gpu(
    solverstate::cuPdhgSolverState,
    problem::cuFisherproblem,
    )
    
    current_primal_residual = CUDA.norm(solverstate.current_primal_sum' .- problem.beq,Inf)/(1+max(CUDA.norm(solverstate.current_primal_sum,Inf),CUDA.norm(problem.beq,Inf)))
    current_primal_grad, current_primal_norm, current_dual_norm = compute_grad_gpu_with_norm(solverstate.current_primal_solution,problem.u,solverstate.current_dual_solution,problem.w)
    current_dual_residual = CUDA.norm(min.(current_primal_grad,0),Inf)/(1+max(current_primal_norm,current_dual_norm))
    current_primal_grad_positive = max.(current_primal_grad,0.0)
    current_gap = CUDA.norm(current_primal_grad_positive.*solverstate.current_primal_solution,Inf)/(1 + max(CUDA.norm(current_primal_grad_positive,Inf),CUDA.norm(solverstate.current_primal_solution,Inf)))

    avg_primal_residual = CUDA.norm(solverstate.avg_primal_sum' .- problem.beq,Inf)/(1+max(CUDA.norm(solverstate.avg_primal_sum,Inf),CUDA.norm(problem.beq,Inf)))
    avg_primal_grad,avg_primal_norm, avg_dual_norm = compute_grad_gpu_with_norm(solverstate.avg_primal_solution,problem.u,solverstate.avg_dual_solution,problem.w)
    avg_dual_residual = CUDA.norm(min.(avg_primal_grad,0),Inf)/(1+max(avg_primal_norm,avg_dual_norm))
    avg_primal_grad_positive = max.(avg_primal_grad,0.0)
    avg_gap = CUDA.norm(avg_primal_grad_positive.*solverstate.avg_primal_solution,Inf)/(1 + max(CUDA.norm(avg_primal_grad_positive,Inf),CUDA.norm(solverstate.avg_primal_solution,Inf)))

    return residual(current_primal_residual,current_dual_residual,current_gap,avg_primal_residual,avg_dual_residual,avg_gap

    )
end

function compute_grad_gpu(
    x::CUDA.CUSPARSE.CuSparseMatrixCSR{Float64, Int32},
    u::CUDA.CUSPARSE.CuSparseMatrixCSR{Float64, Int32},
    y::CuVector{Float64},
    w::CuVector{Float64},
    )
    grad_matrix = -w./sum(u .* x, dims=2).*u .- y'
    return grad_matrix
end

function compute_grad_gpu_with_norm(
    x::CUDA.CUSPARSE.CuSparseMatrixCSR{Float64, Int32},
    u::CUDA.CUSPARSE.CuSparseMatrixCSR{Float64, Int32},
    y::CuVector{Float64},
    w::CuVector{Float64},
    )
    m,n = size(u)
    rowptr = u.rowPtr  
    colind = u.colVal  
    nzval = u.nzVal   
    size_row = length(rowptr) - 1        
    threads_per_block = 256
    num_blocks = cld(size_row, threads_per_block)
    grad_matrix = similar(u)
    grad_matrix2 = similar(u)
    vec = -w./sum(u .* x, dims=2)
    @cuda threads=threads_per_block blocks=num_blocks vec_u_y_in_cuda(rowptr, colind,size_row, u.nzVal, vec, y,grad_matrix.nzVal,grad_matrix2.nzVal)
    
    primal_norm = CUDA.norm(grad_matrix2,Inf)
    dual_norm = CUDA.norm(y,Inf)
    return grad_matrix, primal_norm,dual_norm
end

function vec_u_y_in_cuda(rowptr, colind,size_row, u, vec, y,new_value1,new_value2)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if i <= size_row
        start_idx = rowptr[i]
        end_idx = rowptr[i + 1] - 1
        for j = start_idx:end_idx
            new_value1[j] = vec[i]* u[j] - y[colind[j]]
            new_value2[j] = vec[i]* u[j]
        end
    end
    return
end

function CSR_dot_x_matrix_in_cuda(rowptr, colind,size_row, u, tau, w, p, x_matrix, y,new_value)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if i <= size_row
        start_idx = rowptr[i]
        end_idx = rowptr[i + 1] - 1

        for j = start_idx:end_idx
            new_value[j] = max(tau*w[i]/p[i]* u[j] + x_matrix[j] + tau*y[colind[j]],0)
        end
    end
    return
end

function primal_update_in_cuda(rowptr, colind, size_row, u, tau, w, p,
                                            x_matrix, y, x_matrix_new_value,new_p,
                                            U, L, tol, iteration)
      i = blockIdx().x  
      candidate_idx = threadIdx().x  
      if i <= size_row
          start_idx = rowptr[i]
          end_idx = rowptr[i + 1] - 1
          shared_U = @cuStaticSharedMem(Float64,32)
          shared_L = @cuStaticSharedMem(Float64,32)
          if candidate_idx == 1

              local_new_p = 0.0
              tau_w_i_over_p = tau * w[i] / max(p[i],1e-15)
              for j = start_idx:end_idx
                  x_matrix_new_value[j] = max(tau_w_i_over_p * u[j] + x_matrix[j] + tau * y[colind[j]], 0.0)
                  local_new_p += u[j] * x_matrix_new_value[j]
              end
              new_p[i] = local_new_p
              U[i] = max(p[i], local_new_p)
              L[i] = min(p[i], local_new_p)
          end
          sync_threads()
          while abs(new_p[i] - p[i]) >= tol && iteration[i] <= 200
              if candidate_idx == 1
                  iteration[i] += 1
              end
              sync_threads()
              p[i] = new_p[i]
              L_val = L[i]
              U_val = U[i]
              p_candidate = max(L_val + (candidate_idx / 32.0) * (U_val - L_val), 1e-15)
              tau_w_i_over_candidate = tau * w[i] / p_candidate
              new_p_candidate = 0.0
              for j = start_idx:end_idx
                  temp = max(tau_w_i_over_candidate * u[j] + x_matrix[j] + tau * y[colind[j]], 0.0)
                  new_p_candidate += u[j] * temp
              end
              shared_U[candidate_idx] = max(p_candidate, new_p_candidate)
              shared_L[candidate_idx] = min(p_candidate, new_p_candidate)
              sync_threads()
              stride = 16
              while stride > 0
                  if candidate_idx <= stride
                      shared_U[candidate_idx] = min(shared_U[candidate_idx], shared_U[candidate_idx + stride])
                      shared_L[candidate_idx] = max(shared_L[candidate_idx], shared_L[candidate_idx + stride])
                  end
                  sync_threads()
                  stride ÷= 2
              end
                  U[i] = shared_U[1]
                  L[i] = shared_L[1]
                  new_p[i] = (U[i]+L[i])/2
              sync_threads()
          end
                tau_w_i_over_p = tau * w[i] / max(new_p[i],1e-15)
                for j = start_idx:end_idx
                    x_matrix_new_value[j] = max(tau_w_i_over_p * u[j] + x_matrix[j] + tau * y[colind[j]], 0)
                end
      end
      return
end

function CSR_SUM_kernel!(values::CuDeviceVector{Float64}, 
                    indices::CuDeviceVector{Int32},  # 改为Int32
                    output::CuDeviceMatrix{Float64, 1},
                    N::Int)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= N
        CUDA.@atomic output[1, indices[i]] += values[i]
    end
    return nothing
end

function CSR_sum(A_csr,csr_sum)

    csr_sum .= 0.0
    N = length(A_csr.nzVal)
    threads = 256
    blocks = cld(N, threads)
    CUDA.@sync @cuda threads=threads blocks=blocks CSR_SUM_kernel!(A_csr.nzVal, A_csr.colVal, csr_sum, N)

end
