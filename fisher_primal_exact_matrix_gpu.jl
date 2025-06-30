# GPU_id = 1
# ENV["CUDA_VISIBLE_DEVICES"] = "$GPU_id"

using LinearAlgebra
using SparseArrays
using Printf
using Random
using JuMP
using Ipopt
using Plots
using Serialization
import CUDA
using CUDA
using XLSX

include("model_definition.jl")
include("solver_gpu_dense.jl")

function main()
results = []
    for i in 1:1
        println("\nRun Experiment #$i")
m = 10000   
n = 400   
Random.seed!(i)

record_xlsx = true
problem = generate_problem(m, n)
d_problem = fisher_cpu_to_gpu(problem)
println("Problem generated successfully.")

println("\nRunning PDHG optimization...")
    pdhg_time = @elapsed begin
        total_iterations = PDHG_gpu_adaptive_restart(d_problem)
end
println("PDHG optimization completed.")

println("PDHG runtime (seconds): ", pdhg_time)
println("Total iterations: ", total_iterations)

GC.gc()
CUDA.reclaim()
push!(results, (total_iterations, pdhg_time))
if record_xlsx
XLSX.openxlsx("fisher_pdhcg_$(n)_$(m).xlsx", mode="w") do xf
    sheet = xf[1]
    sheet["A1"] = "Iteration Count"
    sheet["B1"] = "Time (s)"
    for (i, (PDHG_iteration, pdhg_time)) in enumerate(results)
        sheet["A$(i+1)"] = PDHG_iteration
        sheet["B$(i+1)"] = pdhg_time
    end
end
end
end

end
function PDHG_gpu_adaptive_restart(d_problem::cuFisherproblem, iteration_limit=20000000)
    #问题输入
    m = d_problem.m
    n = d_problem.n
    w = d_problem.w
    x0 = d_problem.x0

    primal_size = m
    dual_size = n
    initial_stepsize = 2.0/sqrt(m)
    initial_primal_dual_weight = sqrt(n/m)

    sum_x = CUDA.sum(x0,dims=1)
    solverstate = cuPdhgSolverState(
        1.0.*x0,
        1.0.*sum_x,
        CUDA.zeros(Float64, dual_size),
        0.5.*x0,
        0.5.*sum_x,
        CUDA.zeros(Float64, dual_size),
        1.0.*x0,
        1.0.*sum_x,
        CUDA.zeros(Float64, dual_size),
        initial_stepsize,
        initial_primal_dual_weight,
        1,
        false,
        1,
    )

    bufferstate = cuBufferstate(
        1.0.*x0,
        1.0.*sum_x,
        CUDA.zeros(Float64, dual_size),
        initial_stepsize,
        initial_primal_dual_weight,
        1.0.*x0,
        CUDA.zeros(Float64, primal_size),
        CUDA.zeros(Float64, primal_size),
        CUDA.zeros(Float64, primal_size),
        CUDA.zeros(Float64, primal_size),
        1.0,
    )
    Restart_info = cuRestartinfo(
        1.0.*x0,
        CUDA.zeros(Float64,dual_size),
        0.0,
        0.0,
        10.0,
        10.0,
    )
    Residual = residual(
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
    )

    iteration = 0
    println("initialize success")
    restart_count = 0

    while true
        iteration +=1

        if mod(iteration,120)==0||iteration == iteration_limit + 1 || solverstate.numerical_error

            Residual = compute_residual_matrix_gpu(solverstate,d_problem)
            current_residual = max(Residual.current_primal_residual,Residual.current_dual_residual, Residual.current_gap)
            avg_residual = max(Residual.avg_primal_residual,Residual.avg_dual_residual, Residual.avg_gap)
            primal_residual = min(current_residual,avg_residual)
            bufferstate.residual = primal_residual

            better_residual = min(current_residual,avg_residual)
            if better_residual <=0.2*Restart_info.last_res_residual
                Restart_Flag = true
            else
                if better_residual <=0.8*Restart_info.last_res_residual && better_residual>=Restart_info.last_residual
                Restart_Flag = true
                else
                    if solverstate.inner_iteration>=0.2*solverstate.total_number_iterations
                    Restart_Flag = true
                    else
                        Restart_Flag = false
                    end
                end
            end

            Restart_info.last_residual=better_residual

            if Restart_Flag && solverstate.total_number_iterations>=320
                restart_count += 1
                if current_residual<= avg_residual
                    solverstate.avg_dual_solution .= solverstate.current_dual_solution
                    solverstate.avg_primal_solution.nzVal .= solverstate.current_primal_solution.nzVal
                    Restart_info.last_res_residual = current_residual
                    bufferstate.primal_solution.nzVal .= solverstate.current_primal_solution.nzVal
                    bufferstate.dual_solution .= solverstate.current_dual_solution

                else
                    solverstate.current_dual_solution .=solverstate.avg_dual_solution
                    solverstate.current_primal_solution.nzVal .= solverstate.avg_primal_solution.nzVal
                    Restart_info.last_res_residual = avg_residual
                    bufferstate.primal_solution.nzVal .= solverstate.avg_primal_solution.nzVal
                    bufferstate.dual_solution .= solverstate.avg_dual_solution
                end
                solverstate.inner_iteration = 1

                Restart_info.res_primal_distence = CUDA.norm(Restart_info.last_res_primal.nzVal - solverstate.current_primal_solution.nzVal)
                Restart_info.res_dual_distence = CUDA.norm(Restart_info.last_res_dual - solverstate.current_dual_solution)
                Restart_info.last_res_primal.nzVal .= solverstate.current_primal_solution.nzVal
                Restart_info.last_res_dual .= solverstate.current_dual_solution
                
                
                solverstate.primal_weight  = compute_new_primal_weight(Restart_info,solverstate.primal_weight,d_problem.m)
                if restart_count % 3 == 0
                    if solverstate.primal_weight < 1e-5 * initial_primal_dual_weight || solverstate.primal_weight > 1e5 * initial_primal_dual_weight
                       solverstate.primal_weight = initial_primal_dual_weight
                    end
                end
            end
            @printf("Iteration %d, current_primal_residual: %e, current_dual_residual: %e, current_complementary: %e\n,avg_primal_residual: %e, avg_dual_residual: %e, avg_complementary: %e\n,stepsize: %e, primal_dual_weight: %e\n", 
            iteration,
            Residual.current_primal_residual,Residual.current_dual_residual, Residual.current_gap,
            Residual.avg_primal_residual,Residual.avg_dual_residual, Residual.avg_gap,solverstate.step_size,
            solverstate.primal_weight)
            if primal_residual<=1e-4 || iteration >= iteration_limit+1
                break
            end
        end

        take_step_exact_matrix_gpu!(d_problem,solverstate,bufferstate,initial_stepsize)
        weight = 1 / (1.0 + solverstate.inner_iteration)

        solverstate.avg_primal_solution.nzVal .+=  weight*(solverstate.current_primal_solution.nzVal .- solverstate.avg_primal_solution.nzVal)
        solverstate.avg_dual_solution .+= weight*(solverstate.current_dual_solution.-solverstate.avg_dual_solution)
        solverstate.avg_primal_sum .+= weight*(solverstate.current_primal_sum .-solverstate.avg_primal_sum)
        solverstate.inner_iteration += 1
        solverstate.total_number_iterations +=1

    end
return solverstate.total_number_iterations
end

main()