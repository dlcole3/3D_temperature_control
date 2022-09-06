using DynamicNLPModels, NLPModels, Random, LinearAlgebra
using DelimitedFiles, SparseArrays, NLPModelsIpopt
using QuadraticModels, DataFrames, JLD, Printf

include("PDE_boundary_3d_heating.jl")

function time_PDE(lqdm_list, kkt_system, algorithm, device)
    lens = length(lqdm_list)

    t       = zeros(lens)
    f       = zeros(lens)
    iters   = zeros(lens)
    status  = []
    ips_tt  = zeros(lens)
    ips_st  = zeros(lens)
    ips_eft = zeros(lens)
    ips_lst = zeros(lens)

    for i in 1:length(lqdm_list)

        lqdm = lqdm_list[i]

        iters[i], t[i], s, f[i], ips_tt[i], ips_st[i], ips_eft[i], ips_lst[i] = solve_PDE(lqdm, kkt_system, algorithm, device)
        push!(status, s)
    end
    return t, iters, status, f, ips_tt, ips_st, ips_eft, ips_lst
end

function solve_PDE(lqdm, kkt_system, algorithm, device)

    N  = lqdm.dynamic_data.N
    ns = lqdm.dynamic_data.ns
    for i in 1:2
        if device == 1

            sol_ref = ipopt(lqdm; output_file = "output_files/cpu_sparse_Trange_ns$(ns)_N$(N)", linear_solver="ma27", print_timing_statistics="yes")

            if i == 2
                return sol_ref.iter, sol_ref.elapsed_time, sol_ref.status, sol_ref.objective, 0, 0, 0, 0
            end
        end
    end
end

function build_lqdm(N_range, nx_range, lenx, dt, Tmax, Tstart; dense::Bool = true)
    lqdm_list = []
    for i in N_range
        for j in nx_range

            @time lqdm = build_3D_PDE(i, j, lenx, dt, Tmax, Tstart; dense = dense)
            push!(lqdm_list, lqdm)

            println("Done with T = $i and nx = $j")
        end
    end
    return lqdm_list
end

function run_timers(N_range, nx_range, file_name1, n_vals, lenx, dt, Tmax, Tstart)
    lqdm_list_sparse = build_lqdm(N_range, nx_range, lenx, dt, Tmax, Tstart; dense=false)

    stats = Dict{Symbol, DataFrame}()

    tcpu, kcpu, scpu, fcpu, ttcpu, stcpu, eftcpu, lstcpu    = time_PDE(lqdm_list_sparse, MadNLP.SPARSE_KKT_SYSTEM, 0, 1)
    df = DataFrame(name = [@sprintf("prof%03d", i) for i = 1:length(tcpu)], f = fcpu, t = tcpu, status = scpu, iter = kcpu, tot_time = ttcpu, sol_time = stcpu, fun_eval_time = eftcpu, lin_sol_time = lstcpu, n_vals = n_vals)
    stats[:CPU_MA27] = df

    JLD.save(file_name1, "data", stats)
end


N_range = [10, 20, 30, 40, 50, 75, 100, 125, 150, 200, 250, 300, 350, 400, 450, 500]
nx_range = 4

run_timers(N_range, nx_range, "output_files/nz4_sparse.jld", N_range, .02, .5, 350., 300.)
