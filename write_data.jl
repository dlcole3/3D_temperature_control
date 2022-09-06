using CSV, DataFrames, JLD, MadNLP

dense_T50   = load("output_files/T50_condensed.jld")["data"]
dense_T150  = load("output_files/T150_condensed.jld")["data"]
dense_T250  = load("output_files/T250_condensed.jld")["data"]
dense_nz4   = load("output_files/nz4_condensed.jld")["data"]
sparse_T50  = load("output_files/T50_sparse.jld")["data"]
sparse_T150 = load("output_files/T150_sparse.jld")["data"]
sparse_T250 = load("output_files/T250_sparse.jld")["data"]
sparse_nz4  = load("output_files/nz4_sparse.jld")["data"]


column_titles_T = "T, f, t, status, iter, tot_time, sol_time, fun_eval_time, lin_sol_time"
column_titles_nz = "nz, f, t, status, iter, tot_time, sol_time, fun_eval_time, lin_sol_time"

function printdata(d, df, i)
    println(d, df.n_vals[i], ",", df.f[i], ",", df.t[i], ",", df.status[i], ",", df.iter[i], ",", df.tot_time[i], ",",
    df.sol_time[i], ",", df.fun_eval_time[i], ",", df.lin_sol_time[i])
end

open("3d_temp_control_data.csv","w") do d
    println(d, "sparse CPU N = 50")
    println(d, column_titles_nz)
    for i in 1:size(sparse_T50[:CPU_MA27], 1)
       printdata(d, sparse_T50[:CPU_MA27], i)
    end
    println(d, "condensed CPU N = 50")
    println(d, column_titles_nz)
    for i in 1:size(dense_T50[:CPU_CH], 1)
       printdata(d, dense_T50[:CPU_CH], i)
    end
    println(d, "condensed GPU N = 50")
    println(d, column_titles_nz)
    for i in 1:size(dense_T50[:GPU_CH], 1)
       printdata(d, dense_T50[:GPU_CH], i)
    end

    println(d)
    println(d, "sparse CPU N = 150")
    println(d, column_titles_nz)
    for i in 1:size(sparse_T150[:CPU_MA27], 1)
       printdata(d, sparse_T150[:CPU_MA27], i)
    end
    println(d, "condensed CPU N = 150")
    println(d, column_titles_nz)
    for i in 1:size(dense_T150[:CPU_CH], 1)
       printdata(d, dense_T150[:CPU_CH], i)
    end
    println(d, "condensed GPU N = 150")
    println(d, column_titles_nz)
    for i in 1:size(dense_T150[:GPU_CH], 1)
       printdata(d, dense_T150[:GPU_CH], i)
    end

    println(d)
    println(d, "sparse CPU N = 250")
    println(d, column_titles_nz)
    for i in 1:size(sparse_T250[:CPU_MA27], 1)
       printdata(d, sparse_T250[:CPU_MA27], i)
    end
    println(d, "condensed CPU N = 250")
    println(d, column_titles_nz)
    for i in 1:size(dense_T250[:CPU_CH], 1)
       printdata(d, dense_T250[:CPU_CH], i)
    end
    println(d, "condensed GPU N = 250")
    println(d, column_titles_nz)
    for i in 1:size(dense_T250[:GPU_CH], 1)
       printdata(d, dense_T250[:GPU_CH], i)
    end

    println(d)
    println(d, "sparse CPU nz = 4")
    println(d, column_titles_T)
    for i in 1:size(sparse_nz4[:CPU_MA27], 1)
       printdata(d, sparse_nz4[:CPU_MA27], i)
    end
    println(d, "condensed CPU nz = 4")
    println(d, column_titles_T)
    for i in 1:size(dense_nz4[:CPU_CH], 1)
       printdata(d, dense_nz4[:CPU_CH], i)
    end
    println(d, "condensed GPU nz = 4")
    println(d, column_titles_T)
    for i in 1:size(dense_nz4[:GPU_CH], 1)
       printdata(d, dense_nz4[:GPU_CH], i)
    end
end
