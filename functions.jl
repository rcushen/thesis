using LinearAlgebra, Distributions, Distances, NearestNeighbors, Printf, Plots

include("nnlsq.jl")
include("nnlsq_pen.jl")

pyplot();

"""
    creategrid(min, max, resolution)

Creates a uniform grid of points on the region [`min`, `max`] by [`min, max`], proportional to `resolution`.

Output is a (`resolution` ^ 2) x 2 matrix, where each row is a grid point.

"""
function creategrid(min::Number, max::Number, resolution::Int64)
    range = max - min;
    min += (range / resolution) / 2;
    max -= (range / resolution) / 2;
    xs = LinRange(min, max, resolution);
    ys = LinRange(min, max, resolution);
    grid = [[x, y] for x in xs for y in ys];
    grid = transpose(hcat(grid...));
    grid = convert(Array{Float64}, grid);
    return grid
end;

"""
    f(x, range; dist)

Returns a density evaluation of `x` for some density, according to `dist`.

Output is a pdf of `dist` evaluated at `x`.

"""
function f(x::Vector{Float64}, range::Number; dist="uniform")
    if dist == "normal"
        d = MvNormal([range/2, range/2], 1)
        return pdf(d, x)
    elseif dist == "uniform"
        return 1 / (range ^ 2)
    else
        throw(ArgumentError(dist, "unsupported distribution"))
    end
end;

"""
    sampledist(sample_size, range; dist)

Returns a sample from the nominated distribution `dist`, of size `sample_size`.

Output is a `sample_size` x 2 matrix.

"""
function sampledist(sample_size::Integer, range::Number=2π; dist="uniform")
    if dist == "normal"
        d = MvNormal([range/2, range/2], 1);
        sample = rand(d, sample_size)';
    elseif dist == "uniform"
        sample = range .* rand(sample_size, 2)
    end
    sample = convert(Array{Float64}, sample)
    return sample
end

"""
    S(X; map_type)

Returns the evaluation of a dynamical map on the matrix of points `X`, according to `map_type`.

Output is a transformed matrix S(X) of same size.

"""
function S(X::Array{Float64, 2}; map_type="standard")
    if map_type == "standard"
        a = 6;
        y = X[:,2] + a * sin.(X[:,1]);
        x = X[:,1] + y;
        result = [x y];
        result = mod.(result, 2π);
    elseif map_type == "cat"
        result = [(2 * X[:,1]) + X[:,2]  X[:,1] + X[:,2]];
        result = mod.(result, 2π);
    elseif map_type == "identity"
        result = X;
    else
        throw(ArgumentError(map_type, "unsupported map"))
    end
    return result
end;

"""
    S_forward(X, S, n_steps; map_type)

Returns the result of `S` applied to `X` `n_steps` times, iteratively.

Output is a transformed matrix S^n(X) of same size.

"""
function S_forward(X::Array{Float64, 2}, S, n_steps::Int64; map_type="standard")
    for t in 1:n_steps
        X = S(X; map_type);
    end
    return X
end;

"""
    φ(x, z, ϵ)

Returns the evaluation of `x` at the kernel defined by `z` and `ϵ`, on the flat torus with range 2π.

Output is a real scalar.

"""
φ(x::Vector{Float64}, z::Vector{Float64}, ϵ::Number) =  exp( -1 * ( peuclidean(x, z, [2π, 2π]) / ϵ ) ^ 2 );

"""
    mean_NN_distance(X)

Returns the mean distance of the nearest neighbour for each point in `X`.

Output is a real scalar.
"""
function max_NN_distance(X::Array{Float64, 2})
    btree = BruteTree(X');
    idxs, dsts = knn(btree, X', 2);
    neighbour_distances = [dsts[n][1] for n in 1:size(dsts, 1)];
    return maximum(neighbour_distances)
end
"""
    evaluate_funcs(sample, func_locs, φ, ϵ)

Returns the evaluation matrix of `φ` of all points in `sample` against all ϕ functions, specified by `func_locs`.

Output is an m × n matrix, where m is the number of functions and n is the number of data points.
"""
function evaluate_funcs(sample::Array{Float64, 2}, func_locs::Array{Float64, 2}, φ, ϵ::Number)
    n_funcs = size(func_locs, 1);
    sample_size = size(sample, 1);

    Ψ = Array{Float64}(undef, n_funcs, sample_size);
    for b in 1:n_funcs
        for n in 1:sample_size
            Ψ[b, n] = φ(sample[n, :], func_locs[b,:], ϵ);
        end;
    end;
    return Ψ
end

"""
    integrate_phiy(Y, basis_locs, resolution)

Integrates kernels φ centered at the image points `Y` over a Voronoi tesselation of the original `basis_locs`.

Output is an n × m matrix of integral values, where n is the sample size and m is the number of bases.

"""
function integrate_phiy(Y::Array{Float64, 2}, basis_locs::Array{Float64, 2}, range::Number, integral_resolution::Integer, φ, ϵ::Number)
    n_fine_gridpoints = integral_resolution ^ 2;
    fine_grid = creategrid(0, range, integral_resolution);

    n_bases = size(basis_locs, 1);
    sample_size = size(Y, 1);

    distance_matrix = pairwise(PeriodicEuclidean([range, range]), fine_grid, basis_locs, dims=1);

    tile_indexes = argmin(distance_matrix, dims=2);
    tile_indexes = [ind[2] for ind in tile_indexes];
    tile_indexes = tile_indexes[:];

    integral_values = Array{Float64}(undef, sample_size, n_bases);
    for j in 1:sample_size
        y_j = Y[j,:];
        for b in 1:n_bases
            relevant_points = tile_indexes .== b;
            Vb = fine_grid[relevant_points, :];
            n_relevant_points = size(Vb, 1);

            total = 0;
            for i in 1:n_relevant_points
                total += φ(Vb[i,:], y_j, ϵ);
            end
            result = ((range^2) / n_fine_gridpoints) * total;
            integral_values[j,b] = result;
        end
    end
    return integral_values
end;

"""
    construct_P(w, Φ, Ξ, c)

Constructs the estimated matrix P, using the weights `w`, the evaluation matrix `Φ`, and the integral values `Ξ`, with integral weights `c`.

Returns an m × m matrix, where m is the number of bases.
"""
function construct_P(w::Vector{Float64}, Φ::Array{Float64, 2}, Ξ::Array{Float64, 2}, c::Number)
    n_bases = size(Φ, 1);
    sample_size = size(Φ, 2);

    P = Array{Float64}(undef, n_bases, n_bases);
    for b in 1:n_bases
        for k in 1:n_bases
            val = 0;
            for n in 1:sample_size
                val += w[n] * Φ[b,n] * Ξ[n,k];
            end
            val = (1/(c^2)) * val;
            P[b,k] = val;
        end
    end
    return P
end;

"""
    ordered_eigendecomp(M)

Computes a sorted eigendecomposition of the matrix `M`.

Returns a tuple of (λ, Λ), where λ is the vector of eigenvalues and Λ a the matrix of eigenvectors.

"""
function ordered_eigendecomp(M)
    λ, Λ = eigvals(M), eigvecs(M);
    p = sortperm(abs.(λ), rev=true);
    λ = λ[p];
    Λ = Λ[:,p];

    return λ, Λ
end;

"""
    basis_combination(grid, basis_locs, φ, ϵ, α)

Computes a weighted sum of all basis functions, according to `α`.

Returns a vector of length n, where n is the number of gridpoints.

"""
function basis_combination(grid::Array{Float64, 2}, basis_locs::Array{Float64, 2}, φ, ϵ::Number, α::Vector{Float64})
    n_bases = size(basis_locs, 1);
    n_gridpoints = size(grid, 1);

    evaluation_matrix = Array{Float64}(undef, n_gridpoints, n_bases);
    for b in 1:n_bases
        for n in 1:n_gridpoints
            evaluation_matrix[n, b] = α[b] * φ(grid[n, :], basis_locs[b,:], ϵ);
        end
    end
    evaluation_surface = sum(evaluation_matrix, dims=2)[:];
    return evaluation_surface
end;

"""
    Lp_norm(surf; p="one")

Computes the Lp norm of the function g with surface given by surf.

Returns a scalar.

"""
function Lp_norm(surf, range::Number=2π; p="one")
    n_gridpoints = size(surf, 1);
    integral_weights = range^2 / n_gridpoints;
    if p == "one"
        v = integral_weights * sum(abs.(surf));
    elseif p == "two"
        v = sqrt(integral_weights * sum(surf .^2));
    elseif p == "∞"
        v = maximum(abs.(surf))
    else
        throw(ArgumentError(map_type, "unsupported p"))
    end;
    return v
end;

"""
    estimate_P(state_space, sample_size, S, φ, ϵ, basis_grid_size, integral_resolution; dist)

Combines all of the helper functions into a single API for estimating the matrix P.

Returns T.

"""
function estimate_P(grid::Array{Float64, 2}, sample_size::Integer, range::Number=2π, k::Number=0.0, integral_resolution::Integer=100; setup, map_type="standard")

    if setup == "grid"

        sample_grid_size = Int(floor(sqrt(sample_size)));
        if sample_grid_size ^ 2 != sample_size
            throw(ArgumentError(map_type, "invalid sample size"))
        end;
        s = creategrid(0, range, sample_grid_size);

        X = s;
        Y = S(X; map_type);

        basis_locs = X;
        n_bases = sample_size;
        basis_grid_size = sample_grid_size;

        ϵ = range / basis_grid_size;
        c = π * ϵ^2;

        w = range^2 / sample_size * ones(sample_size);

        Ξ = integrate_phiy(Y, basis_locs, range, integral_resolution, φ, ϵ);
        Φ = evaluate_funcs(X, basis_locs, φ, ϵ);
        P = construct_P(w, Φ, Ξ, c);

    elseif setup == "scattered"

        s = sampledist(sample_size);

        X = s;
        Y = S(X; map_type);

        basis_locs = X;
        n_bases = sample_size;

        ϵ = max_NN_distance(X);
        c = π * ϵ^2;

        test_function_grid_size = 50;
        n_test_functions = test_function_grid_size ^ 2;
        test_function_locs = creategrid(0, range, test_function_grid_size);

        ϵ_test_functions = ϵ;

        Ψ = evaluate_funcs(X, test_function_locs, φ, ϵ_test_functions);
        c_test = π * ϵ_test_functions^2;
        C = c_test * ones(n_test_functions);
        w_av = (range^2) / sample_size;

        w, residual, objvalue = nnlsq_pen(Ψ, C, w_av, k);

        Ξ = integrate_phiy(Y, basis_locs, range, integral_resolution, φ, ϵ);
        Φ = evaluate_funcs(X, basis_locs, φ, ϵ);
        P = construct_P(w, Φ, Ξ, c);

    else
        throw(ArgumentError(map_type, "unsupported setup"))
    end;

    output = Dict(
        "grid"=>grid,
        "X"=>X,
        "Y"=>Y,
        "basis_locs"=>basis_locs,
        "ϵ"=>ϵ,
        "c"=>c,
        "range"=>range,
        "integral_resolution"=>integral_resolution,
        "setup"=>setup,
        "map_type"=>map_type,
        "P"=>P,
        "w"=>w
    )
    return output
end;

"""
    P_diagnostics()

Returns a number of diagnostic tests on the estimate P, including several visualisations.

"""
function P_diagnostics(output)
    grid = output["grid"];
    X = output["X"];
    basis_locs = output["basis_locs"];
    ϵ = output["ϵ"];
    c = output["c"];
    range = output["range"];
    integral_resolution = output["integral_resolution"];
    setup = output["setup"];
    map_type = output["map_type"];
    P = output["P"];
    w = output["w"];

    n_gridpoints = size(grid, 1);
    sample_size = size(X,1);
    n_bases = sample_size;

    output_string = string("exp-$setup-$map_type-$sample_size-$integral_resolution");

    metrics = Dict()

    # Start by showing the basis surface
    basis_evaluation_matrix = Array{Float64}(undef, n_gridpoints, n_bases);
    for b in 1:n_bases
        for n in 1:n_gridpoints
            basis_evaluation_matrix[n, b] = φ(grid[n, :], basis_locs[b, :], ϵ);
        end
    end

    basis_surface = sum(basis_evaluation_matrix, dims=2)[:];
    surface(grid[:,1], grid[:,2], basis_surface; legend=false);
    zlims!(0,maximum(basis_surface)*1.1);
    title!("Basis surface")
    savefig(string("estimations/", output_string, "-1_basis_surface.pdf"))

    # Then a sample surface, for the grid setup
    if setup == "grid"
        β = rand(n_bases)
        test_surface = basis_combination(grid, basis_locs, φ, ϵ, β);
        surface(grid[:,1], grid[:,2], test_surface; legend=false);
        title!("Sample function");
        savefig(string("estimations/", output_string, "-1_sample_surface.pdf"));
    end;

    # And a plot of the Lebesgue surface, for the scattered setup
    if setup == "scattered"
        weighted_surface = basis_combination(grid, basis_locs, φ, ϵ, w);
        surface(grid[:,1], grid[:,2], weighted_surface; legend=false)
        scatter!(X[:,1], X[:,2], zeros(sample_size))
        zlims!(0, maximum(weighted_surface)*1.1)
        savefig(string("estimations/", output_string, "-1_lebesgue_surface.pdf"));
    end;


    # Next the eigendecomposition
    λ, Λ = ordered_eigendecomp(P);

    u, v = real.(λ), imag.(λ);
    xc, yc = cos.(LinRange(0, 2π, 500)), sin.(LinRange(0, 2π, 500));

    scatter(u, v, label="eigenvalues");
    plot!(xc, yc, label="unit circle");
    title!("Spectrum (leading eigenvalue: $(λ[1]))");
    savefig(string("estimations/", output_string, "-2_eig_decomp.pdf"));

    # Then the row and column sums
    row_sums = sum(P, dims=2)[:];
    col_sums = sum(P, dims=1)[:];

    plot(row_sums, label="row sums");
    plot!(col_sums, label="column sums");
    xlabel!("index");
    ylabel!("sum");
    title!("Row and column sums");
    savefig(string("estimations/", output_string, "-3_row_col_sums.pdf"));

    # Then the invariant density
    α = real.(Λ[:,1]);
    α = abs.(α);

    estimated_invariant_density = basis_combination(grid, basis_locs, φ, ϵ, α);
    integral = (range^2) * sum(estimated_invariant_density) / n_gridpoints;
    α_normalised = α / integral;
    estimated_invariant_density_normalised = basis_combination(grid, basis_locs, φ, ϵ, α_normalised);

    surface(grid[:,1], grid[:,2], estimated_invariant_density_normalised; legend=false);
    zlims!(0, maximum(estimated_invariant_density_normalised)*1.1);
    title!("Normalised estimate of invariant density");
    savefig(string("estimations/", output_string, "-4_invariant_density_estimate.pdf"));

    # Calculate the convergence metrics
    if setup == "grid"
        integral = (range^2) * sum(basis_surface) / n_gridpoints;
        η = ones(n_bases) / integral;
    elseif setup == "scattered"
        integral = (range^2) * sum(weighted_surface) / n_gridpoints;
        η = w / integral;
    end;

    approximate_invariant_density = basis_combination(grid, basis_locs, φ, ϵ, η);
    estimation_gap = estimated_invariant_density_normalised - approximate_invariant_density;

    constant_function = ones(n_gridpoints) / (range^2);
    approximation_gap = approximate_invariant_density - constant_function;

    metrics["estimation_l1"] = Lp_norm(estimation_gap; p="one");
    metrics["estimation_l2"] = Lp_norm(estimation_gap; p="two");
    metrics["estimation_l∞"] = Lp_norm(estimation_gap; p="∞");

    metrics["approximation_l1"] = Lp_norm(approximation_gap; p="one");
    metrics["approximation_l2"] = Lp_norm(approximation_gap; p="two");
    metrics["approximation_l∞"] = Lp_norm(approximation_gap; p="∞");

    # Last, show the evolution of a function

    β = rand(n_bases)
    initial_density = basis_combination(grid, basis_locs, φ, ϵ, β);
    integral = (range)^2 * sum(initial_density) / n_gridpoints;
    β = β / integral;
    initial_density = basis_combination(grid, basis_locs, φ, ϵ, β);
    integral = (range)^2 * sum(initial_density) / n_gridpoints;
    surface(grid[:,1], grid[:,2], initial_density; legend=false, zlims=(0, maximum(initial_density)*1.1));
    title!("Random initial function (int: $integral)");
    savefig(string("estimations/", output_string, "-5_evolution_0.pdf"))

    β1 = P * β;
    evolved_density = basis_combination(grid, basis_locs, φ, ϵ, β1);
    integral = (range)^2 * sum(evolved_density) / n_gridpoints;
    surface(grid[:,1], grid[:,2], evolved_density; legend=false, zlims=(0, maximum(initial_density)*1.1));
    title!("1 applications of P (int: $integral)");
    savefig(string("estimations/", output_string, "-5_evolution_1.pdf"))

    β2 = P * β1;
    evolved_density = basis_combination(grid, basis_locs, φ, ϵ, β2);
    integral = (range)^2 * sum(evolved_density) / n_gridpoints;
    p2 = surface(grid[:,1], grid[:,2], evolved_density; legend=false, zlims=(0, maximum(initial_density)*1.1));
    title!("2 applications of P (int: $integral)");
    savefig(string("estimations/", output_string, "-5_evolution_2.pdf"));

    β3 = P * β2;
    evolved_density = basis_combination(grid, basis_locs, φ, ϵ, β3);
    integral = (range)^2 * sum(evolved_density) / n_gridpoints;
    surface(grid[:,1], grid[:,2], evolved_density; legend=false, zlims=(0, maximum(initial_density)*1.1));
    title!("3 applications of P (int: $integral)");
    savefig(string("estimations/", output_string, "-5_evolution_3.pdf"));

    β4 = P * β3;
    evolved_density = basis_combination(grid, basis_locs, φ, ϵ, β4);
    integral = (range)^2 * sum(evolved_density) / n_gridpoints;
    surface(grid[:,1], grid[:,2], evolved_density; legend=false, zlims=(0, maximum(initial_density)*1.1));
    title!("4 applications of P (int: $integral)");
    savefig(string("estimations/", output_string, "-5_evolution_4.pdf"));

    return metrics
end;

"""
    plot_metrics(results, sample_sizes, series)

Plots metrics from the results of an experiment captured in `results`, specified by `series`.

"""

function plot_metrics(results, sample_sizes, series)
    n_exps = length(sample_sizes);

    if series == "estimation"
        estimation_l1s = [results[n]["estimation_l1"] for n in 1:n_exps];
        estimation_l2s = [results[n]["estimation_l2"] for n in 1:n_exps];
        estimation_l∞s = [results[n]["estimation_l∞"] for n in 1:n_exps];

        plot(sample_sizes, estimation_l1s, label="l1", markershape=:circle)
        plot!(sample_sizes, estimation_l2s, label="l2", markershape=:circle)
        plot!(sample_sizes, estimation_l∞s, label="l∞", markershape=:circle)
        title!("Estimation error to invariant density")
        xlabel!("N");
        ylabel!("norm");

    elseif series == "approximation"
        approximation_l1s = [results[n]["approximation_l1"] for n in 1:n_exps];
        approximation_l2s = [results[n]["approximation_l2"] for n in 1:n_exps];
        approximation_l∞s = [results[n]["approximation_l∞"] for n in 1:n_exps];

        plot(sample_sizes, approximation_l1s, label="l1", markershape=:circle)
        plot!(sample_sizes, approximation_l2s, label="l2", markershape=:circle)
        plot!(sample_sizes, approximation_l∞s, label="l∞", markershape=:circle)
        title!("Approximation error to invariant density")
        xlabel!("N");
        ylabel!("norm");
    end;
end;
