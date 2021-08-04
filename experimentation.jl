using LinearAlgebra, Distributions, Distances, NearestNeighbors, Printf, Plots

include("functions.jl")
include("nnlsq_pen.jl")

pyplot();

"""
    estimate_P(state_space, sample_size, S, φ, ϵ, basis_grid_size, integral_resolution; dist)

Combines all of the helper functions into a single API for estimating the matrix P.

Returns T.

"""
function estimate_P(grid::Array{Float64, 2}, sample_size::Integer, range::Number=2π, k::Number=0.0, integral_resolution::Integer=100; setup, map_type)

    if setup == "grid"

        sample_grid_size = Int(floor(sqrt(sample_size)));
        if sample_grid_size ^ 2 != sample_size
            throw(ArgumentError("invalid sample size"))
        end;
        s = creategrid(0, range, sample_grid_size);

        X = s;
        Y = S(X; map_type);

        basis_locs = X;
        n_bases = sample_size;
        basis_grid_size = sample_grid_size;

        ϵ = 1 * range / basis_grid_size;
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

        ϵ = max_NN_distance(X, 1.3);
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
    P_diagnostics(output)

Returns a number of diagnostic tests on the estimate P, including several visualisations.

"""
function P_diagnostics(output, visuals=false)
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
    grid_size = Int(sqrt(n_gridpoints));
    sample_size = size(X,1);
    n_bases = sample_size;

    output_string = string("exp-$setup-$map_type-$sample_size-$integral_resolution");

    metrics = Dict()

    # Generate the initial plots...
    # ...for the grid setup
    if setup == "grid"
        β = rand(n_bases);
        test_surface = basis_combination(grid, basis_locs, φ, ϵ, β);
        if visuals
            surface(grid[:,1], grid[:,2], test_surface;
                legend=false,
                zlims=(0, maximum(test_surface)*1.1),
                title="Sample surface")
            scatter!(X[:,1], X[:,2], zeros(sample_size))
            savefig(string("estimations/", output_string, "-1_sample_surface.pdf"));
        end;

        basis_surface = basis_combination(grid, basis_locs, φ, ϵ, ones(n_bases));
        basis_surface_integral = (range)^2 * sum(basis_surface) / n_gridpoints;
        η = ones(n_bases) / basis_surface_integral;

        approximate_invariant_density= basis_combination(grid, basis_locs, φ, ϵ, η);
        if visuals
            surface(grid[:,1], grid[:,2], approximate_invariant_density;
                legend=false,
                zlims=(0, maximum(approximate_invariant_density)*1.2),
                title="Approximate Lebesgue surface")
            scatter!(X[:,1], X[:,2], zeros(sample_size))
            savefig(string("estimations/", output_string, "-1_lebesgue_surface.pdf"));
        end;

    end;

    # ...for the scatter setup
    if setup == "scattered"
        β = ones(n_bases);
        test_surface = basis_combination(grid, basis_locs, φ, ϵ, β);
        if visuals
            surface(grid[:,1], grid[:,2], test_surface;
                legend=false)
            scatter!(X[:,1], X[:,2], zeros(sample_size))
            title!("Sample surface")
            savefig(string("estimations/", output_string, "-1_sample_surface.pdf"));
        end;

        weighted_surface = basis_combination(grid, basis_locs, φ, ϵ, w);
        if visuals
            surface(grid[:,1], grid[:,2], weighted_surface; legend=false)
            scatter!(X[:,1], X[:,2], zeros(sample_size))
            zlims!(0, maximum(weighted_surface)*1.1)
            title!("Approximate Lebesgue surface")
            savefig(string("estimations/", output_string, "-1_lebesgue_surface.pdf"));
        end;
    end;


    # Next the eigendecomposition
    λ, Λ = ordered_eigendecomp(transpose(P));

    u, v = real.(λ), imag.(λ);
    xc, yc = cos.(LinRange(0, 2π, 500)), sin.(LinRange(0, 2π, 500));

    if visuals
        scatter(u, v, label="eigenvalues");
        plot!(xc, yc, label="unit circle");
        title!("Spectrum (leading eigenvalue: $(λ[1]))");
        savefig(string("estimations/", output_string, "-2_eig_decomp.pdf"));
    end;

    # Then the row and column sums
    row_sums = sum(P, dims=2)[:];
    col_sums = sum(P, dims=1)[:];

    if visuals
        plot(row_sums, label="row sums");
        plot!(col_sums, label="column sums");
        xlabel!("index");
        ylabel!("sum");
        title!("Row and column sums");
        savefig(string("estimations/", output_string, "-3_row_col_sums.pdf"));
    end;

    # Then the invariant density
    α = real.(Λ[:,1]);
    α = abs.(α);

    estimated_invariant_density = basis_combination(grid, basis_locs, φ, ϵ, α);
    estimated_invariant_density_integral = ((range^2) / n_gridpoints) * sum(estimated_invariant_density);
    α_normalised = α / estimated_invariant_density_integral;
    estimated_invariant_density_normalised = basis_combination(grid, basis_locs, φ, ϵ, α_normalised);

    if visuals
        surface(grid[:,1], grid[:,2], estimated_invariant_density_normalised;
            legend=false,
            zlims=(0, maximum(estimated_invariant_density_normalised)*1.2),
            title="Normalised estimate of invariant density");
        scatter!(X[:,1], X[:,2], zeros(sample_size));
        savefig(string("estimations/", output_string, "-4_invariant_density_estimate.pdf"));
    end;

    # Calculate the convergence metrics
    if setup == "grid"
        basis_surface_integral = (range)^2 * sum(basis_surface) / n_gridpoints;
        η = ones(n_bases) / basis_surface_integral;
    elseif setup == "scattered"
        weighted_surface_integral = (range^2) * sum(weighted_surface) / n_gridpoints;
        η = w / weighted_surface_integral;
    end;

    if map_type == "standard" || map_type == "cat"
        approximate_invariant_density = basis_combination(grid, basis_locs, φ, ϵ, η);
        true_invariant_density = ones(n_gridpoints) / (range^2);
    elseif map_type == "wave"
        xxs = LinRange(0, range, grid_size);
        yys = ((0.5 * cos.(xxs)) .+ 1) ./ ((range)^2 );
        wave_surface = repeat(yys, inner=grid_size);
        wave_surface_integral = (range^2 / n_gridpoints) * sum(wave_surface)
        wave_surface_normalised = wave_surface ./ wave_surface_integral

        approximate_invariant_density = wave_surface_normalised;
        true_invariant_density = wave_surface_normalised;
    end;

    estimation_gap = estimated_invariant_density_normalised - approximate_invariant_density;
    approximation_gap = approximate_invariant_density - true_invariant_density;

    metrics["estimation_l1"] = Lp_norm(estimation_gap; p="one");
    metrics["estimation_l2"] = Lp_norm(estimation_gap; p="two");
    metrics["estimation_l∞"] = Lp_norm(estimation_gap; p="∞");

    metrics["approximation_l1"] = Lp_norm(approximation_gap; p="one");
    metrics["approximation_l2"] = Lp_norm(approximation_gap; p="two");
    metrics["approximation_l∞"] = Lp_norm(approximation_gap; p="∞");

    # Last, show the evolution of a function
    if visuals
        β = rand(n_bases)
        initial_density = basis_combination(grid, basis_locs, φ, ϵ, β);
        initial_density_integral = ( (range)^2  / n_gridpoints ) * sum(initial_density);

        β = β / initial_density_integral;
        initial_density = basis_combination(grid, basis_locs, φ, ϵ, β);
        initial_density_integral = ( (range)^2  / n_gridpoints ) * sum(initial_density);

        fixed_zlims = (0, maximum(initial_density)*1.2);

        surface(grid[:,1], grid[:,2], initial_density;
            legend=false,
            zlims=fixed_zlims,
            title="Random initial density")
        savefig(string("estimations/", output_string, "-5_evolution_0.pdf"))

        β1 = transpose(P) * β;
        evolved_density = basis_combination(grid, basis_locs, φ, ϵ, β1);
        evolved_integral = ( (range)^2 / n_gridpoints ) * sum(evolved_density);
        surface(grid[:,1], grid[:,2], evolved_density;
            legend=false,
            zlims=fixed_zlims,
            title="1 applications of P (int: $evolved_integral)")
        savefig(string("estimations/", output_string, "-5_evolution_1.pdf"))

        β2 = transpose(P) * β1;
        evolved_density = basis_combination(grid, basis_locs, φ, ϵ, β2);
        evolved_integral = ( (range)^2 / n_gridpoints ) * sum(evolved_density);
        surface(grid[:,1], grid[:,2], evolved_density;
            legend=false,
            zlims=fixed_zlims,
            title="2 applications of P (int: $evolved_integral)")
        savefig(string("estimations/", output_string, "-5_evolution_2.pdf"));

        β3 = transpose(P) * β2;
        evolved_density = basis_combination(grid, basis_locs, φ, ϵ, β3);
        evolved_integral = ( (range)^2 / n_gridpoints ) * sum(evolved_density);
        surface(grid[:,1], grid[:,2], evolved_density;
            legend=false,
            zlims=fixed_zlims,
            title="3 applications of P (int: $evolved_integral)")
        savefig(string("estimations/", output_string, "-5_evolution_3.pdf"));

        β4 = transpose(P) * β3;
        evolved_density = basis_combination(grid, basis_locs, φ, ϵ, β4);
        evolved_integral = ( (range)^2 / n_gridpoints ) * sum(evolved_density);
        surface(grid[:,1], grid[:,2], evolved_density;
            legend=false,
            zlims=fixed_zlims,
            title="4 applications of P (int: $evolved_integral)")
        savefig(string("estimations/", output_string, "-5_evolution_4.pdf"));
    end;

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

        plot(sample_sizes, estimation_l1s, label="L1", markershape=:circle, yaxis=:log)
        plot!(sample_sizes, estimation_l2s, label="L2", markershape=:circle, yaxis=:log)
        plot!(sample_sizes, estimation_l∞s, label="L∞", markershape=:circle, yaxis=:log)
        xlabel!("N");
        ylabel!("norm");

    elseif series == "approximation"
        approximation_l1s = [results[n]["approximation_l1"] for n in 1:n_exps];
        approximation_l2s = [results[n]["approximation_l2"] for n in 1:n_exps];
        approximation_l∞s = [results[n]["approximation_l∞"] for n in 1:n_exps];

        plot(sample_sizes, approximation_l1s, label="L1", markershape=:circle, yaxis=:log)
        plot!(sample_sizes, approximation_l2s, label="L2", markershape=:circle, yaxis=:log)
        plot!(sample_sizes, approximation_l∞s, label="L∞", markershape=:circle, yaxis=:log)
        xlabel!("N");
        ylabel!("norm");
    end;
end;

"""
    average_the_results(results)

Averages the results of a randomised experiment.

"""

function average_the_results(results)
    n_experiments = length(results);
    n_runs = length(results[1]);

    averaged_results = Vector{Dict}(undef, n_experiments);

    for n in 1:n_experiments
        experiment = results[n];
        averages = Dict();

        estimation_l1s = [experiment[m]["estimation_l1"] for m in 1:n_runs];
        estimation_l2s = [experiment[m]["estimation_l2"] for m in 1:n_runs];
        estimation_l∞s = [experiment[m]["estimation_l∞"] for m in 1:n_runs];

        approximation_l1s = [experiment[m]["approximation_l1"] for m in 1:n_runs];
        approximation_l2s = [experiment[m]["approximation_l2"] for m in 1:n_runs];
        approximation_l∞s = [experiment[m]["approximation_l∞"] for m in 1:n_runs];

        averages["estimation_l1"] = mean(estimation_l1s);
        averages["estimation_l2"] = mean(estimation_l2s);
        averages["estimation_l∞"] = mean(estimation_l∞s);

        averages["approximation_l1"] = mean(approximation_l1s);
        averages["approximation_l2"] = mean(approximation_l2s);
        averages["approximation_l∞"] = mean(approximation_l∞s);

        averaged_results[n] = averages;
    end;

    return averaged_results
end;
