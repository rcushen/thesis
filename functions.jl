using LinearAlgebra, Distributions, Distances

include("nnlsq.jl")
include("nnlsq_pen.jl")

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
function sampledist(sample_size::Integer, range::Number; dist="uniform")
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
    evaluate_phi(sample, basis_locs, φ, ϵ)

Returns the evaluation matrix of `φ` of all points in `sample` against all basis functions, specified by `basis_locs`.

Output is an m × n matrix, where m is the number of bases and n is the number of data points.
"""
function evaluate_phi(sample::Array{Float64, 2}, basis_locs::Array{Float64, 2}, φ, ϵ::Number)
    n_bases = size(basis_locs, 1);
    sample_size = size(sample, 1);

    Φ = Array{Float64}(undef, n_bases, sample_size);
    for b in 1:n_bases
        for n in 1:sample_size
            Φ[b, n] = φ(sample[n, :], basis_locs[b,:], ϵ);
        end
    end
    return Φ
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
    construct_T(w, Φ, Ξ, c)

Constructs the estimated matrix T, using the weights `w`, the sample evaluation matrix `Φ`, and the integral values `Ξ`, with integral weights `c`.

Returns an m × m matrix, where m is the number of bases.
"""
function construct_T(w::Vector{Float64}, Φ::Array{Float64, 2}, Ξ::Array{Float64, 2}, c::Number)
    n_bases = size(Φ, 1);
    sample_size = size(Φ, 2);

    T = Array{Float64}(undef, n_bases, n_bases);
    for b in 1:n_bases
        for k in 1:n_bases
            val = 0;
            for n in 1:sample_size
                val += w[n] * Φ[b,n] * Ξ[n,k];
            end
            val = (1/(c^2)) * val;
            T[b,k] = val;
        end
    end
    return T
end;

"""
    estimate_T(state_space, sample_size, S, φ, ϵ, basis_grid_size, integral_resolution; dist)

Combines all of the helper functions into a single API for estimating the matrix T.

Returns T.

"""
function estimate_T(grid::Array{Float64, 2}, sample::Array{Float64, 2}, basis_grid_size::Integer, ϵ::Number, range::Number=2π, integral_resolution::Integer=100; map_type="standard")

    c = π * ϵ^2;

    X = sample;
    Y = S(X; map_type);

    basis_locs = creategrid(0, range, basis_grid_size);
    n_bases = basis_grid_size ^ 2;

    Φ = evaluate_phi(X, basis_locs, φ, ϵ);

    C = c * ones(n_bases);
    w_av = (range^2)/n_bases;

    w, residual, objvalue = nnlsq_pen(Φ, C, w_av, 0.01);

    Ξ = integrate_phiy(Y, basis_locs, range, integral_resolution, φ, ϵ);

    T = construct_T(w, Φ, Ξ, c)
    return T
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
