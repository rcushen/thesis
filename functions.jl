using LinearAlgebra, Distributions, Distances

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
    f_0(x, range; dist)

Returns a density evaluation of `x` for some initial density f0.

Output is a pdf of `dist` evaluated at `x`.

"""
function f0(x, range::Number; dist="uniform")
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
    sampledist(sample_size, width; dist)

Returns a sample from the nominated distribution `dist, of size `sample_size`.

Output is a `sample_size` x 2 matrix.

"""
function sampledist(sample_size, range::Number; dist="uniform")
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
function S(X; map_type)
    if map_type == "standard"
        a = 6;
        result = [X[:,1] + X[:,2]  X[:,2] + a*sin.(X[:,1] + X[:,2])];
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

Returns the result of `S` applied to `X ``n_steps` times, iteratively.

Output is a transformed matrix S^n(X) of same size.

"""
function S_forward(X, S, n_steps::Int64; map_type)
    for t in 1:n_steps
        X = S(X; map_type);
    end
    return X
end;

"""
    φ(x, z, ϵ)

Returns the evaluation of ```x``` at the kernel defined by ```z``` and ```ϵ```.

Output is a real scalar.

"""
φ(x, z, ϵ) =  exp( -1 * ( peuclidean(x, z, [2π, 2π]) / ϵ ) ^ 2 );

"""
    evaluate_phi(sample, basis_locs, φ, ϵ)

Returns the evaluation matrix of all points in a sample against all basis functions.

Output is an m×n matrix, where m is the number of bases and n is the number of data points.
"""
function evaluate_phi(sample, basis_locs, φ, ϵ)
    n_bases = size(basis_locs, 1)
    sample_size = size(sample, 1)

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

Output is an n×m matrix of integral values, where n is the sample size (number of rows in `Y`) and m is the number of bases (number of rows in `basis_locs`)

"""
function integrate_phiy(Y, basis_locs, range::Number, integral_resolution::Integer, ϵ)
    n_fine_gridpoints = integral_resolution ^ 2;
    fine_grid = creategrid(0, range, integral_resolution);

    n_bases = size(basis_locs, 1);
    sample_size = size(Y, 1);

    distance_matrix = pairwise(PeriodicEuclidean([range, range]), fine_grid, basis_locs, dims=1)

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
    construct_L(w, Φ, Ξ)

Constructs the estimated matrix L, using the weights `w`, the sample evaluation matrix `Φ`, and the integral values `Ξ`, with integral weights `c`.

Returns an m×m matrix, where m is the number of bases.
"""
function construct_L(w, Φ, Ξ, c)
    n_bases = size(Φ, 1);
    sample_size = size(Φ, 2);

    L = Array{Float64}(undef, n_bases, n_bases);
    for b in 1:n_bases
        for k in 1:n_bases
            val = 0;
            for n in 1:sample_size
                val += w[n] * Φ[b,n] * Ξ[n,k];
            end
            val = 1/c * val;
            L[b,k] = val;
        end
    end
    return L
end;

"""
    estimate_L(state_space, sample_size, S, φ, ϵ, basis_grid_size, integral_resolution; dist)

Combines all of the helper functions into a single API for estimating the matrix L.

Returns L.

"""
function estimate_L(state_space, sample_size, S, φ, ϵ, basis_grid_size, integral_resolution; dist)

    c = π * ϵ^2;

    width = state_space["max"] - state_space["min"];

    if dist == "normal"
        d = MvNormal([width/2, width/2], 1);
        sample = rand(d, sample_size)';
    elseif dist == "uniform"
        sample = rand(sample_size, 2)
    end

    X = sample;
    Y = S(X);

    basis_locs = creategrid(state_space["min"], state_space["max"], basis_grid_size);
    n_bases = basis_grid_size ^ 2;

    Φ = evaluate_phi(X, basis_locs, φ, ϵ);
    C = c * ones(n_bases);
    w, residual, objvalue = nnlsq(Φ, C, 0);

    Ξ = integrate_phiy(Y, basis_locs, width, integral_resolution, ϵ);

    L = construct_L(w, Φ, Ξ, c)
    return L
end;

"""
    ordered_eigendecomp(M)

Computes a sorted eigendecomposition of M.

Returns a tuple of (lambda, Lambda), where lambda is the vector of eigenvalues and Lambda is the matrix of eigenvectors.

"""
function ordered_eigendecomp(M)
    λ, Λ = eigvals(M), eigvecs(M)
    p = sortperm(abs.(λ), rev=true)
    λ = λ[p];
    Λ = Λ[:,p];

    return λ, Λ
end;

"""
    basis_combination(grid, basis_locs, φ, ϵ, α)

Computes a weighted sum of all basis functions, according to `alpha`.

Returns a vector of length n, where n is the number of gridpoints.

"""
function basis_combination(grid, basis_locs, φ, ϵ, α)
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
