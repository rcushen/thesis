using LinearAlgebra, Distributions, Distances, NearestNeighbors, Printf, Plots, Roots

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
function S(X::Array{Float64, 2}, a::Number=1.0, δ::Number=1.0, b::Number=0.5; map_type="standard")
    if map_type == "standard"
        y = X[:,2] + a * sin.(X[:,1]);
        x = X[:,1] + y;
        #x = X[:,1] + X[:,2];
        #y = X[:,2] + (a * sin.(x));
        result = [x y];
        result = mod.(result, 2π);
    elseif map_type == "cat"
        result = [(2 * X[:,1]) + X[:,2]  X[:,1] + X[:,2]];
        result = mod.(result, 2π);
    elseif map_type == "Tδ"
        pt1 = [(2 * X[:,1]) + X[:,2]  X[:,1] + X[:,2]];
        pt2 = [cos.(2*π* X[:,1]) sin.(4*π*X[:,2])];
        result = pt1 + δ * pt2
        result = mod.(result, 2π);
    elseif map_type == "wave"

        function D(X::Array{Float64, 2}, b::Number)
            result = [X[:,1] + b * sin.(X[:,1]) X[:,2]]
            result = mod.(result, 2π);
            return result
        end;

        function T(X::Array{Float64, 2})
            result = [(2 * X[:,1]) + X[:,2]  X[:,1] + X[:,2]];
            result = mod.(result, 2π);
            return result
        end;

        function Dinv(X::Array{Float64, 2}, b::Number)
            sample_size = size(X, 1);
            xprimes = Vector{Float64}(undef, sample_size);
            for n in 1:sample_size
                f(x) = x + sin(x) - X[n,1];
                xprimes[n] = find_zero(f, (-2π, 2π))
            end;
            result = [xprimes X[:,2]]
            result = mod.(result, 2π);
            return result
        end;

        result = Dinv(T(D(X, b)), b)
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
function max_NN_distance(X::Array{Float64, 2}, scalar::Number=1.0)
    btree = BruteTree(X');
    idxs, dsts = knn(btree, X', 2);
    neighbour_distances = [dsts[n][1] for n in 1:size(dsts, 1)];
    return maximum(neighbour_distances) * scalar
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
