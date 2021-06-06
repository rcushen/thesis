using JuMP, Gurobi, LinearAlgebra

function nnlsq(A, b, lb)
    # Minimises \|Ax-b\|^2, subject to x≥lb, where lb is a real number (like 0 for nonnegative least squares)
    # Call with x, residual, objvalue = nnlsq(A, b, lb)

    model = Model(Gurobi.Optimizer)
    # how to set optimizer parameters, not necessary here.
    #    set_optimizer_attribute(model, "Method", 0)
    #    set_optimizer_attribute(model, "Presolve", 0)

    n1 = size(A, 1)
    n2 = size(A, 2) 
    
    @variable(model, x[i=1:n2] ≥ lb)
    @variable(model, residual[i=1:n1])
     
    @constraint(model, residual .== A * x - b)

    @objective(model, Min, sum(residual.^2))

    
    # Solve problem using MIP solver
    JuMP.optimize!(model)
    println("Objective value is: ", JuMP.objective_value(model))

    return JuMP.value.(x), JuMP.value.(residual), JuMP.objective_value(model)

end

