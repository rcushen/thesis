using JuMP, Gurobi, LinearAlgebra

function nnlsq_pen(A, b, x̄, p)
    # Minimises \|Ax-b\|^2 + p\|x-x̄\|^2, subject to x≥0, where x̄ is the average value expected in the vector x, and p≥0 is a small scalar penalty
    # Call with x, residual, objvalue = nnlsq(A, b, x̄, p)

    model = Model(Gurobi.Optimizer)
    # how to set optimizer parameters, not necessary here.
    #    set_optimizer_attribute(model, "Method", 0)
    #    set_optimizer_attribute(model, "Presolve", 0)

    n1 = size(A, 1)
    n2 = size(A, 2)

    @variable(model, x[i=1:n2] ≥ 0)
    @variable(model, residual[i=1:n1])

    @constraint(model, residual .== A * x - b)

    #has to be quadratic penalty below.  Because sum of x ≈ constant and a linear penalty has no effect.  This is why the first penalty and the second penalty yield almost identical results.
    @objective(model, Min, sum(residual.^2).+p*sum((x.-x̄).^2))
    #the above yields almost identical results to the below, but very slightly better in terms of total weight accuracy
    #@objective(model, Min, sum(residual.^2).+penalty*sum(x.^2))

    # Solve problem
    JuMP.optimize!(model)
    println("Objective value is: ", JuMP.objective_value(model))

    return JuMP.value.(x), JuMP.value.(residual), JuMP.objective_value(model)

end
