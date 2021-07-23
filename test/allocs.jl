"""
    @wrappedallocs(expr)
Given an expression, this macro wraps that expression inside a new function
which will evaluate that expression and measure the amount of memory allocated
by the expression. Wrapping the expression in a new function allows for more
accurate memory allocation detection when using global variables (e.g. when
at the REPL).
For example, `@wrappedallocs(x + y)` produces:
```julia
function g(x1, x2)
    @allocated x1 + x2
end
g(x, y)
```
You can use this macro in a unit test to verify that a function does not
allocate:
```
@test @wrappedallocs(x + y) == 0
```
"""
macro wrappedallocs(expr)
  argnames = [gensym() for a in expr.args]
  quote
    function g($(argnames...))
      @allocated $(Expr(expr.head, argnames...))
    end
    $(Expr(:call, :g, [esc(a) for a in expr.args]...))
  end
end

if Sys.isunix()
  @testset "Allocation tests" begin
    @testset "$S" for S in [LBFGSSolver, TrunkSolver]
      function al_stats(::LBFGSSolver, output, nlp, x)
        GenericExecutionStats(output.status, nlp, solution = x)
        @allocated GenericExecutionStats(output.status, nlp, solution = x)
      end
      function al_stats(solver::TrunkSolver, output, nlp, x)
        GenericExecutionStats(output.status, nlp, solution = x)
        al = @allocated GenericExecutionStats(output.status, nlp, solution = x)
        al_cg = @wrappedallocs cg!(solver.subsolver, solver.H, solver.gx)
        al += al_cg * output.iter
        al += 96 * output.iter # Unidentified allocation
      end

      for n in [2, 20, 200, 2000]
        nlp = SimpleModel(n)
        solver = S(nlp)
        x = copy(nlp.meta.x0)
        al_diff = 0
        al = 0
        with_logger(NullLogger()) do
          output = solve!(solver, nlp)
          al_diff = al_stats(solver, output, nlp, x)
          reset!(solver)
          reset!(nlp)
          al = @wrappedallocs solve!(solver, nlp)
        end
        @test al - al_diff == 0
      end
    end
  end
end
