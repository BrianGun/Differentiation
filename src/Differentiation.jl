module Differentiation

struct Variable
end

function derivativeRules(f,x::Variable,y::Union{Variable,Nothing}=nothing)


    rules = Dict{Function,Function}(
        (log(x)->1/x),
        (exp(x)->exp(x)),
        (x^n -> n*x^(n-1)),
        (x*y -> (y,x)),
        (x+y -> (1,1)),
        (x-y -> (1,-1)),
        (x/y -> (1/y,-x/y^2)),
        (x^y -> (y*x^(y-1),x^y*log(x))),
        (sin(x) -> cos(x)),
        (cos(x) -> -sin(x)),
        (tan(x) -> sec(x)^2),
        (sec(x) -> sec(x)*tan(x)), #derivatives below here have not been checked
        (csc(x) -> -csc(x)*cot(x)),
        (cot(x) -> -csc(x)^2),
        (sinh(x) -> cosh(x)),
        (cosh(x) -> sinh(x)),
        (tanh(x) -> sech(x)^2),
        (sech(x) -> -sech(x)*tanh(x)),
        (csch(x) -> -csch(x)*coth(x)),
        (coth(x) -> -csch(x)^2),
        (asin(x) -> 1/sqrt(1-x^2)),
        (acos(x) -> -1/sqrt(1-x^2)),
        (atan(x) -> 1/(1+x^2)),
        (asec(x) -> 1/(x*sqrt(x^2-1))),
        (acsc(x) -> -1/(x*sqrt(x^2-1))),
        (acot(x) -> -1/(1+x^2)),
        (asinh(x) -> 1/sqrt(1+x^2)),
        (acosh(x) -> 1/sqrt(x^2-1)),
        (atanh(x) -> 1/(1-x^2)),
        (asech(x) -> -1/(x*sqrt(1-x^2))),
        (acsch(x) -> -1/(x*sqrt(1+x^2))),
        (acoth(x) -> -1/(1-x^2)),



    )
end

log(x) -> 

end # module
