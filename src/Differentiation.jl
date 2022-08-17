module Differentiation

struct Variable
end

struct Node{C<:Union{Variable,Node,Tuple{Node,Node}}}
    f::Function
    v::C
end


function derivativeRules(f,x::Variable,y::Union{Variable,Nothing}=nothing)


    rules = Dict{Function,Function}(
        (log->1/x),
        (exp->exp(x)),
        (Base.:^ -> n*x^(n-1)), #need a way to distinguish this, x^n, from x^y
        (Base.:* -> (y,x)),
        (Base.:+ -> (1,1)),
        (Base.:- -> (1,-1)),
        (Base.:/ -> (1/y,-x/y^2)),
        (Base.:^ -> (y*x^(y-1),x^y*log(x))), #this is the x^y case
        (sin -> cos(x)),
        (cos -> -sin(x)),
        (tan -> sec(x)^2),
        (sec -> sec(x)*tan(x)), #derivatives below here have not been checked
        (csc -> -csc(x)*cot(x)),
        (cot -> -csc(x)^2),
        (sinh -> cosh(x)),
        (cosh -> sinh(x)),
        (tanh -> sech(x)^2),
        (sech -> -sech(x)*tanh(x)),
        (csch -> -csch(x)*coth(x)),
        (coth -> -csch(x)^2),
        (asin -> 1/sqrt(1-x^2)),
        (acos -> -1/sqrt(1-x^2)),
        (atan -> 1/(1+x^2)),
        (asec -> 1/(x*sqrt(x^2-1))),
        (acsc -> -1/(x*sqrt(x^2-1))),
        (acot -> -1/(1+x^2)),
        (asinh -> 1/sqrt(1+x^2)),
        (acosh -> 1/sqrt(x^2-1)),
        (atanh -> 1/(1-x^2)),
        (asech -> -1/(x*sqrt(1-x^2))),
        (acsch -> -1/(x*sqrt(1+x^2))),
        (acoth -> -1/(1-x^2))
    )
end

#need a macro here to autogenerate all these calls

Base.exp(x::Node{Variable}) 
Base.exp(n::Node{Node})
Base.exp(n::Node{Tuple{Node,Node}})


end # module
