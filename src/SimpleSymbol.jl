
struct Variable <: Real
    name::Symbol
end
export Variable

struct Constant <: Real
    value::Float64
end
export Constant

struct DAG{T} <: Real
    children::Union{DAG,Tuple{DAG,DAG},Variable,Constant,Nothing}
    partial::Union{DAG,Tuple{DAG,DAG},Constant}
end
export DAG


variable(name::Symbol) = DAG{Variable}(Variable(name),Constant(1.0))
export variable

#doesn't work
macro variable(var)
    return :($(esc(var)) = variable(:($(esc(var)))))
end
export @variable

constant(value::Float64) = DAG{Constant}(Constant(value),Constant(0.0))

#need to write code to generate all function and derivative DAGs automatically

Base.:*(a::DAG,b::DAG) = DAG{Base.:*}((a,b),(b,a))
Base.:+(a::DAG,b::DAG) = DAG{Base.:+}((a,b),(constant(1.0),constant(1.0)))
Base.sqrt(a::DAG) = DAG{Base.sqrt}(a,:(.5*$a^(-.5))) #need deferred evaluation to avoid endless recursion. This probably won't work

variables() = (variable(:a),variable(:b))
export variables

function test()
    a = variable(:a)
    b = variable(:b)
    a*b
end
export test

