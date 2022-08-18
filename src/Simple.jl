mutable struct Node
    children::Union{Node,Tuple{Node,Node},Nothing}
end

Base.:+(x::Node,y::Node) = :(+($x,$y))

operation(a::Expr) = a.