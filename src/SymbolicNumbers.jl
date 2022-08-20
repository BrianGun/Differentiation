using Calculus

const ReComp = Union{Real,Complex}

struct SymbolicNumber{T<:ReComp} <: Number
    value::T
    epsilon::T
end
SymbolicNumber(x::S, y::T) where {S<:ReComp,T<:ReComp} = SymbolicNumber(promote(x,y)...)
SymbolicNumber(x::ReComp) = SymbolicNumber(x, zero(x))
SymbolicNumber{T}(x::ReComp) where T<:ReComp = SymbolicNumber{T}(T(x), zero(T))

"""Not working"""
macro variable(a)
    return quote
        const $a = Variable($a)
    end
end


const ɛ = SymbolicNumber(false, true)
const imɛ = SymbolicNumber(Complex(false, false), Complex(false, true))

const SymbolicNumber128 = SymbolicNumber{Float64}
const SymbolicNumber64  = SymbolicNumber{Float32}
const SymbolicNumber32  = SymbolicNumber{Float16}
const SymbolicNumberComplex256 = SymbolicNumber{ComplexF64}
const SymbolicNumberComplex128 = SymbolicNumber{ComplexF32}
const SymbolicNumberComplex64  = SymbolicNumber{ComplexF16}

Base.convert(::Type{SymbolicNumber{T}}, z::SymbolicNumber{T}) where {T<:ReComp} = z
Base.convert(::Type{SymbolicNumber{T}}, z::SymbolicNumber) where {T<:ReComp} = SymbolicNumber{T}(convert(T, value(z)), convert(T, epsilon(z)))
Base.convert(::Type{SymbolicNumber{T}}, x::Number) where {T<:ReComp} = SymbolicNumber{T}(convert(T, x), convert(T, 0))
Base.convert(::Type{T}, z::SymbolicNumber) where {T<:ReComp} = (epsilon(z)==0 ? convert(T, value(z)) : throw(InexactError()))

Base.promote_rule(::Type{SymbolicNumber{T}}, ::Type{SymbolicNumber{S}}) where {T<:ReComp,S<:ReComp} = SymbolicNumber{promote_type(T, S)}
Base.promote_rule(::Type{SymbolicNumber{T}}, ::Type{S}) where {T<:ReComp,S<:ReComp} = SymbolicNumber{promote_type(T, S)}
Base.promote_rule(::Type{SymbolicNumber{T}}, ::Type{T}) where {T<:ReComp} = SymbolicNumber{T}

Base.widen(::Type{SymbolicNumber{T}}) where {T} = SymbolicNumber{widen(T)}

value(z::SymbolicNumber) = z.value
epsilon(z::SymbolicNumber) = z.epsilon

value(x::Number) = x
epsilon(x::Number) = zero(typeof(x))

SymbolicNumber(x::ReComp, y::ReComp) = SymbolicNumber(x, y)
SymbolicNumber(x::ReComp) = SymbolicNumber(x)
SymbolicNumber(z::SymbolicNumber) = z

function Base.complex(x::SymbolicNumber, y::SymbolicNumber)
    SymbolicNumber(complex(value(x), value(y)), complex(epsilon(x), epsilon(y)))
end
Base.complex(x::Real, y::SymbolicNumber) = complex(SymbolicNumber(x), y)
Base.complex(x::SymbolicNumber, y::Real) = complex(x, SymbolicNumber(y))
Base.complex(::Type{SymbolicNumber{T}}) where {T} = SymbolicNumber{complex(T)}

const realpart = value
const SymbolicNumberpart = epsilon

Base.isnan(z::SymbolicNumber) = isnan(value(z))
Base.isinf(z::SymbolicNumber) = isinf(value(z))
Base.isfinite(z::SymbolicNumber) = isfinite(value(z))
isSymbolicNumber(x::SymbolicNumber) = true
isSymbolicNumber(x::Number) = false
Base.eps(z::SymbolicNumber) = eps(value(z))
Base.eps(::Type{SymbolicNumber{T}}) where {T} = eps(T)

function SymbolicNumber_show(io::IO, z::SymbolicNumber{T}, compact::Bool) where T<:Real
    x, y = value(z), epsilon(z)
    if isnan(x) || isfinite(y)
        compact ? show(IOContext(io, :compact=>true), x) : show(io, x)
        if signbit(y)==1 && !isnan(y)
            y = -y
            print(io, compact ? "-" : " - ")
        else
            print(io, compact ? "+" : " + ")
        end
        compact ? show(IOContext(io, :compact=>true), y) : show(io, y)
        printtimes(io, y)
        print(io, "ɛ")
    else
        print(io, "SymbolicNumber{",T,"}(", x, ",", y, ")")
    end
end

function SymbolicNumber_show(io::IO, z::SymbolicNumber{T}, compact::Bool) where T<:Complex
    x, y = value(z), epsilon(z)
    xr, xi = reim(x)
    yr, yi = reim(y)
    if isnan(x) || isfinite(y)
        compact ? show(IOContext(io, :compact=>true), x) : show(io, x)
        if signbit(yr)==1 && !isnan(y)
            yr = -yr
            print(io, " - ")
        else
            print(io, " + ")
        end
        if compact
            if signbit(yi)==1 && !isnan(y)
                yi = -yi
                show(IOContext(io, :compact=>true), yr)
                printtimes(io, yr)
                print(io, "ɛ-")
                show(IOContext(io, :compact=>true), yi)
            else
                show(IOContext(io, :compact=>true), yr)
                print(io, "ɛ+")
                show(IOContext(io, :compact=>true), yi)
            end
        else
            if signbit(yi)==1 && !isnan(y)
                yi = -yi
                show(io, yr)
                printtimes(io, yr)
                print(io, "ɛ - ")
                show(io, yi)
            else
                show(io, yr)
                print(io, "ɛ + ")
                show(io, yi)
            end
        end
        printtimes(io, yi)
        print(io, "imɛ")
    else
        print(io, "SymbolicNumber{",T,"}(", x, ",", y, ")")
    end
end

function SymbolicNumber_show(io::IO, z::SymbolicNumber{T}, compact::Bool) where T<:Bool
    x, y = value(z), epsilon(z)
    if !value(z) && epsilon(z)
        print(io, "ɛ")
    else
        print(io, "SymbolicNumber{",T,"}(", x, ",", y, ")")
    end
end

function SymbolicNumber_show(io::IO, z::SymbolicNumber{Complex{T}}, compact::Bool) where T<:Bool
    x, y = value(z), epsilon(z)
    xr, xi = reim(x)
    yr, yi = reim(y)
    if !xr
        if xi*!yr*!yi
            print(io, "im")
        elseif !xi*yr*!yi
            print(io, "ɛ")
        elseif !xi*!yr*yi
            print(io, "imɛ")
        end
    else
        print(io, "SymbolicNumber{",T,"}(", x, ",", y, ")")
    end
end

function printtimes(io::IO, x::Real)
    if !(isa(x,Integer) || isa(x,Rational) ||
         isa(x,AbstractFloat) && isfinite(x))
        print(io, "*")
    end
end

Base.show(io::IO, z::SymbolicNumber) = SymbolicNumber_show(io, z, get(IOContext(io), :compact, false))

function Base.read(s::IO, ::Type{SymbolicNumber{T}}) where T<:ReComp
    x = read(s, T)
    y = read(s, T)
    SymbolicNumber{T}(x, y)
end
function Base.write(s::IO, z::SymbolicNumber)
    write(s, value(z))
    write(s, epsilon(z))
end


## Generic functions of SymbolicNumber numbers ##

Base.convert(::Type{SymbolicNumber}, z::SymbolicNumber) = z
Base.convert(::Type{SymbolicNumber}, x::Number) = SymbolicNumber(x)

Base.:(==)(z::SymbolicNumber, w::SymbolicNumber) = value(z) == value(w)
Base.:(==)(z::SymbolicNumber, x::Number) = value(z) == x
Base.:(==)(x::Number, z::SymbolicNumber) = value(z) == x

Base.isequal(z::SymbolicNumber, w::SymbolicNumber) = isequal(value(z),value(w)) && isequal(epsilon(z), epsilon(w))
Base.isequal(z::SymbolicNumber, x::Number) = isequal(value(z), x) && isequal(epsilon(z), zero(x))
Base.isequal(x::Number, z::SymbolicNumber) = isequal(z, x)

Base.isless(z::SymbolicNumber{<:Real},w::SymbolicNumber{<:Real}) = value(z) < value(w)
Base.isless(z::Real,w::SymbolicNumber{<:Real}) = z < value(w)
Base.isless(z::SymbolicNumber{<:Real},w::Real) = value(z) < w

Base.hash(z::SymbolicNumber) = (x = hash(value(z)); epsilon(z)==0 ? x : bitmix(x,hash(epsilon(z))))

Base.float(z::Union{SymbolicNumber{T}, SymbolicNumber{Complex{T}}}) where {T<:AbstractFloat} = z
Base.complex(z::SymbolicNumber{<:Complex}) = z

Base.floor(z::SymbolicNumber) = floor(value(z))
Base.ceil(z::SymbolicNumber)  = ceil(value(z))
Base.trunc(z::SymbolicNumber) = trunc(value(z))
Base.round(z::SymbolicNumber) = round(value(z))
Base.floor(::Type{T}, z::SymbolicNumber) where {T<:Real} = floor(T, value(z))
Base.ceil( ::Type{T}, z::SymbolicNumber) where {T<:Real} = ceil( T, value(z))
Base.trunc(::Type{T}, z::SymbolicNumber) where {T<:Real} = trunc(T, value(z))
Base.round(::Type{T}, z::SymbolicNumber) where {T<:Real} = round(T, value(z))

for op in (:real, :imag, :conj, :float, :complex)
    @eval Base.$op(z::SymbolicNumber) = SymbolicNumber($op(value(z)), $op(epsilon(z)))
end

Base.abs(z::SymbolicNumber) = sqrt(abs2(z))
Base.abs2(z::SymbolicNumber) = real(conj(z)*z)

Base.real(z::SymbolicNumber{<:Real}) = z
Base.abs(z::SymbolicNumber{<:Real}) = z ≥ 0 ? z : -z

Base.angle(z::SymbolicNumber{<:Real}) = z ≥ 0 ? zero(z) : one(z)*π
function Base.angle(z::SymbolicNumber{Complex{T}}) where T<:Real
    if z == 0
        if imag(epsilon(z)) == 0
            SymbolicNumber(zero(T), zero(T))
        else
            SymbolicNumber(zero(T), convert(T, Inf))
        end
    else
        real(log(sign(z)) / im)
    end
end

Base.flipsign(x::SymbolicNumber,y::SymbolicNumber) = y == 0 ? flipsign(x, epsilon(y)) : flipsign(x, value(y))
Base.flipsign(x, y::SymbolicNumber) = y == 0 ? flipsign(x, epsilon(y)) : flipsign(x, value(y))
Base.flipsign(x::SymbolicNumber, y) = SymbolicNumber(flipsign(value(x), y), flipsign(epsilon(x), y))

# algebraic definitions
conjSymbolicNumber(z::SymbolicNumber) = SymbolicNumber(value(z),-epsilon(z))
absSymbolicNumber(z::SymbolicNumber) = abs(value(z))
abs2SymbolicNumber(z::SymbolicNumber) = abs2(value(z))

# algebra

Base.:+(z::SymbolicNumber, w::SymbolicNumber) = SymbolicNumber(value(z)+value(w), epsilon(z)+epsilon(w))
Base.:+(z::Number, w::SymbolicNumber) = SymbolicNumber(z+value(w), epsilon(w))
Base.:+(z::SymbolicNumber, w::Number) = SymbolicNumber(value(z)+w, epsilon(z))

Base.:-(z::SymbolicNumber) = SymbolicNumber(-value(z), -epsilon(z))
Base.:-(z::SymbolicNumber, w::SymbolicNumber) = SymbolicNumber(value(z)-value(w), epsilon(z)-epsilon(w))
Base.:-(z::Number, w::SymbolicNumber) = SymbolicNumber(z-value(w), -epsilon(w))
Base.:-(z::SymbolicNumber, w::Number) = SymbolicNumber(value(z)-w, epsilon(z))

# avoid ambiguous definition with Bool*Number
Base.:*(x::Bool, z::SymbolicNumber) = ifelse(x, z, ifelse(signbit(real(value(z)))==0, zero(z), -zero(z)))
Base.:*(x::SymbolicNumber, z::Bool) = z*x

Base.:*(z::SymbolicNumber, w::SymbolicNumber) = SymbolicNumber(value(z)*value(w), epsilon(z)*value(w)+value(z)*epsilon(w))
Base.:*(x::Number, z::SymbolicNumber) = SymbolicNumber(x*value(z), x*epsilon(z))
Base.:*(z::SymbolicNumber, x::Number) = SymbolicNumber(x*value(z), x*epsilon(z))

Base.:/(z::SymbolicNumber, w::SymbolicNumber) = SymbolicNumber(value(z)/value(w), (epsilon(z)*value(w)-value(z)*epsilon(w))/(value(w)*value(w)))
Base.:/(z::Number, w::SymbolicNumber) = SymbolicNumber(z/value(w), -z*epsilon(w)/value(w)^2)
Base.:/(z::SymbolicNumber, x::Number) = SymbolicNumber(value(z)/x, epsilon(z)/x)

for f in [:(Base.:^), :(NaNMath.pow)]
    @eval function ($f)(z::SymbolicNumber{T1}, w::SymbolicNumber{T2}) where {T1, T2}
        T = promote_type(T1, T2) # for type stability in ? : statements
        val = $f(value(z), value(w))

        ezvw = epsilon(z) * value(w) # for using in ? : statement
        du1 = iszero(ezvw) ? zero(T) : ezvw * $f(value(z), value(w) - 1)
        ew = epsilon(w) # for using in ? : statement
        # the float is for type stability because log promotes to floats
        du2 = iszero(ew) ? zero(float(T)) : ew * val * log(value(z))
        du = du1 + du2

        SymbolicNumber(val, du)
    end
end

Base.mod(z::SymbolicNumber, n::Number) = SymbolicNumber(mod(value(z), n), epsilon(z))

# introduce a boolean !iszero(n) for hard zero behaviour to combat NaNs
function pow(z::SymbolicNumber, n::AbstractFloat)
    return SymbolicNumber(value(z)^n, !iszero(n) * (epsilon(z) * n * value(z)^(n - 1)))
end
function pow(z::SymbolicNumber{T}, n::Integer) where T
    iszero(n) && return SymbolicNumber(one(T), zero(T)) # avoid DomainError Int^(negative Int)
    isone(z) && return SymbolicNumber(one(T), epsilon(z) * n)
    return SymbolicNumber(value(z)^n, epsilon(z) * n * value(z)^(n - 1))
end
# these first two definitions are needed to fix ambiguity warnings
for T1 ∈ (:Integer, :Rational, :Number)
    @eval Base.:^(z::SymbolicNumber{T}, n::$T1) where T = pow(z, n)
end


NaNMath.pow(z::SymbolicNumber{T}, n::Number) where T = SymbolicNumber(NaNMath.pow(value(z),n), epsilon(z)*n*NaNMath.pow(value(z),n-1))
NaNMath.pow(z::Number, w::SymbolicNumber{T}) where T = SymbolicNumber(NaNMath.pow(z,value(w)), epsilon(w)*NaNMath.pow(z,value(w))*log(z))

Base.inv(z::SymbolicNumber) = SymbolicNumber(inv(value(z)),-epsilon(z)/value(z)^2)

# force use of NaNMath functions in derivative calculations
function to_nanmath(x::Expr)
    if x.head == :call
        funsym = Expr(:.,:NaNMath,Base.Meta.quot(x.args[1]))
        return Expr(:call,funsym,[to_nanmath(z) for z in x.args[2:end]]...)
    else
        return Expr(:call,[to_nanmath(z) for z in x.args]...)
    end
end
to_nanmath(x) = x




for (funsym, exp) in Calculus.symbolic_derivatives_1arg()
    funsym == :exp && continue
    funsym == :abs2 && continue
    funsym == :inv && continue
    if isdefined(SpecialFunctions, funsym)
        @eval function SpecialFunctions.$(funsym)(z::SymbolicNumber)
            x = value(z)
            xp = epsilon(z)
            SymbolicNumber($(funsym)(x),xp*$exp)
        end
    elseif isdefined(Base, funsym)
        @eval function Base.$(funsym)(z::SymbolicNumber)
            x = value(z)
            xp = epsilon(z)
            SymbolicNumber($(funsym)(x),xp*$exp)
        end
    end
    # extend corresponding NaNMath methods
    if funsym in (:sin, :cos, :tan, :asin, :acos, :acosh, :atanh, :log, :log2, :log10,
          :lgamma, :log1p)
        funsym = Expr(:.,:NaNMath,Base.Meta.quot(funsym))
        @eval function $(funsym)(z::SymbolicNumber)
            x = value(z)
            xp = epsilon(z)
            SymbolicNumber($(funsym)(x),xp*$(to_nanmath(exp)))
        end
    end
end

# only need to compute exp/cis once
Base.exp(z::SymbolicNumber) = (expval = exp(value(z)); SymbolicNumber(expval, epsilon(z)*expval))
Base.cis(z::SymbolicNumber) = (cisval = cis(value(z)); SymbolicNumber(cisval, im*epsilon(z)*cisval))

Base.exp10(x::SymbolicNumber) = (y = exp10(value(x)); SymbolicNumber(y, y * log(10) * epsilon(x)))

## TODO: should be generated in Calculus
Base.sinpi(z::SymbolicNumber) = SymbolicNumber(sinpi(value(z)),epsilon(z)*cospi(value(z))*π)
Base.cospi(z::SymbolicNumber) = SymbolicNumber(cospi(value(z)),-epsilon(z)*sinpi(value(z))*π)

function Base.atan(y::SymbolicNumber, x::SymbolicNumber)
    u = value(x)^2 + value(y)^2
    return SymbolicNumber(atan(value(y), value(x)), (value(x)/u) * epsilon(y) - (value(y)/u) * epsilon(x))
end
function Base.atan(y::SymbolicNumber, x::Real)
    u = x^2 + value(y)^2
    return SymbolicNumber(atan(value(y), x), (x/u) * epsilon(y))
end
function Base.atan(y::Real, x::SymbolicNumber)
    u = value(x)^2 + y^2
    return SymbolicNumber(atan(y, value(x)), (y/u) * -epsilon(x))
end

Base.checkindex(::Type{Bool}, inds::AbstractUnitRange, i::SymbolicNumber) = checkindex(Bool, inds, value(i))
