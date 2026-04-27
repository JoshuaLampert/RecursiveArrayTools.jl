module RecursiveArrayToolsRaggedArraysDiffEqBaseExt

import RecursiveArrayTools: AbstractRaggedVectorOfArray
import DiffEqBase

# Mirror the AbstractVectorOfArray dispatch in DiffEqBase so that adaptive ODE
# solvers compute the correct RMS-normalized norm instead of the unnormalized
# Euclidean norm.  Without these methods, ODE_DEFAULT_NORM falls through to
# `norm(u)` = sqrt(sum_abs2), which is sqrt(n_elements) times larger than the
# intended RMS norm, making the adaptive controller target a stricter tolerance
# than requested (abstol/reltol).

function DiffEqBase.UNITLESS_ABS2(x::AbstractRaggedVectorOfArray)
    return mapreduce(DiffEqBase.UNITLESS_ABS2, +, x.u;
        init = zero(real(eltype(x))))
end

function DiffEqBase.recursive_length(u::AbstractRaggedVectorOfArray)
    return sum(DiffEqBase.recursive_length, u.u; init = 0)
end

function DiffEqBase.ODE_DEFAULT_NORM(u::AbstractRaggedVectorOfArray, _)
    return Base.FastMath.sqrt_fast(
        DiffEqBase.UNITLESS_ABS2(u) / max(DiffEqBase.recursive_length(u), 1))
end

end # module
