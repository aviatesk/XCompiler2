# adapted from https://gist.github.com/Keno/d0c2df947f67be543036238a0caeb1c6

let counter = 0
    global fast_sin, get_fast_sin_counter, reset_counter
    fast_sin(x::Float64) = (counter += 1; sin(x::Float64))
    get_fast_sin_counter() = counter
    reset_counter() = (counter = 0)
end

# implementation
# ==============

import Core:
    MethodInstance
const CC = Core.Compiler
import .CC:
    AbstractInterpreter,
    NativeInterpreter,
    InferenceState,
    OptimizationState,
    OptimizationParams,
    IRCode,
    optimize,
    argextype,
    widenconst
import Base:
    @invoke,
    get_world_counter,
    to_tuple_type
import Base.Meta: isexpr
using XCompiler

# implements very inefficient rewrite of `sin(::Float64)` -> `fast_sin(::Float64)`;
# ignores anything involved with inlining and discards static calls — this is just PoC
struct FastSinInterpreter <: XInterpreter
    native::NativeInterpreter
    cache_key::UInt
    function FastSinInterpreter(world = get_world_counter(); cache_key = 0, inlining=false, kwargs...)
        native = NativeInterpreter(world;
                                   inf_params = XCompiler.XInferenceParams(; kwargs...),
                                   opt_params = XCompiler.XOptimizationParams(; inlining, kwargs...),
                                   )
        cache_key = convert(UInt, cache_key)
        cache_key = hash(native.inf_params, cache_key)
        cache_key = hash(native.opt_params, cache_key)
        XCompiler.maybe_init_cache!(cache_key)

        return new(native, cache_key)
    end
end
XCompiler.native(interp::FastSinInterpreter)    = interp.native
XCompiler.cache_key(interp::FastSinInterpreter) = interp.cache_key

let
    tme = first(methods(fast_sin, (Float64,)))
    ttt = to_tuple_type((typeof(fast_sin), Float64,))

    function CC.optimize(interp::FastSinInterpreter, opt::OptimizationState, params::OptimizationParams, @nospecialize(result))
        @assert isnothing(opt.ir)

        linfo = opt.linfo
        if !(linfo.def === tme && linfo.specTypes === ttt)
            (; src, sptypes, slottypes) = opt
            for (i, x) in enumerate(src.code)
                if isexpr(x, :call) && length(x.args) == 2
                    ft = widenconst(argextype(x.args[1], src, sptypes, slottypes))
                    if ft === typeof(sin)
                        at = widenconst(argextype(x.args[2], src, sptypes, slottypes))
                        if at === Float64
                            src.code[i] = Expr(:call, GlobalRef(@__MODULE__, :fast_sin), x.args[2])
                        end
                    end
                end
            end
        end

        return optimize(native(interp), opt, params, result)
    end
end

# test
# ====

using Test

get_sin() = sin
function f(x, replace)
    reset_counter()
    @testset "simple" begin
        sin(x)
        @test get_fast_sin_counter() == (replace ? 1 : 0)
    end
    @testset "a bit complex but still inferred" begin
        get_sin()(x)
        @test get_fast_sin_counter() == (replace ? 2 : 0)
    end
    @testset "dynamic dispatch" begin
        get_sin()(Base.inferencebarrier(x))
        @test get_fast_sin_counter() == (replace ? 3 : 0) # fail, we can't hijack the dynamic dispatch
    end
end

@testset "testset" begin
    @testset "customized compilation" begin
        with(FastSinInterpreter()) do
            f(1.0, true)
        end
    end

    @testset "don't affect native code cache" begin
        f(1.0, false) # fail, since code cache can be inserted outside of `CC.code_cache`
    end
end
