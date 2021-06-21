# adapted from https://gist.github.com/Keno/d0c2df947f67be543036238a0caeb1c6

# a random implementation of the `sin(::Float64)` -> `fast_sin(::Float64)` rewriter

let counter = 0
    global fast_sin, get_fast_sin_counter, reset_counter
    fast_sin(x::Float64) = (counter += 1; sin(x::Float64))
    get_fast_sin_counter() = counter
    reset_counter() = (counter = 0)
end

import Core:
    MethodInstance
const CC = Core.Compiler
import .CC:
    specialize_method,
    AbstractInterpreter,
    NativeInterpreter,
    OptimizationState,
    OptimizationParams,
    IRCode,
    finish
import Base:
    get_world_counter,
    to_tuple_type
import Base.Meta: isexpr
using XCompiler

struct FastSinInterpreter <: XInterpreter
    native::NativeInterpreter
    cache_key::UInt
    function FastSinInterpreter(world = get_world_counter(); cache_key = 0, kwargs...)
        native = NativeInterpreter(world;
                                   inf_params = XCompiler.XInferenceParams(; kwargs...),
                                   opt_params = XCompiler.XOptimizationParams(; kwargs...),
                                   )
        cache_key = convert(UInt, cache_key)
        cache_key = hash(native.inf_params, cache_key)
        cache_key = hash(native.opt_params, cache_key)

        return new(native, cache_key)
    end
end
XCompiler.native(interp::FastSinInterpreter)    = interp.native
XCompiler.cache_key(interp::FastSinInterpreter) = interp.cache_key

let interp′ = FastSinInterpreter()
    target =
        XCompiler.enter_gf_by_type!(interp′, to_tuple_type((typeof(sin), Float64)))[2].result.linfo => # from
        XCompiler.enter_gf_by_type!(interp′, to_tuple_type((typeof(fast_sin), Float64)))[2].result.linfo # to

    function CC.finish(interp::FastSinInterpreter, opt::OptimizationState, params::OptimizationParams, ir::IRCode, @nospecialize(result))
        ret = finish(native(interp), opt, params, ir, result)

        for stmt in opt.ir.stmts
            inst = stmt[:inst]
            if isexpr(inst, :invoke) && inst.args[1]::MethodInstance === first(target)
                new = Expr(:invoke,
                           last(target),
                           GlobalRef(@__MODULE__, :fast_sin),
                           inst.args[3:end]...,
                           )
                CC.setindex!(stmt, new, :inst)
            elseif isexpr(inst, :call) && (f = first(inst.args); isa(f, GlobalRef) && f.name === :sin)
                inst.args[1] = GlobalRef(@__MODULE__, :fast_sin)
            elseif isexpr(inst, :call) && first(inst.args) === sin
                inst.args[1] = fast_sin
            end
        end

        return ret
    end
end

# test

using Test

get_sin() = sin
function static_dispatch(x)
    reset_counter()
    sin(x)
    @test get_fast_sin_counter() == 1
    get_sin()(x)
    @test get_fast_sin_counter() == 2
    get_sin()(Base.inferencebarrier(x)) # not well inferred, but we know the function at least
    @test get_fast_sin_counter() == 3
end

with(FastSinInterpreter()) do
    @testset "static dispatch" begin
        static_dispatch(1.0) # should be okay
    end
end

@testset "make this testset fail" begin
    static_dispatch(1.0) # we want this not to pass, but XCompiler just reuses the native cache ...
end

# we also can't handle dynamic dispatches
function dynamic_dispatch(x)
    reset_counter()
    sin(x)
    @test get_fast_sin_counter() == 1
    get_sin()(x)
    @test get_fast_sin_counter() == 2
    Base.inferencebarrier(get_sin)()(x)
    @test get_fast_sin_counter() == 3
end

with(FastSinInterpreter()) do
    @testset "dynamic dispatch" begin
        dynamic_dispatch(1.0) # fail
    end
end
