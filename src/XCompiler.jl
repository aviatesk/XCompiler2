module XCompiler

# overloads
# ---------

import Core.Compiler:
    # types.jl
    InferenceParams,
    OptimizationParams,
    get_world_counter,
    get_inference_cache,
    lock_mi_inference,
    unlock_mi_inference,
    add_remark!,
    may_optimize,
    may_compress,
    may_discard_trees,
    verbose_stmt_info,
    # xcache.jl
    code_cache,
    get_inference_cache,
    cache_lookup

# usings
# ------

import Core:
    CodeInfo,
    MethodInstance,
    CodeInstance,
    MethodMatch,
    LineInfoNode,
    SimpleVector,
    Builtin,
    Typeof,
    svec

import Core.Compiler:
    AbstractInterpreter,
    NativeInterpreter,
    InferenceState,
    InferenceResult,
    OptimizationState,
    WorldRange,
    WorldView,
    MethodCallResult,
    VarTable,
    IRCode,
    to_tuple_type,
    _methods_by_ftype,
    specialize_method,
    inlining_enabled,
    typeinf # entry

import Base:
    unwrap_unionall,
    rewrap_unionall,
    destructure_callex

const CC = Core.Compiler

# AbstractInterpreter interface
# =============================

abstract type XInterpreter <: AbstractInterpreter end

# XInterpreter API requirements

native(interp::XInterpreter) = throw(native)
cache_key(::XInterpreter)    = throw(cache_key)

# TODO allow customization

CC.InferenceParams(interp::XInterpreter)    = InferenceParams(native(interp))
CC.OptimizationParams(interp::XInterpreter) = OptimizationParams(native(interp))
CC.get_world_counter(interp::XInterpreter)  = get_world_counter(native(interp))

CC.lock_mi_inference(::XInterpreter,   ::MethodInstance) = nothing
CC.unlock_mi_inference(::XInterpreter, ::MethodInstance) = nothing

CC.add_remark!(interp::XInterpreter, sv, s) = add_remark!(native(interp), sv, s)

CC.may_optimize(interp::XInterpreter)      = may_optimize(native(interp))
CC.may_compress(interp::XInterpreter)      = may_compress(native(interp))
CC.may_discard_trees(interp::XInterpreter) = may_discard_trees(native(interp))
CC.verbose_stmt_info(interp::XInterpreter) = verbose_stmt_info(native(interp))

XInferenceParams(; ipo_constant_propagation::Bool        = true,
                   aggressive_constant_propagation::Bool = false,
                   unoptimize_throw_blocks::Bool         = true,
                   max_methods::Int                      = 3,
                   union_splitting::Int                  = 4,
                   apply_union_enum::Int                 = 8,
                   tupletype_depth::Int                  = 3,
                   tuple_splat::Int                      = 32,
                   __xconfigs...) =
    return InferenceParams(; ipo_constant_propagation,
                             aggressive_constant_propagation,
                             unoptimize_throw_blocks,
                             max_methods,
                             union_splitting,
                             apply_union_enum,
                             tupletype_depth,
                             tuple_splat,
                             )
XOptimizationParams(; inlining::Bool                = inlining_enabled(),
                      inline_cost_threshold::Int    = 100,
                      inline_nonleaf_penalty::Int   = 1000,
                      inline_tupleret_bonus::Int    = 250,
                      inline_error_path_cost::Int   = 20,
                      max_methods::Int              = 3,
                      tuple_splat::Int              = 32,
                      union_splitting::Int          = 4,
                      unoptimize_throw_blocks::Bool = true,
                      __xconfigs...) =
    return OptimizationParams(; inlining,
                                inline_cost_threshold,
                                inline_nonleaf_penalty,
                                inline_tupleret_bonus,
                                inline_error_path_cost,
                                max_methods,
                                tuple_splat,
                                union_splitting,
                                unoptimize_throw_blocks,
                                )
# # assert here that they create same objects as the original constructors
@assert XInferenceParams()    == InferenceParams()
@assert XOptimizationParams() == OptimizationParams()

# cache
# =====

# global
# ------

"""
    X_CODE_CACHE::$(typeof(X_CODE_CACHE))

Keeps `CodeInstance` cache associated with `mi::MethodInstace` that represent the result of
an inference on `mi` performed by `XInterpreter`.
The cache is partitioned by identities of each `XInterpreter`, and thus running a pipeline
with different configurations will yeild a different cache and never influenced by the
previous inference.
This cache is completely separated from the `NativeInterpreter`'s global cache, so that
XCompiler.jl's analysis never interacts with actual code execution (like, execution of `XCompiler` itself).
"""
const X_CODE_CACHE = IdDict{UInt, IdDict{MethodInstance,CodeInstance}}()

function maybe_init_cache!(cache_key::UInt)
    haskey(X_CODE_CACHE, cache_key) && return
    X_CODE_CACHE[cache_key] = IdDict{MethodInstance,CodeInstance}()
end
__clear_cache!()       = empty!(X_CODE_CACHE)

function CC.code_cache(interp::XInterpreter)
    cache = XGlobalCache(interp)
    worlds = WorldRange(get_world_counter(interp))
    return WorldView(cache, worlds)
end

struct XGlobalCache
    interp::XInterpreter
end

# cache existence for this `analyzer` is ensured on its construction
x_code_cache(interp::XInterpreter)         = X_CODE_CACHE[cache_key(interp)]
x_code_cache(wvc::WorldView{XGlobalCache}) = x_code_cache(wvc.cache.interp)

native_code_cache(interp::XInterpreter)         = CC.code_cache(native(interp))
native_code_cache(wvc::WorldView{XGlobalCache}) = native_code_cache(wvc.cache.interp)

CC.haskey(wvc::WorldView{XGlobalCache}, mi::MethodInstance) = haskey(x_code_cache(wvc), mi)

function CC.get(wvc::WorldView{XGlobalCache}, mi::MethodInstance, default)
    x = get(x_code_cache(wvc), mi, default)
    if x === default
        return default
    end
    # now we can assume code exists within the native cache, just retrieve it
    return CC.get(native_code_cache(wvc), mi, default)
end

function CC.getindex(wvc::WorldView{XGlobalCache}, mi::MethodInstance)
    r = CC.get(wvc, mi, nothing)
    r === nothing && throw(KeyError(mi))
    return r::CodeInstance
end

function CC.setindex!(wvc::WorldView{XGlobalCache}, ci::CodeInstance, mi::MethodInstance)
    x_code_cache(wvc)[mi] = ci
    add_x_callback!(mi) # register the callback on invalidation
    CC.setindex!(native_code_cache(wvc), ci, mi) # bypass to the native cache
    return nothing
end

function add_x_callback!(linfo)
    if !isdefined(linfo, :callbacks)
        linfo.callbacks = Any[invalidate_x_cache!]
    else
        if !any(@nospecialize(cb)->cb===invalidate_x_cache!, linfo.callbacks)
            push!(linfo.callbacks, invalidate_x_cache!)
        end
    end
    return nothing
end

function invalidate_x_cache!(replaced, max_world, depth = 0)
    for cache in values(X_CODE_CACHE); delete!(cache, replaced); end

    if isdefined(replaced, :backedges)
        for mi in replaced.backedges
            mi = mi::MethodInstance
            if !any(cache->haskey(cache, mi), values(X_CODE_CACHE))
                continue # otherwise fall into infinite loop
            end
            invalidate_x_cache!(mi, max_world, depth+1)
        end
    end
    return nothing
end

# local
# -----

# to inspect local cache, we will just look for `get_inference_cache(native(interp))`
# and thus we won't create a global store to hold these caches

CC.get_inference_cache(interp::XInterpreter) = XLocalCache(interp)

struct XLocalCache
    interp::XInterpreter
end

CC.cache_lookup(linfo::MethodInstance, given_argtypes::Vector{Any}, inf_cache::XLocalCache) =
    return cache_lookup(linfo, given_argtypes, get_inference_cache(native(inf_cache.interp)))

CC.push!(inf_cache::XLocalCache, inf_result::InferenceResult) =
    return CC.push!(get_inference_cache(native(inf_cache.interp)), inf_result)

# entry
# =====

function with(@nospecialize(f), interp::XInterpreter)
    tt = Tuple{Typeof(f)}
    enter_gf_by_type!(interp, tt)
    return f()
end

# TODO `enter_call_builtin!` ?
function enter_gf_by_type!(interp::XInterpreter,
                           @nospecialize(tt::Type{<:Tuple}),
                           world::UInt = get_world_counter(interp),
                           )
    mms = _methods_by_ftype(tt, InferenceParams(interp).MAX_METHODS, world)
    @assert mms !== false "unable to find matching method for $(tt)"

    filter!(mm->mm.spec_types===tt, mms)
    @assert length(mms) == 1 "unable to find single target method for $(tt)"

    mm = first(mms)::MethodMatch

    return enter_method_signature!(interp, mm.method, mm.spec_types, mm.sparams)
end

function enter_method_signature!(interp::XInterpreter,
                                 m::Method,
                                 @nospecialize(atype),
                                 sparams::SimpleVector,
                                 world::UInt = get_world_counter(interp),
                                 )
    maybe_init_cache!(cache_key(interp))

    mi = specialize_method(m, atype, sparams)

    result = InferenceResult(mi)

    frame = InferenceState(result, #= cached =# true, interp)

    typeinf(interp, frame)

    return interp, frame
end

# exports

export
    XInterpreter,
    native,
    cache_key,
    with

end # module XCompiler
