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
    typeinf, # entry
    typeinf_ext_toplevel # entry for native compilation

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

Keeps `CodeInstance` associated with `mi::MethodInstace`, which is a customized compilation
result of `mi` performed by `XInterpreter`.
The cache is partitioned by identities of each `XInterpreter`, and thus running a pipeline
with different configurations will yeild a different cache and not be influenced by the
previous compilation.
This cache should be separated from the `NativeInterpreter`'s global cache, but currently
code cache can be created in many different points of compilation pipeline, and thus a
customized compilation will pollute the native cache.
"""
const X_CODE_CACHE = IdDict{UInt, IdDict{MethodInstance,CodeInstance}}()

function maybe_init_cache!(cache_key::UInt)
    haskey(X_CODE_CACHE, cache_key) && return
    X_CODE_CACHE[cache_key] = IdDict{MethodInstance,CodeInstance}()
end
__clear_cache!() = empty!(X_CODE_CACHE)

function CC.code_cache(interp::XInterpreter)
    cache = XGlobalCache(interp)
    worlds = WorldRange(get_world_counter(interp))
    return WorldView(cache, worlds)
end

struct XGlobalCache{Interp<:XInterpreter}
    interp::Interp
end

# cache existence for this `analyzer` is ensured on its construction
x_code_cache(interp::XInterpreter) = X_CODE_CACHE[cache_key(interp)]
x_code_cache(wvc::WorldView{XGlobalCache{Interp}}) where Interp<:XInterpreter = x_code_cache(wvc.cache.interp)

native_code_cache(interp::XInterpreter) = CC.code_cache(native(interp))
native_code_cache(wvc::WorldView{XGlobalCache{Interp}}) where Interp<:XInterpreter = native_code_cache(wvc.cache.interp)

CC.haskey(wvc::WorldView{XGlobalCache{Interp}}, mi::MethodInstance) where Interp<:XInterpreter =
    haskey(x_code_cache(wvc), mi)

function CC.get(wvc::WorldView{XGlobalCache{Interp}}, mi::MethodInstance, default) where Interp<:XInterpreter
    r = get(x_code_cache(wvc), mi, default)
    r !== default && return r
    return CC.get(native_code_cache(wvc), mi, default)
end

function CC.getindex(wvc::WorldView{XGlobalCache{Interp}}, mi::MethodInstance) where Interp<:XInterpreter
    r = CC.get(wvc, mi, nothing)
    r === nothing && throw(KeyError(mi))
    return r::CodeInstance
end

function CC.setindex!(wvc::WorldView{XGlobalCache{Interp}}, ci::CodeInstance, mi::MethodInstance) where Interp<:XInterpreter
    x_code_cache(wvc)[mi] = ci
    add_x_callback!(mi) # register the callback on invalidation
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
    x_typeinf_ext_toplevel(mi::MethodInstance, world::UInt) =
        typeinf_ext_toplevel(interp, mi)

    try
        ccall(:jl_set_typeinf_func, Cvoid, (Any,), x_typeinf_ext_toplevel)
        return f()
    catch err
        rethrow(err)
    finally
        ccall(:jl_set_typeinf_func, Cvoid, (Any,), typeinf_ext_toplevel)
    end
end

# exports

export
    XInterpreter,
    native,
    cache_key,
    with

end # module XCompiler
