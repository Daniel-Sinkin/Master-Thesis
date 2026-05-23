module ParallelGradient

using Distributed

export @addprocs_and_everywhere, @everywhere_async

macro addprocs_and_everywhere(args...)
    return esc(:(nothing))
end

macro everywhere_async(args...)
    if length(args) == 1
        ex = args[1]
        return esc(:(Distributed.@everywhere $ex))
    elseif length(args) == 2
        procs = args[1]
        ex = args[2]
        return esc(quote
            for p in $procs
                Distributed.remotecall_eval(Main, p, $(QuoteNode(ex)))
            end
            nothing
        end)
    end
    error("@everywhere_async expects an expression or process list plus expression")
end

end
