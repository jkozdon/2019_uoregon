using CuArrays
using CUDAnative
CuArrays.allowscalar(false)
using LinearAlgebra
using BenchmarkTools
using GPUifyLoops

function axpy!(z, α, x, y)
  @inbounds for k = 1:length(x)
    z[k] = α * x[k] + y[k]
  end
end

function psuedo_knl_axpy!(z, α, x, y, nthreads, nblocks)
  N = length(x)
  @assert N <= nthreads * nblocks
  @inbounds for bid = 1:nblocks
    for tid = 1:nthreads
      k = tid + (bid - 1) * nthreads
      k <= N && (z[k] = α * x[k] + y[k])
    end
  end
end

function knl_axpy!(z, α, x, y)
  N = length(x)

  nthreads = blockDim().x
  bid = blockIdx().x
  tid = threadIdx().x

  k = tid + (bid - 1) * nthreads
  @inbounds k <= N && (z[k] = α * x[k] + y[k])
  nothing
end

function gpuify_axpy!(z, α, x, y, nblocks, nthreads)
  N = length(x)
  @inbounds @loop for boff in (0; (blockIdx().x-1) * blockDim().x)
    @loop for tid in (1:N; threadIdx().x)
      k = tid + boff
      k <= N && (z[k] = α * x[k] + y[k])
    end
  end
end

let
  N = 100000000
  α = rand()
  h_x = rand(N)
  h_y = rand(N)
  h_z = similar(h_x)


  h_z .= α .* h_x .+ h_y
  axpy!(h_z, α, h_x, h_y)
  @assert h_z == α * h_x + h_y

  println("broadcasted")
  display(@benchmark $h_z .= $α .* $h_x .+ $h_y)
  println("axpy")
  display(@benchmark axpy!($h_z, $α, $h_x, $h_y))

  h_z .= 0
  nthreads = 256
  nblocks = fld1(N, nthreads)
  psuedo_knl_axpy!(h_z, α, h_x, h_y, nthreads, nblocks)
  @assert h_z == α * h_x + h_y

  println("psuedo_knl_axpy!")
  display(@benchmark psuedo_knl_axpy!($h_z, $α, $h_x, $h_y, $nthreads, $nblocks))

  h_z .= 0
  nthreads = 256
  nblocks = fld1(N, nthreads)
  @launch CPU() threads=nthreads blocks=nblocks gpuify_axpy!(h_z, α, h_x, h_y)
  @assert h_z == α * h_x + h_y

  println("gpuify (CPU)")
  display(@benchmark @launch CPU() threads=$nthreads blocks=$nblocks gpuify_axpy!($h_z, $α, $h_x, $h_y))

  #=
  d_x = CuArray(h_x)
  d_y = CuArray(h_y)
  d_z = similar(d_x)
  display(@benchmark CuArrays.@sync @cuda threads=$nthreads blocks=$nblocks knl_axpy!($d_z, $α, $d_x, $d_y))
  @assert isapprox(Array(d_z), α * h_x + h_y, norm=(x)->norm(x, Inf))


  d_z .= 0
  nthreads = 256
  nblocks = fld1(N, nthreads)
  @launch CPU() threads=nthreads blocks=nblocks gpuify_axpy!(d_z, α, d_x, d_y)
  @assert isapprox(Array(d_z), α * h_x + h_y, norm=(x)->norm(x, Inf))

  println("gpuify (GPU)")
  display(@benchmark @launch CPU() threads=$nthreads blocks=$nblocks gpuify_axpy!($d_z, $α, $d_x, $d_y))
  =#
end
