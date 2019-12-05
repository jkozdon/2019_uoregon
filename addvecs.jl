using CuArrays
using CUDAnative
CuArrays.allowscalar(false)

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
      k = tid + (bid - 1) * nblocks
      k < N && (z[k] = α * x[k] + y[k])
    end
  end
end

function knl_axpy!(z, α, x, y)
  N = length(x)

  nblocks = blockDim().x
  bid = blockIdx().x
  tid = threadIdx().x

  k = tid + (bid - 1) * nblocks
  @inbounds k < N && (z[k] = α * x[k] + y[k])
end

let
  N = 10000
  α = rand()
  h_x = rand(N)
  h_y = rand(N)
  h_z = similar(h_x)

  axpy!(h_z, α, h_x, h_y)
  @assert h_z == α * h_x + h_y

  h_z .= 0
  nthreads = 256
  nblocks = fld1(N, nthreads)
  psuedo_knl_axpy!(h_z, α, h_x, h_y, nthreads, nblocks)

  d_x = CuArray(h_x)
  d_y = CuArray(h_y)
  d_z = similar(d_x)
  @cuda threads=nthreads blocks=nblocks knl_axpy!(h_z, α, h_x, h_y)

end
