function axpy!(z, α, x, y)
  @inbounds for k = 1:length(x)
    z[k] = α * x[k] + y[k]
  end
end

let
  N = 10000
  α = rand()
  h_x = rand(N)
  h_y = rand(N)
  h_z = similar(h_x)

  axpy!(h_z, α, h_x, h_y)

  @assert h_z == α * h_x + h_y
end
