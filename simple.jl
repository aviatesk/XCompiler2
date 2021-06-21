# adapted from https://gist.github.com/Keno/d0c2df947f67be543036238a0caeb1c6

let counter = 0
	global get_fast_sin_counter, reset_counter
	fast_sin(x::Float64) = (counter += 1; sin(x::Float64))
	get_fast_sin_counter() = counter
	reset_counter() = (counter = 0)
end

get_sin() = sin

function my_func(x)
	reset_counter()
	sin(x)
	@test get_fast_sin_counter() == 1
	get_sin()(x)
	@test get_fast_sin_counter() == 2
	get_sin()(Base.inferencebarrier(x))
	@test get_fast_sin_counter() == 3
end

with(FastSinPlugin) do
	my_func(1.0)
end
