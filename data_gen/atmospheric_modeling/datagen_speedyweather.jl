import Distributed: @distributed

@everywhere using SpeedyWeather
@everywhere using Printf

n_samples = 1200
spectral_grid = SpectralGrid(trunc=63,nlev=1,Grid=FullClenshawGrid)
orography = EarthOrography(spectral_grid,smoothing=false)
implicit = SpeedyWeather.ImplicitShallowWater(spectral_grid,α=0.5)
output = OutputWriter(spectral_grid,ShallowWater,output_dt=1/4,output_vars=[:u,:v,:pres,:orography])

@sync @distributed for sample in 1:n_samples
    output.id = @sprintf("%04d",sample)
    initial_conditions = SpeedyWeather.RandomWaves()
    model = ShallowWaterModel(;spectral_grid,orography,output,initial_conditions,implicit);
	simulation = initialize!(model);
	run!(simulation,n_days=1/2,output=true)
end