from yahpo_gym import local_config
from yahpo_gym import benchmark_set
import yahpo_gym.benchmarks.lcbench


local_config.init_config()
local_config.set_data_path("/home/yahpo_data-1.0")

# Select a Benchmark
bench = benchmark_set.BenchmarkSet("lcbench")
# List available instances
bench.instances
# Set an instance
bench.set_instance("189908")
# Sample a point from the configspace (containing parameters for the instance and budget)
value = (
    bench.get_opt_space(drop_fidelity_params=True, seed=0)
    .sample_configuration(1)
    .get_dictionary()
)
value["epoch"] = 52
print(type(bench.config_space))
print(value)
# Computation time in seconds (s)
computation_time = bench.objective_function(value)[0]["time"]

# Power consumption in watts (W)
power_consumption = 150

# Energy consumption in watt-hours (Wh)
energy_consumption = power_consumption * (computation_time / 3600)

# Carbon intensity in kilograms of CO2 per kilowatt-hour (kgCO2/kWh)
carbon_intensity = 0.347

# Convert energy consumption to kilowatt-hours (kWh)
energy_consumption_kwh = energy_consumption / 1000

# Calculate carbon emissions in kilograms of CO2 (kgCO2)
carbon_emissions = energy_consumption_kwh * carbon_intensity

# print(f"Estimated carbon emissions: {carbon_emissions} kgCO2")
