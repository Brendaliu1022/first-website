using Pkg
Pkg.activate("D:/MATLAB/Julia_control")
Pkg.instantiate()

using Clarabel
using GLMakie
using JuMP
using LinearAlgebra
using Random

# Parameters
A = [1.0 0.2; 0.0 1.0]
B = [0.02 0.0; 0.2 0.0]
D = [0.4 0.0; 0.4 0.6]
N = 30
n = 2
p = 2
# Boundary cond
μ_0 = [30.0, -5.0]
μ_N = [0.0, 0.0]
Σ_0 = [5.0 -1.0; -1.0 1.0]
Σ_N = [0.5 -0.4; -0.4 2.0]
# Cost function parameters 
Q = 0.5 * I(2)
R = 1.0
# optimization model, covariance steering SDP
model = Model(Clarabel.Optimizer)
# Decision variables
@variable(model, μ[1:n, 1:N])
@variable(model, Σ[1:n, 1:n, 1:N])
@variable(model, v[1:p, 1:N-1])
@variable(model, U[1:p, 1:n, 1:N-1])
@variable(model, Y[1:p, 1:p, 1:N-1])

# Objective function
@objective(model, Min, 
    sum(tr(Q * Σ[:,:,k]) for k in 1:N) + 
    sum(tr(R * Y[:,:,k]) for k in 1:N-1) +
    sum(μ[:,k]' * Q * μ[:,k] for k in 1:N) + 
    sum(v[:,k]' * R * v[:,k] for k in 1:N-1)) 

# Mean dynamics constraints
for k in 1:N-1
    @constraint(model, μ[:,k+1] .== A * μ[:,k] + B * v[:,k])
end

# Covariance dynamics constraints
for k in 1:N-1
    cov_next = A * Σ[:,:,k] * A' + B * U[:,:,k] * A' + A * U[:,:,k]' * B' + B * Y[:,:,k] * B' + D * D'
    @constraint(model, Σ[:,:,k+1] .== cov_next)
end

# LMI constraints
for k in 1:N-1
    @constraint(model, [Σ[:,:,k] U[:,:,k]'; U[:,:,k] Y[:,:,k]] in PSDCone())
end

# Boundary conditions
@constraint(model, μ[:,1] .== μ_0)
@constraint(model, μ[:,N] .== μ_N)
@constraint(model, Σ[:,:,1] .== Σ_0)
@constraint(model, Σ[:,:,N] .== Σ_N)

# Just optimize 
optimize!(model)
@show termination_status(model)
# Extract solution
μ_opt = value.(μ)
Σ_opt = value.(Σ)
v_opt = value.(v)
U_opt = value.(U)

#= Feedback gains 
K_k = U_k Σ_k^{-1} =#
K_opt = zeros(p, n, N-1)
for k in 1:N-1
    K_opt[:,:,k] = U_opt[:,:,k] * inv(Σ_opt[:,:,k])
end

# Sample paths 
num_samples = 10
samples = zeros(n, N, num_samples)

# Ini cond from Gaussian distribution
Random.seed!(123) # remain the same number in the following 

for i in 1:num_samples
    # Sample from init Gaussian distribution N(μ_0, Σ_0)
    samples[:, 1, i] = μ_0 + cholesky(Hermitian(Σ_0)).L * randn(2)
end

# Propagate, using the optimal control law with process noise
for i in 1:num_samples
    for k in 1:N-1
        # Control law
        # u_k = K_k (x_k - μ_k) + v_k
        δx = samples[:, k, i] - μ_opt[:,k]
        u = K_opt[:,:,k] * δx + v_opt[:,k]
        
        # Process noise
        process_noise = cholesky(Hermitian(D * D')).L * randn(2)
        # Next state
        samples[:, k+1, i] = A * samples[:, k, i] + B * u + process_noise
    end
end

# Visualize
fig = Figure(size=(1200, 900))
# Main trajectory plot for SDP approach
ax1 = Axis(fig[1, 1], 
    title="Ten sample paths with optimal state mean and three-standard-deviation tolerance region by SDP approach",
    xlabel="x₁", 
    ylabel="x₂")

# Colors for sample paths
colors = [:red, :blue, :green, :orange, :purple, :brown, :pink, :gray, :olive, :cyan]

# Plot uncertainty boundaries at EVERY time step

for k in 1:N
    # Compute ellipse points for tolerance region
    F = eigen(Hermitian(Σ_opt[:,:,k]))
    θ = range(0, 2π, length=80)
    ellipse_x = zeros(length(θ))
    ellipse_y = zeros(length(θ))
    for (idx, θ_i) in enumerate(θ)
        point = 3.0 * F.vectors * [cos(θ_i) * sqrt(F.values[1]), sin(θ_i) * sqrt(F.values[2])]
        ellipse_x[idx] = μ_opt[1,k] + point[1]
        ellipse_y[idx] = μ_opt[2,k] + point[2]
    end
    
    # Start and end 
    if k == 1
        # start point red 
        ellipse_color = :red
        line_style = :dash
        line_width = 2.5
        label_text = "Initial Boundary (t=1)"
    elseif k == N
        # end point blue 
        ellipse_color = :blue
        line_style = :dash
        line_width = 2.5
        label_text = "Final Boundary (t=$N)"
    else
        # paths in the way 
        ellipse_color = :green
        line_style = :solid
        line_width = 1
        opacity = 1
        label_text = ""
    end
    
    # Plot the uncertainty ellipse
    if k == 1 || k == N
        lines!(ax1, ellipse_x, ellipse_y, color=ellipse_color, linewidth=line_width, 
               linestyle=line_style,label=label_text)
    else
        lines!(ax1, ellipse_x, ellipse_y, color=(ellipse_color, opacity), 
               linewidth=line_width, linestyle=line_style, 
               label=k == 2 ? "Intermediate Boundaries" : "")
    end
end

# Plot sample paths
for i in 1:num_samples
    lines!(ax1, samples[1,:,i], samples[2,:,i], color=colors[i], linewidth=1.5,
           label=i == 1 ? "Sample Paths" : "")
    scatter!(ax1, [samples[1,1,i]], [samples[2,1,i]], color=colors[i], markersize=6)
    scatter!(ax1, [samples[1,end,i]], [samples[2,end,i]], color=colors[i], markersize=6, marker=:diamond)
end

# Plot optimal mean trajectory
lines!(ax1, μ_opt[1,:], μ_opt[2,:], color=:black, linewidth=3, label="Optimal Mean")

# Plot initial and target means
scatter!(ax1, [μ_0[1]], [μ_0[2]], color=:red, markersize=10, label="Initial Mean")
scatter!(ax1, [μ_N[1]], [μ_N[2]], color=:blue, markersize=10, label="Target Mean")

# Add a simple legend
legend_elements = [
    [LineElement(color = :black, linewidth = 3)],  # Mean trajectory
    [MarkerElement(color = :red, marker = :circle, markersize = 8)],  # Initial mean
    [MarkerElement(color = :blue, marker = :diamond, markersize = 8)],  # Target mean
    [LineElement(color = :red, linestyle = :dash, linewidth = 2.5)],  # Initial 3σ boundary
    [LineElement(color = :blue, linestyle = :dash, linewidth = 2.5)],  # Final 3σ boundary
    [LineElement(color = :green, linewidth = 1)],  # Intermediate 3σ boundaries
    [LineElement(color = :red, linewidth = 1.5)],  # Sample paths
    ]

legend_labels = [
    "Optimal Mean Trajectory",
    "Initial Mean",
    "Target Mean", 
    "Initial 3σ Boundary (t=1)",
    "Final 3σ Boundary (t=$N)",
    "Intermediate 3σ Boundaries",
    "Sample Paths"]

Legend(fig[1, 2], legend_elements, legend_labels, "Legend", framevisible = true)

# State history plots
ax2 = Axis(fig[2, 1], 
    title="State Component x₁ History",xlabel="Time Step", ylabel="x₁")

ax3 = Axis(fig[2, 2], 
    title="State Component x₂ History", xlabel="Time Step", ylabel="x₂")

time_steps = 1:N
for i in 1:num_samples
    lines!(ax2, time_steps, samples[1,:,i], color=(colors[i], 0.5))
    lines!(ax3, time_steps, samples[2,:,i], color=(colors[i], 0.5))
end

# Plot mean trajectory
lines!(ax2, time_steps, μ_opt[1,:], color=:black, linewidth=2, label="Mean")
lines!(ax3, time_steps, μ_opt[2,:], color=:black, linewidth=2, label="Mean")

# Add 3σ bounds
σ1 = [sqrt(Σ_opt[1,1,k]) for k in 1:N]
σ2 = [sqrt(Σ_opt[2,2,k]) for k in 1:N]
band!(ax2, time_steps, μ_opt[1,:] - 3σ1, μ_opt[1,:] + 3σ1, color=(:green, 0.1))
band!(ax3, time_steps, μ_opt[2,:] - 3σ2, μ_opt[2,:] + 3σ2, color=(:green, 0.1))

# Control and covariance evolution plots
ax4 = Axis(fig[3, 1], 
    title="Mean Control History", xlabel="Time Step", ylabel="v₁")

ax5 = Axis(fig[3, 2], 
    title="Covariance Norm Evolution", xlabel="Time Step", ylabel="‖Σ‖")

# Plot mean control (first component)
control_steps = 1:N-1
lines!(ax4, control_steps, v_opt[1,:], color=:orange, linewidth=2)

# Plot covariance norm evolution
cov_norms = [norm(Σ_opt[:,:,k]) for k in 1:N]
lines!(ax5, time_steps, cov_norms, color=:purple, linewidth=2)
scatter!(ax5, [1, N], [norm(Σ_0), norm(Σ_N)], color=[:red, :blue], markersize=8)

# Display optimal cost
optimal_cost = objective_value(model)
Label(fig[0, :], 
    "Optimal Covariance Control - SDP Approach (Cost: $(round(optimal_cost, digits=4)))", 
    fontsize=16)

display(fig)

# Print optimization results
println("\n" * "="^80)
println("COVARIANCE STEERING WITH UNCERTAINTY BOUNDARIES AT EVERY TIME STEP")
println("="^80)
println("\nVisualization Features:")
println("• RED dashed ellipse: Initial 3σ uncertainty boundary (t=1)")
println("• BLUE dashed ellipse: Final 3σ uncertainty boundary (t=$N)")
println("• Green ellipses: Intermediate 3σ uncertainty boundaries")
println("• Each ellipse represents the 99.7% confidence region at that time")

println("\nUncertainty Evolution Analysis:")
for k in [1,10, 20, 25, N]
    σ1 = sqrt(Σ_opt[1,1,k])
    σ2 = sqrt(Σ_opt[2,2,k])
    ellipse_area = π * 3σ1 * 3σ2
    println("Time $k: σ₁ = $(round(σ1, digits=3)), σ₂ = $(round(σ2, digits=3)), " *
            "3σ ellipse area = $(round(ellipse_area, digits=3))")
end

println("\nControl Performance:")
println("Optimal cost: ", round(optimal_cost, digits=6))
println("Final mean: ", round.(μ_opt[:,end], digits=4))
println("Target mean: ", μ_N)
println("Final mean error: ", round(norm(μ_opt[:,end] - μ_N), digits=6))
println("Final covariance norm: ", round(norm(Σ_opt[:,:,end]), digits=4))
println("Target covariance norm: ", round(norm(Σ_N), digits=4))
println("Final covariance error: ", round(norm(Σ_opt[:,:,end] - Σ_N), digits=6))
println("Mean control norm: ", round(norm(v_opt), digits=4))
println("="^80)
