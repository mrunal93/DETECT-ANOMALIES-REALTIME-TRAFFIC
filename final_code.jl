using CSV, DataFrames, Dates, Flux, Statistics, ConformalPrediction
using Flux: @epochs, mae, mse, throttle, Optimisers
using Plots
function import_and_convert(file_path,columnname)
    df = DataFrame(CSV.File(file_path)) 

    rename!(df, :value => columnname)
end

dc1 = import_and_convert(raw"C:\Users\mrn26\Documents\DBS\Machine_learning\Individual_Project\NAB-master\data\realTraffic\occupancy_6005.csv","occupancy")
dc2 = import_and_convert(raw"C:\Users\mrn26\Documents\DBS\Machine_learning\Individual_Project\NAB-master\data\realTraffic\occupancy_t4013.csv","occupancy")
ds1 = import_and_convert(raw"C:\Users\mrn26\Documents\DBS\Machine_learning\Individual_Project\NAB-master\data\realTraffic\speed_6005.csv","speed")
ds2 = import_and_convert(raw"C:\Users\mrn26\Documents\DBS\Machine_learning\Individual_Project\NAB-master\data\realTraffic\speed_7578.csv","speed")
df3 = import_and_convert(raw"C:\Users\mrn26\Documents\DBS\Machine_learning\Individual_Project\NAB-master\data\realTraffic\speed_t4013.csv","speed")
dt1 = import_and_convert(raw"C:\Users\mrn26\Documents\DBS\Machine_learning\Individual_Project\NAB-master\data\realTraffic\TravelTime_387.csv","TravelTime")
dt2 = import_and_convert(raw"C:\Users\mrn26\Documents\DBS\Machine_learning\Individual_Project\NAB-master\data\realTraffic\TravelTime_451.csv","TravelTime")

merge_dc = vcat(dc1,dc2)
merge_ds = vcat(ds1,ds2,df3)
merge_dt =vcat(dt1,dt2)


combine = outerjoin(merge_dc,merge_ds,merge_dt, on = :timestamp)

CSV.write("output.csv", combine)
# 1. Data Loading and Preprocessing
# ==================================
# df = CSV.read("output_with_predictions.csv", DataFrame)
df = CSV.read("output.csv", DataFrame)
df.timestamp = DateTime.(df.timestamp, dateformat"dd-mm-yyyy HH:MM")
# date_format = DateFormat("y-m-d H:M:S")

# df[!, :timestamp] = DateTime.(df[!, :timestamp], date_format)
# Clean data
dropmissing!(df)
unique!(df, :timestamp)
sort!(df, :timestamp)
function calculate_stats(df, col)
    return (
        Mean = round(mean(df[!, col]), digits=1),
        Min = minimum(df[!, col]),
        Max = maximum(df[!, col]),
        Std_Dev = round(std(df[!, col]), digits=1)
    )
end

# Create summary table
summary_df = DataFrame(
    Column = ["occupancy", "speed", "TravelTime"],
    Mean = [calculate_stats(df, col).Mean for col in [:occupancy, :speed, :TravelTime]],
    Min = [calculate_stats(df, col).Min for col in [:occupancy, :speed, :TravelTime]],
    Max = [calculate_stats(df, col).Max for col in [:occupancy, :speed, :TravelTime]],
    Std_Dev = [calculate_stats(df, col).Std_Dev for col in [:occupancy, :speed, :TravelTime]]
)
# 2. Feature Engineering
# ======================
function create_sequences(X, y, window_size=24)
    X_seq = [X[i:i+window_size-1, :] for i in 1:size(X,1)-window_size]
    y_seq = y[window_size+1:end]
    return cat(X_seq..., dims=3), y_seq
end

# 3. Data Preparation
# ===================
features = [:occupancy, :speed]
target = :TravelTime

# Convert to arrays
X = Matrix(df[:, features])
y = Vector(df[:, target])

# Normalization
function minmax_scale(x)
    min, max = extrema(x)
    return (x .- min) ./ (max - min), min, max
end

X_scaled, X_min, X_max = minmax_scale(X)
y_scaled, y_min, y_max = minmax_scale(y)

# Create sequences
window_size = 24
X_seq, y_seq = create_sequences(X_scaled, y_scaled, window_size)

# 4. Conformal Prediction Split
# =============================
n = size(X_seq, 3)
split_ratio = (0.6, 0.2, 0.2)
train_idx = 1:floor(Int, n*split_ratio[1])
cal_idx = last(train_idx)+1:last(train_idx)+floor(Int, n*split_ratio[2])
test_idx = last(cal_idx)+1:n

X_train, y_train = X_seq[:, :, train_idx], y_seq[train_idx]
X_cal, y_cal = X_seq[:, :, cal_idx], y_seq[cal_idx]
X_test, y_test = X_seq[:, :, test_idx], y_seq[test_idx]

# 5. LSTM Model Architecture
# ==========================
model = Chain(
    LSTM(size(X_train, 1), 40),
    # Dropout(0.2),
    # LSTM(35, 17),
    x -> x[:, end, :],  
    # Dense(17, 8),
    Dense(40, 1),
    vec
)

# 6. Model Training
# =================
function inverse_scale(scaled, min, max)
    return scaled .* (max - min) .+ min
end

opt = Optimisers.Adam()
state = Optimisers.setup(opt, model)

# Early stopping setup
best_loss = Inf
patience = 10
counter = 0

# Training loop
for epoch in 1:500
    Flux.train!(model, [(X_train, y_train)], state) do m, x, y
        Flux.mse(m(x), y)
    end
    
    # Validation loss
    current_loss = Flux.mse(model(X_cal), y_cal)
    
    # Early stopping
    if current_loss < best_loss
        best_loss = current_loss
        counter = 0
    else
        counter += 1
    end
    
    epoch % 10 == 0 && println("Epoch $epoch: Val Loss = $(round(current_loss, digits=4))")
    counter >= patience && break
end

# 7. Conformal Prediction Setup
# ==============================
# Get calibration predictions
y_cal_pred = model(X_cal)

# Calculate residuals
residuals = abs.(y_cal .- y_cal_pred)

# Get conformal quantile (95% confidence)
α = 0.05
q = quantile(residuals, 1 - α)

# 8. Anomaly Detection
# ====================
# Test predictions
y_test_pred = model(X_test)

# Create prediction intervals
lower = y_test_pred .- q
upper = y_test_pred .+ q

# Inverse scaling
y_test_actual = inverse_scale(y_test, y_min, y_max)
lower_actual = inverse_scale(lower, y_min, y_max)
upper_actual = inverse_scale(upper, y_min, y_max)
y_test_pred_actual = inverse_scale(y_test_pred, y_min, y_max)
rmse = sqrt(mean((y_test_actual .- y_test_pred_actual).^2))
# Detect anomalies
anomalies = (y_test_actual .< lower_actual) .| (y_test_actual .> upper_actual)
anomaly_indices = findall(anomalies)
anomaly_scores = y_test_actual[anomalies] - [y_test_actual[i] > upper_actual[i] ? upper_actual[i] : lower_actual[i] for i in anomaly_indices]

# 9. Visualization
# ================
# Get timestamps
test_timestamps = df.timestamp[window_size+1:end][test_idx]

# Create plot
p = plot(test_timestamps, y_test_actual, 
    label="Actual Travel Time", linewidth=2, legend=:topleft)
plot!(test_timestamps, y_test_pred_actual, 
    label="Predicted", linewidth=2)
plot!(test_timestamps, lower_actual, 
    label="95% Lower Bound", color=:green, linestyle=:dash)
plot!(test_timestamps, upper_actual, 
    label="95% Upper Bound", color=:green, linestyle=:dash)
scatter!(test_timestamps[anomaly_indices], y_test_actual[anomalies], 
    label="Anomalies", color=:red, markersize=5)

title!("Travel Time Anomaly Detection with Conformal Prediction")
xlabel!("Timestamp") 
ylabel!("Travel Time (seconds)")
savefig("conformal_anomalies_Double_50.png")

# 10. Results Analysis
# ====================
# Create detailed results table
timestamp_dict = Dict{DateTime, NamedTuple}()
for row in eachrow(df)
    timestamp_dict[row.timestamp] = (
        speed = row.speed,
        occupancy = row.occupancy
    )
end


# Create results dataframe
results = DataFrame(
    Timestamp = test_timestamps[anomaly_indices],
    Speed = [timestamp_dict[ts].speed for ts in test_timestamps[anomaly_indices]],
    Occupancy = [timestamp_dict[ts].occupancy for ts in test_timestamps[anomaly_indices]],
    Actual_TravelTime = y_test_actual[anomalies],
    Predicted_TravelTime = y_test_pred_actual[anomalies],
    Lower_Bound = lower_actual[anomalies],
    Upper_Bound = upper_actual[anomalies],
    Deviation = anomaly_scores
)



println("\nDetected ", nrow(results), " anomalous events:")
show(results, allrows=true, allcols=true)

# Save results
CSV.write("anomaly_results.csv", results)


