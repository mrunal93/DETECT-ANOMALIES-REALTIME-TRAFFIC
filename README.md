# DETECT-ANOMALIES-REALTIME-TRAFFIC

# Traffic Anomaly Detection with LSTM & Conformal Prediction 🚦📊

![Julia](https://img.shields.io/badge/Julia-1.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)

A Julia-based machine learning project to detect unusual traffic patterns using **LSTM networks** and **Conformal Prediction**. Built for reliability and interpretability in time-series anomaly detection.

---

## 📌 Overview
This project analyzes real-world traffic data (speed, occupancy, travel time) to identify anomalies caused by traffic jams, accidents, or sensor errors. By combining **LSTM temporal modeling** with **statistically rigorous conformal prediction intervals**, the system flags deviations from expected patterns with 95% confidence.

**Dataset**: [Numenta Anomaly Benchmark (NAB) RealTraffic](https://github.com/numenta/NAB/tree/master/data/realTraffic)

---

## 🚀 Features
- **Data Preprocessing**: Merges multi-source CSVs, handles missing values, and normalizes features.
- **LSTM Model**: Built with Flux.jl to learn temporal traffic patterns.
- **Conformal Prediction**: Generates dynamic thresholds for anomaly detection.
- **Visualization**: Highlights anomalies against prediction intervals (e.g., actual travel time vs. predicted range).

---

## 🛠️ Installation
1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/traffic-anomaly-detection.git
   cd traffic-anomaly-detection
Install Julia 1.8+: Download from julialang.org.

Install dependencies:

julia
] add CSV DataFrames Dates Flux ConformalPrediction Plots
🧪 Usage
1. Data Preparation
Merge and clean raw CSV files:

julia
using CSV, DataFrames
df = CSV.read("raw_data/*.csv", DataFrame)
# Clean missing values, normalize, and split into sequences (see Methodology in Report.docx)
2. Model Training
Define and train the LSTM:

julia
using Flux
model = Chain(
  LSTM(2 => 50),  # 2 input features (speed, occupancy)
  Dense(50 => 1)  # Predict travel time
)
loss(x, y) = Flux.mse(model(x), y)
Flux.train!(loss, params(model), data, ADAM())
3. Conformal Prediction
Compute 95% prediction intervals:

julia
using ConformalPrediction
residuals = abs.(y_cal .- model(X_cal))
q = quantile(residuals, 0.95)
intervals = [y_pred - q, y_pred + q]
4. Visualization
Plot anomalies outside prediction bounds:

julia
using Plots
plot(actual_times, label="Actual Travel Time")
plot!(predicted_times, ribbon=(q, q), label="95% Interval")
📊 Results
Anomaly Detection: Identified outliers where actual travel time exceeded prediction intervals (e.g., 809s vs. predicted [76s, 336s]).

Performance: LSTM captured temporal trends, while conformal prediction added statistical robustness.

Prediction Intervals

🔑 Challenges & Future Work
Challenges: Handling abrupt traffic shifts, LSTM training time.

Future Improvements:

Replace LSTM with GRU for efficiency.

Integrate real-time data streams.

Dynamic threshold adjustments.

📚 References
LSTM Math Explained

Flux.jl Documentation

Conformal Prediction Theory

🤝 Contributing
Contributions are welcome! Open an issue or submit a PR for:

Performance optimizations

Additional datasets

Enhanced visualization tools

📧 Contact
For questions or collaborations, contact Mrunal Sunil Fulzel.

