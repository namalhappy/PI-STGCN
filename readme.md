# PI-STGCN: Physics-Informed Spatio-Temporal Graph Convolutional Network for River Discharge Forecasting

PI-STGCN is a physics-informed deep learning model for multi-step river discharge forecasting. The model combines spatio-temporal graph convolutional learning with river cascade routing constraints to improve both prediction accuracy and hydrological interpretability.

## Overview

River discharge forecasting is important for flood risk management, water resource planning, and hydrological monitoring. Traditional data-driven models can learn temporal patterns, but they often ignore the physical connectivity between upstream and downstream river stations.

PI-STGCN addresses this by combining:

- **Spatial learning** using graph convolutional networks
- **Temporal learning** using temporal convolutional layers
- **Physics-informed learning** using cascade routing constraints
- **Multi-station forecasting** for river networks
- **Multi-step prediction** for future discharge values

## Model Architecture

The PI-STGCN model includes three main components:

1. **Graph Convolution Module**  
   Learns spatial relationships between river stations using a cascade-based adjacency matrix.

2. **Temporal Convolution Module**  
   Captures temporal discharge patterns from historical flow sequences.

3. **Physics-Informed Routing Loss**  
   Adds hydrological consistency by encouraging downstream predictions to follow upstream-to-downstream routing behavior.

