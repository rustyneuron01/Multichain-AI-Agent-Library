# SN44 Miner System Requirements

This document outlines the minimum and recommended system requirements for running an SN44 miner node, along with performance expectations for different hardware configurations.

## System Requirements

### Minimum Requirements

- **CPU**: 4+ cores
- **RAM**: 8GB
- **Storage**: 20GB SSD
- **Network**: 100Mbps, stable connection
- **OS**: Ubuntu 20.04+ or macOS 12+

### Recommended Requirements

- **GPU**: NVIDIA RTX 3060 12GB or better
- **CPU**: 8+ cores
- **RAM**: 16GB
- **Storage**: 50GB NVMe SSD
- **Network**: 1Gbps, stable connection
- **OS**: Ubuntu 22.04 LTS

## Performance Benchmarks

Below are real-world performance metrics for different hardware configurations processing soccer video analysis:

### CPU Only (e.g., DigitalOcean 8GB Droplet)

- **Processing Time**: ~9,138 seconds per challenge
- **Average FPS**: 0.08
- **Viability**: Not recommended for production
- **Cost Efficiency**: Low

### Apple Silicon (MacBook Pro M2)

- **Processing Time**: ~140 seconds per challenge
- **Average FPS**: 3.49
- **Viability**: Suitable for testing/development
- **Cost Efficiency**: Medium

### NVIDIA GPU (RTX 4090)

- **Processing Time**: ~22 seconds per challenge
- **Average FPS**: 22.36
- **Viability**: Ideal for production
- **Cost Efficiency**: High

## Hardware Recommendations

### Best Performance (Production)

- NVIDIA RTX 4090 24GB
- 32GB RAM
- 12+ core CPU
- NVMe SSD

### Good Performance (Budget)

- NVIDIA RTX 3060 12GB
- 16GB RAM
- 8+ core CPU
- SSD

### Minimum Viable (Testing)

- Apple M1/M2
- 16GB RAM
- SSD

### Not Recommended

- CPU-only systems
- Systems with <8GB RAM
- Systems without SSD storage

## Notes

1. **Processing Time Impact**

   - CPU-only: ~152 minutes per challenge
   - Apple Silicon: ~2.3 minutes per challenge
   - NVIDIA GPU: ~22 seconds per challenge

2. **Cost Considerations**

   - GPU solutions offer best performance/cost ratio
   - CPU-only solutions are not cost-effective due to extremely low throughput
   - Cloud GPU instances may be more cost-effective than CPU-only solutions

3. **Scaling Considerations**

   - Multiple GPUs are supported but not required
   - Memory usage scales with batch size
   - Storage requirements may increase over time

4. **Network Requirements**
   - Stable connection is crucial for challenge downloads
   - Bandwidth affects video download times
   - Low latency improves validator communication

For optimal performance and reliability in a production environment, we strongly recommend using a system with a modern NVIDIA GPU.
