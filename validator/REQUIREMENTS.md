# SN44 Validator System Requirements

This document outlines the system requirements for running an SN44 validator node, based on its operational responsibilities of managing challenges, evaluating responses, and maintaining subnet weights.

## System Requirements

### Minimum Requirements

- **CPU**: 4+ cores
- **RAM**: 8GB
- **Storage**: 50GB SSD
- **Network**: 100Mbps, stable connection
- **OS**: Ubuntu 20.04+ or macOS 12+

### Recommended Requirements

- **CPU**: 8+ cores
- **RAM**: 16GB
- **Storage**: 200GB NVMe SSD
- **Network**: 1Gbps, stable connection
- **OS**: Ubuntu 22.04 LTS

## Operational Requirements

The validator performs several concurrent operations that influence hardware needs:

### 1. Challenge Management

- Distributes challenges every 5 minutes
- Manages concurrent connections with multiple miners
- Requirements:
  - Stable network connection
  - Sufficient bandwidth for video distribution
  - Low latency for miner communication

### 2. Response Processing

- Handles concurrent response collection
- Stores responses in SQLite database
- Requirements:
  - Fast storage I/O
  - Sufficient RAM for concurrent operations
  - Network capacity for receiving responses

### 3. Response Evaluation

- Processes frame evaluations using OpenAI's GPT-4o
- Manages concurrent API requests
- Requirements:
  - Stable internet connection
  - RAM for response processing
  - Fast CPU for preprocessing

### 4. Weight Management

- Updates weights every 21 minutes
- Maintains subnet state
- Requirements:
  - Reliable network connection
  - Fast storage for database operations
  - Sufficient CPU for calculations

## Performance Considerations

### Storage Requirements

- **Database**: ~50MB per day (varies with miner count)
- **Challenge Cache**: ~1GB (temporary storage)
- **System**: ~10GB
- **Total**: Minimum 50GB, Recommended 200GB

### Memory Usage

- **Challenge Distribution**: ~100MB per concurrent challenge
- **Response Processing**: ~500MB per concurrent response
- **Evaluation**: ~500MB for processing
- **Database Operations**: ~500MB
- **System Operations**: ~1GB
- **Total Active**: 4-8GB typical usage

### Network Requirements

- **Inbound**: 100Mbps minimum (for response collection)
- **Outbound**: 100Mbps minimum (for challenge distribution)
- **Monthly Traffic**: ~500GB-1TB (varies with miner count)
- **Latency**: <100ms to major networks recommended

## Hardware Recommendations

### Production Setup

- **CPU**: AMD Ryzen 7 or Intel i7 (8+ cores)
- **RAM**: 32GB DDR4
- **Storage**: 500GB NVMe SSD
- **Network**: 1Gbps dedicated line
- **Backup**: Secondary SSD for database backups

### Minimum Viable Setup

- **CPU**: AMD Ryzen 5 or Intel i5 (6+ cores)
- **RAM**: 16GB DDR4
- **Storage**: 200GB SSD
- **Network**: 100Mbps stable connection

### Development/Testing

- **CPU**: 4+ cores
- **RAM**: 8GB
- **Storage**: 50GB SSD
- **Network**: Standard broadband

## Cost Considerations

### Cloud Hosting (Monthly Estimates)

- **Basic**: $50-100 (minimum viable setup)
- **Standard**: $100-200 (recommended setup)
- **Premium**: $200-400 (production setup)

### Operating Costs

- **Network**: 500GB-1TB bandwidth monthly
- **API**: OpenAI GPT-4o usage (~$0.01 per evaluation)
- **Storage**: Growing at ~1.5GB per day
- **Backup**: Consider additional storage costs

## Notes

1. **Scaling Factors**

   - Number of active miners
   - Challenge frequency
   - Evaluation batch size

2. **Critical Components**

   - Database performance
   - Network reliability
   - API rate limits
   - Storage I/O speed

3. **Monitoring Recommendations**
   - CPU usage and load
   - Memory utilization
   - Network throughput
   - Storage capacity and I/O
   - API usage and costs

For optimal operation, we recommend using dedicated hardware or high-performance cloud instances with priority on storage I/O and network stability.
