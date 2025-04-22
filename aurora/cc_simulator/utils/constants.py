"""Constants for congestion control simulation."""

# Network constants
BYTES_PER_PACKET = 1500
BITS_PER_BYTE = 8
MAX_CWND = 5000
MAX_RATE = 20000  # packets per second
MIN_CWND = 2
MIN_RATE = 5  # packets per second
TCP_INIT_CWND = 10

# Event types
EVENT_TYPE_ACK = 'A'
EVENT_TYPE_SEND = 'S'

# RL constants
MI_RTT_PROPORTION = 1.0  # Monitoring interval proportion
REWARD_SCALE = 0.001

# Aurora constants
START_SENDING_RATE = 100  # packets per second
AURORA_ROUND = 0  # Whether to use round-based Aurora

# BBR constants
BBR_HIGH_GAIN = 2.89  # 2/ln(2)
BTLBW_FILTER_LEN = 10  # packet-timed round trips
RTPROP_FILTER_LEN = 10  # seconds
PROBE_RTT_DURATION = 0.2  # 200ms
BBR_MIN_PIPE_CWND = 4  # packets
BBR_GAIN_CYCLE_LEN = 8 