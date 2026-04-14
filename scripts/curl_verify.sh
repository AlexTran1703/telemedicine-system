# Get the directory where this script lives
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Add it to PATH, only for this script’s runtime
MAIN_DIR=$(dirname "$SCRIPT_DIR")
export PATH=$MAIN_DIR:$PATH

URL="https://localhost:5000"
# Function to generate 2000 random samples
generate_random_samples() {
    samples=""
    for i in {1..2000}; do
        # Generate a random number and append it to the sample string with a comma
        samples+="$((RANDOM % 101)),"
    done
    # Remove the trailing comma from the string
    samples=${samples%,}
    # Enclose the samples in square brackets to make it a valid JSON array
    echo "[$samples]"
}

# Function to send data to the server
parsing_json() {
    # Read file and convert to comma-separated values
    files="$MAIN_DIR/scripts/test_ecg/ecg_1.txt"
    samples=$(awk 'BEGIN {ORS=","} {print $1}' $files | sed 's/,$//')
    # Wrap in JSON object
    json=$(
        cat <<EOF
{
  "device_id": "_device_1",
  "UUID": 10,
  "samples": [$samples],
  "sampling_rate": 100
}
EOF
    )

    # Optionally print to check
    echo "$json"
}
send_data() {
    #     local samples=$1
    #     json_data=$(cat <<EOF
    # {
    #   "name": "Device 1",
    #   "UUID": 10,
    #   "samples": $samples
    # }
    # EOF
    # )
    json_data=$(parsing_json)
    url="https://localhost:5000/api/data"

    # Send POST request to the server using curl
    response=$(curl --insecure  --write-out "%{http_code}" -X POST "$url" -H "Content-Type: application/json" -d "$json_data")

    # Check the response code
    if [[ "$response" == "200" ]]; then
        echo "Response from server: Valid JSON received!"
    else
        echo "Error: $response"
    fi
}

verity_cert() {
    curl -k $URL/cert-fingerprint
}

# Parse command-line options
while getopts "sv" opt; do
    case $opt in
    s)
        # Generate the samples
        samples=$(generate_samples)
        send_data "$samples"
        ;;
    v)
        #verify certificate
        verity_cert
        ;;
    *) ;;
    esac
done
