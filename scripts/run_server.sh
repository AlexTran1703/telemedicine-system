# Get the directory where this script lives
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Add it to PATH, only for this script’s runtime
MAIN_DIR=$(dirname "$SCRIPT_DIR")
export PATH=$MAIN_DIR:$PATH

KEY_DIR="$MAIN_DIR/key/"
# Define the file paths
key="$KEY_DIR/telemedicine-key.pem"
cert="$KEY_DIR/telemedicine-cert.pem"

# Check if either of the two files doesn't exist
if [[ ! -e "$key" || ! -e "$cert" ]]; then
    echo "Create CA"
    pushd $MAIN_DIR
    mkdir -p key
    ./scripts/self_certificate.sh $key $cert
    popd
else
    echo "CA is already existed"
fi

# Example: now you can call executables in that directory
echo "Starting Server"
(pushd $MAIN_DIR && python3 $MAIN_DIR/server/main.py)