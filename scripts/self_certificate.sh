#!/bin/bash
# Get the directory where this script lives
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Add it to PATH, only for this script’s runtime
MAIN_DIR=$(dirname "$SCRIPT_DIR")
KEY_DIR="$MAIN_DIR/key/"
export PATH="$MAIN_DIR:$PATH"

if [[ -e "$1" && -e "$2" ]]; then
    key="$1"
    cert="$2"
else
    key="$KEY_DIR/telemedicine-key.pem"
    cert="$KEY_DIR/telemedicine-cert.pem"
fi

openssl req -x509 -newkey rsa:4096 -keyout $key -out $cert -days 365 -nodes
